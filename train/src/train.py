'''
@File       :   train.py
@Time       :   2023/02/04 10:51:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   Train reward model.
'''

import os
from config.options import *
from config.utils import *
from config.learning_rates import get_learning_rate_scheduler
os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
opts.BatchSize = opts.batch_size * opts.accumulation_steps * opts.gpu_num

from rank_dataset import rank_dataset
from rank_pair_dataset import rank_pair_dataset
from ImageReward import ImageReward

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.backends import cudnn

import sys


import os
import sys
import logging
from datetime import datetime

def std_log():
    """
    标准日志设置函数。
    
    如果当前进程的排名为0（主进程），则创建日志文件路径，确保日志目录存在，并将标准输出重定向到日志文件和命令行中。
    """
    if get_rank() == 0:
        # 创建一个唯一的保存路径
        save_path = make_path()
        log_filename = os.path.join(config['log_base'], f"{save_path}.txt")

        # 确保日志基础目录存在，如果不存在则创建
        if not os.path.exists(config['log_base']):
            os.makedirs(config['log_base'])

        # 配置logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器，输出到日志文件
        file_handler = logging.FileHandler(log_filename, mode="w")
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器，输出到命令行
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 将处理器添加到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # 替代sys.stdout为logger
        sys.stdout = LoggerWriter(logger, logging.INFO)
        sys.stderr = LoggerWriter(logger, logging.ERROR)

class LoggerWriter:
    """
    将print语句重定向到logging的辅助类。
    """
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
        
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
            
    def flush(self):
        pass

# def std_log():
#     """
#     标准日志设置函数。
    
#     如果当前进程的排名为0（主进程），则创建日志文件路径，确保日志目录存在，并将标准输出重定向到日志文件中。
#     """
#     if get_rank() == 0:
#         # 创建一个唯一的保存路径
#         save_path = make_path()
#         # 确保日志基础目录存在，如果不存在则创建
#         makedir(config['log_base'])
#         # 将标准输出重定向到指定的日志文件
#         sys.stdout = open(os.path.join(config['log_base'], "{}.txt".format(save_path)), "w")


def init_seeds(seed, cuda_deterministic=True):
    """
    初始化随机种子以确保结果的可重复性。
    
    参数:
        seed (int): 用于初始化的随机种子。
        cuda_deterministic (bool): 是否启用CUDA的确定性算法。默认为True，确保更高的可复现性，但可能会降低性能。
    """
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    if cuda_deterministic:  # 如果启用CUDA的确定性算法，则会更慢，但结果更可复现
       # 启用CUDA的确定性算法
       cudnn.deterministic = True
       # 禁用CUDA的自动调优以避免非确定性算法
       cudnn.benchmark = False
    else:  # 如果不启用CUDA的确定性算法，则会更快，但结果可能不太可复现
       # 禁用CUDA的确定性算法
       cudnn.deterministic = False
       # 启用CUDA的自动调优以提高性能
       cudnn.benchmark = True


def loss_func(reward):
    """
    计算损失和准确率。

    参数:
        reward (Tensor): 模型输出的奖励值，形状为 (batch_size, 2)。

    返回:
        loss (Tensor): 平均交叉熵损失。
        loss_list (Tensor): 每个样本的交叉熵损失。
        acc (Tensor): 准确率，表示 reward[:, 0] 是否大于 reward[:, 1] 的比例。
    """
    # 创建目标张量，所有元素为0，类型为长整型，并移动到与reward相同的设备， 将 target 全部设置为 0，假设所有样本对中左边的样本都更好
    target = torch.zeros(reward.shape[0], dtype=torch.long).to(reward.device)
    
    # 计算交叉熵损失，不进行降维（返回每个样本的损失）
    loss_list = F.cross_entropy(reward, target, reduction='none')
    
    # 计算平均损失
    loss = torch.mean(loss_list)
    
    # 计算奖励差值，reward的第0列减去第1列
    reward_diff = reward[:, 0] - reward[:, 1]
    
    # 计算准确率，reward_diff大于0的比例
    acc = torch.mean((reward_diff > 0).clone().detach().float())
    
    return loss, loss_list, acc


if __name__ == "__main__":
    
    if opts.std_log:
        std_log()

    """
    主程序入口，用于初始化环境、设置设备、加载数据集、配置分布式训练（如果启用），并创建数据加载器。
    """
    
    # if opts.distributed:
    #     # 初始化分布式进程组，使用NCCL后端进行通信
    #     torch.distributed.init_process_group(backend="nccl")
    #     # 获取当前进程的全局排名
    #     local_rank = torch.distributed.get_rank()
    #     # 设置当前进程使用的GPU设备
    #     torch.cuda.set_device(local_rank)
    #     # 创建对应GPU的设备对象
    #     device = torch.device("cuda", local_rank)
    #     # 初始化随机种子，确保不同进程有不同的种子以增加多样性
    #     init_seeds(opts.seed + local_rank)
        
    # else:
    #     # 如果不启用分布式训练，选择使用GPU（如果可用）或CPU作为设备
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     # 初始化随机种子，确保结果的可重复性
    #     init_seeds(opts.seed)

    if opts.distributed:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        global_rank = torch.distributed.get_rank()
        init_seeds(opts.seed + global_rank)
        print(f"Initialized process with global_rank: {global_rank}, local_rank: {local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        global_rank = 0
        init_seeds(opts.seed)
        print("Initialized single process training")

    # 初始化可视化工具（例如TensorBoard），用于记录训练过程中的指标
    print("Initializing visualizer...")
    writer = visualizer()

    if opts.rank_pair:
        # 如果启用rank_pair模式，加载成对排序的数据集
        train_dataset = rank_pair_dataset("train")
        valid_dataset = rank_pair_dataset("valid")
        test_dataset = rank_pair_dataset("test")
    else:
        # 否则，加载常规的排序数据集
        train_dataset = rank_dataset("train")
        valid_dataset = rank_dataset("valid")
        test_dataset = rank_dataset("test")
    
    if opts.distributed:
        # 如果启用分布式训练，使用分布式采样器以确保每个进程加载不同的数据子集
        train_sampler = DistributedSampler(train_dataset)
        # 创建分布式训练的数据加载器，使用分布式采样器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=opts.batch_size, 
            sampler=train_sampler, 
            collate_fn=collate_fn if not opts.rank_pair else None
        )
    else:
        # 否则，创建常规的数据加载器，启用数据洗牌
        train_loader = DataLoader(
            train_dataset, 
            batch_size=opts.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn if not opts.rank_pair else None
        )
    
    # 创建验证集的数据加载器，启用数据洗牌
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=opts.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn if not opts.rank_pair else None
    )
    
    # 创建测试集的数据加载器，启用数据洗牌
    test_loader = DataLoader(
        test_dataset, 
        batch_size=opts.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn if not opts.rank_pair else None
    )

    # 设置训练迭代次数
    opts.train_iters = opts.epochs * len(train_loader)
    # 计算每多少步进行一次验证
    steps_per_valid = len(train_loader) // opts.valid_per_epoch

    # 打印训练集的长度
    print("len(train_dataset) = ", len(train_dataset))
    # 打印每个epoch的迭代次数
    print("train_dataset.iters_per_epoch = ", train_dataset.iters_per_epoch)
    # 打印训练数据加载器的长度
    print("len(train_loader) = ", len(train_loader))
    # 打印每多少步进行一次验证
    print("steps_per_valid = ", steps_per_valid)

    # 初始化奖励模型，并将其移动到指定设备（GPU或CPU）
    model = ImageReward(device).to(device)

    # 如果指定了预加载模型的路径，则加载预训练模型
    if opts.preload_path:
        model = preload_model(model)
        print("model_opts.preload_path:", model)

    # 初始化Adam优化器，设置学习率、动量参数和epsilon值
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opts.lr,
        betas=(opts.adam_beta1, opts.adam_beta2),
        eps=opts.adam_eps
    )

    # 获取学习率调度器
    scheduler = get_learning_rate_scheduler(optimizer, opts)

    # 如果启用了分布式训练，将模型包装为分布式数据并行模型
    if opts.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
        print("model_opts.distributed:", model)

    # 如果当前进程是主进程（rank 0），进行验证结果的打印和记录
    if get_rank() == 0:
        # 设置模型为评估模式
        model.eval()
        valid_loss = []       # 用于存储验证集的损失
        valid_acc_list = []   # 用于存储验证集的准确率

        # 禁用梯度计算，提高验证过程的效率
        with torch.no_grad():
            # 遍历验证集数据加载器
            for step, batch_data_package in enumerate(valid_loader):
                # 获取模型对验证数据的奖励输出
                reward = model(batch_data_package)
                # 计算损失和准确率
                loss, loss_list, acc = loss_func(reward)
                # 将每个样本的损失添加到valid_loss列表中
                valid_loss.append(loss_list)
                # 将准确率添加到valid_acc_list列表中
                valid_acc_list.append(acc.item())

        # 将所有样本的损失拼接成一个张量
        valid_loss = torch.cat(valid_loss, 0)
        # 打印验证集的平均损失和准确率
        print('Validation - Iteration %d | Loss %6.5f | Acc %6.4f' % (
            0,
            torch.mean(valid_loss),
            sum(valid_acc_list) / len(valid_acc_list)
        ))
        # 将验证损失记录到可视化工具中
        writer.add_scalar('Validation-Loss', torch.mean(valid_loss), global_step=0)
        # 将验证准确率记录到可视化工具中
        writer.add_scalar('Validation-Acc', sum(valid_acc_list) / len(valid_acc_list), global_step=0)  

    # 初始化最佳验证损失为一个很大的数，确保第一次验证时模型会被保存
    best_loss = 1e9
    # 清空优化器的梯度缓存
    optimizer.zero_grad()

    # 以下三行代码被注释掉，可能用于调整学习率或其他训练参数
    # fix_rate_list = [float(i) / 10 for i in reversed(range(10))]
    # fix_epoch_edge = [opts.epochs / (len(fix_rate_list)+1) * i for i in range(1, len(fix_rate_list)+1)]
    # fix_rate_idx = 0

    # 初始化列表，用于存储每个批次的损失和准确率
    losses = []
    acc_list = []

    # 开始遍历所有的训练轮次（epochs）
    for epoch in range(opts.epochs):
        
        # 遍历训练数据加载器中的每一个批次
        for step, batch_data_package in enumerate(train_loader):
            # 将模型设置为训练模式
            model.train()
            # 将批次数据输入模型，获取奖励输出
            reward = model(batch_data_package)
            # 计算损失和准确率

            loss, loss_list, acc = loss_func(reward)
            # 对损失进行梯度累积的正则化处理
            loss = loss / opts.accumulation_steps
            # 反向传播，计算梯度
            loss.backward()

            # 打印 MLP 参数的梯度
            print("\n\nMLP层参数的梯度情况:\n\n")
            for name, param in model.module.mlp.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad_mean = {param.grad.abs().mean().item()}")
                else:
                    print(f"{name}: no gradient")

            # 打印未冻结的 BLIP 模型参数的梯度
            print("\n\n未冻结的 BLIP 模型参数的梯度:\n\n")
            for name, param in model.module.blip.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        print(f"{name}: grad_mean = {param.grad.abs().mean().item()}")
                    else:
                        print(f"{name}: no gradient")


            # 将当前批次的损失添加到损失列表中
            losses.append(loss_list)
            # 将当前批次的准确率添加到准确率列表中
            acc_list.append(acc.item())

            # 计算当前的迭代次数
            iterations = epoch * len(train_loader) + step + 1
            # 根据梯度累积步数计算训练迭代次数
            train_iteration = iterations / opts.accumulation_steps
            
            # 如果当前迭代次数达到梯度累积步数的倍数，则更新模型参数
            if (iterations % opts.accumulation_steps) == 0:
                # 更新优化器，应用梯度下降
                optimizer.step()
                # 清空优化器的梯度缓存
                optimizer.zero_grad()
                # 更新学习率调度器
                scheduler.step()
                
                # 如果当前进程是主进程（rank 0），则打印并记录训练结果
                if get_rank() == 0:
                    # 将所有累积的损失拼接成一个张量
                    losses_log = torch.cat(losses, 0)
                    # 打印当前训练迭代的损失和准确率
                    # print('Iteration %d | Loss %6.5f | Acc %6.4f' % (
                    #     train_iteration, 
                    #     torch.mean(losses_log), 
                    #     sum(acc_list) / len(acc_list)
                    # ))
                    # 将训练损失记录到可视化工具（如TensorBoard）
                    writer.add_scalar('Train-Loss', torch.mean(losses_log), global_step=train_iteration)
                    # 将训练准确率记录到可视化工具
                    writer.add_scalar('Train-Acc', sum(acc_list) / len(acc_list), global_step=train_iteration)
                    
                # 清空损失和准确率列表，为下一个累积周期做准备
                losses.clear()
                acc_list.clear()
            
            # 如果当前迭代次数达到验证步数的倍数，则进行验证
            if (iterations % steps_per_valid) == 0:
                # 仅在主进程（rank 0）中执行验证
                if get_rank() == 0:
                    # 将模型设置为评估模式
                    model.eval()
                    # 初始化验证集的损失和准确率列表
                    valid_loss = []
                    valid_acc_list = []
                    # 禁用梯度计算，提高验证效率
                    with torch.no_grad():
                        # 遍历验证数据加载器中的每一个批次
                        for step, batch_data_package in enumerate(valid_loader):
                            # 将批次数据输入模型，获取奖励输出
                            reward = model(batch_data_package)
                            # 计算损失和准确率
                            loss, loss_list, acc = loss_func(reward)
                            # 将当前批次的损失添加到验证损失列表中
                            valid_loss.append(loss_list)
                            # 将当前批次的准确率添加到验证准确率列表中
                            valid_acc_list.append(acc.item())
                
                    # 将所有验证批次的损失拼接成一个张量
                    valid_loss = torch.cat(valid_loss, 0)
                    # 打印当前验证迭代的损失和准确率
                    print('Validation - Iteration %d | Loss %6.5f | Acc %6.4f' % (
                        train_iteration, 
                        torch.mean(valid_loss), 
                        sum(valid_acc_list) / len(valid_acc_list)
                    ))
                    # 将验证损失记录到可视化工具
                    writer.add_scalar('Validation-Loss', torch.mean(valid_loss), global_step=train_iteration)
                    # 将验证准确率记录到可视化工具
                    writer.add_scalar('Validation-Acc', sum(valid_acc_list) / len(valid_acc_list), global_step=train_iteration)
                        
                    # 如果当前验证损失优于之前的最佳损失，则保存模型
                    if torch.mean(valid_loss) < best_loss:
                        print("Best Val loss so far. Saving model")
                        # 更新最佳验证损失
                        best_loss = torch.mean(valid_loss)
                        print("best_loss = ", best_loss)
                        # 保存当前模型的状态
                        save_model(model)

    # 测试模型
    if get_rank() == 0:
        # 打印训练完成的提示信息
        print("training done")
        print("test: ")
        
        # 加载最佳模型权重
        model = load_model(model)
        # 将模型设置为评估模式，禁用Dropout和BatchNorm等训练特性
        model.eval()

        # 初始化列表，用于存储测试集的损失和准确率
        test_loss = []
        acc_list = []
        
        # 禁用梯度计算，提高测试过程的效率
        with torch.no_grad():
            # 遍历测试数据加载器中的每一个批次
            for step, batch_data_package in enumerate(test_loader):
                # 将批次数据输入模型，获取奖励输出
                reward = model(batch_data_package)
                # 计算损失和准确率
                loss, loss_list, acc = loss_func(reward)
                # 将当前批次的损失添加到测试损失列表中
                test_loss.append(loss_list)
                # 将当前批次的准确率添加到准确率列表中
                acc_list.append(acc.item())

        # 将所有测试批次的损失拼接成一个张量，便于计算平均损失
        test_loss = torch.cat(test_loss, 0)
        # 打印测试集的平均损失和准确率
        print('Test Loss %6.5f | Acc %6.4f' % (torch.mean(test_loss), sum(acc_list) / len(acc_list)))