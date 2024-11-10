'''
@File       :   rank_pair_dataset.py
@Time       :   2023/03/02 15:38:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   Rank dataset as pair for reward model.
'''

import os
import json
import math
import random
import torch
from torch.utils.data import Dataset
from config.utils import *
from config.options import *
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from transformers import BertTokenizer
import clip

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

class rank_pair_dataset(Dataset):
    def __init__(self, dataset):
        self.preprocess = _transform(config['BLIP']['image_size'])
        self.tokenizer = init_tokenizer()
        
        if opts.load_pair_store:
            self.dataset_path = os.path.join(config['pair_store_base'], f"{dataset}.pth")
            self.data = torch.load(self.dataset_path)
        else:
            self.dataset_path = os.path.join(config['data_base'], f"{dataset}.json")
            with open(self.dataset_path, "r") as f:
                self.data = json.load(f)
                random.shuffle(self.data)
            self.data = self.make_data()
        
        self.iters_per_epoch = int(math.ceil(len(self.data)*1.0/opts.batch_size))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
    def store_dataset(self, dataset):
        makedir(config['pair_store_base'])
        torch.save(self.data, os.path.join(config['pair_store_base'], f"{dataset}.pth"))
    
    from PIL import UnidentifiedImageError

    def make_data(self):
        data_items = []
        bar = tqdm(range(len(self.data)), desc=f'making dataset: ')
        
        for item in self.data:
            img_set = []
            skip_item = False  # 标记是否跳过当前 item
            
            for generations in item["generations"]:
                img_path = os.path.join(config['image_base'], generations)
                try:
                    pil_image = Image.open(img_path)
                    image = self.preprocess(pil_image)
                    img_set.append(image)
                except UnidentifiedImageError:
                    print(f"无法识别图像文件: {img_path}，跳过整个 item。")
                    skip_item = True
                    break  # 跳出当前 item 的图像加载循环
                except Exception as e:
                    print(f"读取图像文件时发生错误: {img_path}，错误信息: {str(e)}，跳过整个 item。")
                    skip_item = True
                    break  # 跳出当前 item 的图像加载循环
            
            if skip_item:
                print(f"跳过 item: {item}")
                bar.update(1)
                continue  # 跳过当前 item，处理下一个 item
            
            # 确保 img_set 中的图像数量与原始 generations 一致
            if len(img_set) != len(item["generations"]):
                print(f"图像数量不一致，跳过 item: {item}")
                bar.update(1)
                continue
            
            # 继续处理文本和标签
            text_input = self.tokenizer(item["prompt"], padding='max_length', truncation=True, max_length=35, return_tensors="pt")
            labels = item["ranking"]
            
            for id_l in range(len(labels)):
                for id_r in range(id_l + 1, len(labels)):
                    dict_item = {}
                    dict_item['clip_text'] = clip.tokenize(item["prompt"], truncate=True)
                    dict_item['text_ids'] = text_input.input_ids
                    dict_item['text_mask'] = text_input.attention_mask
                    if labels[id_l] < labels[id_r]:
                        dict_item['img_better'] = img_set[id_l]
                        dict_item['img_worse'] = img_set[id_r]
                    elif labels[id_l] > labels[id_r]:
                        dict_item['img_better'] = img_set[id_r]
                        dict_item['img_worse'] = img_set[id_l]
                    else:
                        continue
                    data_items.append(dict_item)
            
            bar.update(1)
        
        return data_items
