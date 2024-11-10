import os
from PIL import Image
from collections import defaultdict

# Define the directories to process
directories = [
    "/bjzhyai03/workhome/luoqinyu/data_meme/download_raw/EmojiPackage",
    "/bjzhyai03/workhome/luoqinyu/data_meme/neg_meme_EmojiPackage"
]

# Dictionary to hold format counts
format_counts = defaultdict(int)
total_converted = 0

# Supported image formats by Pillow
supported_formats = Image.registered_extensions()

def convert_to_webp(input_path, output_path):
    """
    Converts an image to WebP format.
    """
    try:
        with Image.open(input_path) as img:
            img.save(output_path, 'WEBP')
        return True
    except Exception as e:
        print(f"Failed to convert {input_path}: {e}")
        return False

# Iterate through each directory
for directory in directories:
    if not os.path.isdir(directory):
        print(f"Directory does not exist: {directory}")
        continue

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            ext = ext.lower()

            # Check if the file has a supported image extension
            if ext in supported_formats:
                try:
                    with Image.open(file_path) as img:
                        original_format = img.format
                        format_counts[original_format] += 1

                        # Define the output file path with .webp extension
                        output_file = os.path.splitext(file_path)[0] + ".webp"

                        # Avoid reconverting if WebP already exists
                        if not os.path.exists(output_file):
                            success = convert_to_webp(file_path, output_file)
                            if success:
                                total_converted += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            else:
                print(f"Skipped non-image file: {file_path}")

# Output the statistics
print("\nConversion Statistics:")
print(f"Total different original formats: {len(format_counts)}")
for fmt, count in format_counts.items():
    print(f"Format '{fmt}': {count} files")
print(f"Total files converted to WebP: {total_converted}")