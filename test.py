import os
import re
# Define the directory path
dir_path = "/home/LeiFeng/xiaolong/moe_quantize/autogptq_deepseek-ai"

# List to hold the names of subdirectories
subdirs = []

# Walk through the directory
for entry in os.scandir(dir_path):
    if entry.is_dir():
        match = re.search(r'top(\d+)', entry.name)
        if match:
            # Extract the number from the match
            number = int(match.group(1))
            if number >3:
                subdirs.append(entry.name)
    
# Print the subdirectories
print(subdirs)

# bits = []
# for dir in subdirs:
#     bit = dir.split('w_bit_')[1]
#     bits.append(bit)
    
# print(bits)