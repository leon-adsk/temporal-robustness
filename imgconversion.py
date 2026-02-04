import numpy as np
import pandas as pd
import matplotlib.image as img
from pathlib import Path
from PIL import Image
from pathlib import Path

def get_width_from_array(array):    
        FILE_SIZE_TO_WIDTH_RULES = [
            {"max_kb": 10,   "width": 32},
            {"min_kb": 10,   "max_kb": 30,   "width": 64},
            {"min_kb": 30,   "max_kb": 60,   "width": 128},
            {"min_kb": 60,   "max_kb": 100,  "width": 256},
            {"min_kb": 100,  "max_kb": 200,  "width": 384},
            {"min_kb": 200,  "max_kb": 500,  "width": 512},
            {"min_kb": 500,  "max_kb": 1000, "width": 768},
            {"min_kb": 1000, "width": 1024}
        ]

        BYTES_PER_KB = 1024

        file_size = len(array)
        file_size_kb = file_size / BYTES_PER_KB
        # print(file_size_kb)

        for rule in FILE_SIZE_TO_WIDTH_RULES:
            # Check for the smallest size (< 10 kB)
            if "max_kb" in rule and "min_kb" not in rule and file_size_kb < rule["max_kb"]:
                return rule["width"]

            # Check for the largest size (> 1000 kB)
            if "min_kb" in rule and "max_kb" not in rule and file_size_kb >= rule["min_kb"]: # >= covers the 'greater than' range
                return rule["width"]

            # Check for intermediate ranges (X - Y kB)
            if "min_kb" in rule and "max_kb" in rule:
                if rule["min_kb"] <= file_size_kb < rule["max_kb"]:
                    return rule["width"]
                
def pe_to_img(bin_path: Path, img_path: Path) -> tuple[int, int]:
    """takes a binary and converts it to a sqaure-ish image, returns the dimensions"""
    binary = bin_path.read_bytes()

    data = np.frombuffer(binary, dtype=np.uint8)
    n = data.size
    
    if n == 0:
        return (0, 0)
    
    width = int(np.ceil(np.sqrt(n)))
    height = int(np.ceil(n / width))
    
    pad_len = (width * height) - n
    if pad_len > 0:
        data = np.pad(data, (0, pad_len), mode='constant')

    img_arr = data.reshape((width, height))

    out_file = img_path / f'{bin_path.name}.png'
    Image.fromarray(img_arr).save(out_file)
    
    return (width, height)

def convert(dataset_dir):
    print("beginning conversion of PE samples to PNG images - this process may take several minutes, do not interrupt")
    samples_dir = dataset_dir / 'samples'

    imgs_dir = dataset_dir / 'images'
    imgs_dir.mkdir(parents=True, exist_ok=True)

    past_dir = imgs_dir / 'past'
    future_dir = imgs_dir / 'future'
    benign_dir = imgs_dir / 'benign'

    benign_dir.mkdir(parents=True, exist_ok=True)
    future_dir.mkdir(parents=True, exist_ok=True)
    past_dir.mkdir(parents=True, exist_ok=True)   

    # Store ids for all files in sets to easily know which dir to save the images to
    df = pd.read_csv(dataset_dir / 'samples.csv')
    df['date'] = pd.to_datetime(df['submitted'])
    n = len(df)

    benign_ids = set(df[df['list'] == 'Whitelist']['id'])
    past_ids = set(df[(df['list'] == 'Blacklist') & (df['date'] < pd.Timestamp('2019-01-01').tz_localize('CET'))]['id'])
    future_ids = set(df[(df['list'] == 'Blacklist') & (df['date'] > pd.Timestamp('2019-01-01').tz_localize('CET'))]['id'])

    buckets = [(benign_ids, benign_dir), (past_ids, past_dir), (future_ids, future_dir)]

    # convert
    processed = 0
    nums = {}
    for sample in samples_dir.glob('*'):
        processed += 1
        if not sample.is_file:
            continue
        
        for bucket in buckets:
            sample_id = int(sample.name)
            if sample_id in bucket[0]:
                if bucket[1].name in nums:
                    nums[bucket[1].name] += 1
                else:
                    print(f"first {bucket[1].name} at sample {processed}")
                    nums[bucket[1].name] = 1
                pe_to_img(sample, bucket[1])
        
        if processed % 500 == 0:
            print(f'{processed}/{n}', end='\r')

    print(f'{processed}/{n}', end='\r')
    print(nums)
