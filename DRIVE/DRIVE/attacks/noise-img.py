import h5py
import numpy as np
import os
import cv2
from tqdm import tqdm

def add_gaussian_noise(image, mean=0, sigma=0.10):
    image = image / 255.0
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0, 1)
    output = (noisy_image * 255).astype(np.uint8)
    return output

def save_image(image_data, filename):
    bgr_image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, bgr_image_data)

def modify_image_group(h5_file_path):

    with h5py.File(h5_file_path, 'a') as h5file:
        group_keys = list(h5file.keys())
        print("Group keys:", group_keys)
        group_progress = tqdm(group_keys, desc="Processing groups")
        
        for group_key in group_progress:
            group = h5file[group_key]
            if 'image' in group:
                image_data = group['image'][:]
            
                if image_data.dtype != np.uint8:
                    image_data = image_data.astype(np.float32) / 255.0  
                    image_data = (image_data * 255).astype(np.uint8)

                original_image_path = '/hpc2hdd/home/tianlangxue/XAI4AD/concept_gridlock/my_perturbation/original.jpg'
                save_image(image_data[0], original_image_path)
                print(f"Saved original image to {original_image_path}")
                # import pdb
                # pdb.set_trace()

                noisy_image_data = np.zeros_like(image_data)
                image_progress = tqdm(range(image_data.shape[0]), desc=f"Adding noise to images in group '{group_key}'")
                for i in image_progress:
                    noisy_image_data[i] = add_gaussian_noise(image_data[i])

                noise_image_path = '/hpc2hdd/home/tianlangxue/XAI4AD/concept_gridlock/my_perturbation/noise.jpg'
                save_image(noisy_image_data[0], noise_image_path)
                print(f"Saved noise image to {noise_image_path}")

                group['image'][...] = noisy_image_data

                print(f"Added Gaussian noise to all images in group '{group_key}'")

# 配置参数
h5_file_path = '/hpc2hdd/home/tianlangxue/XAI4AD/comma2k19data/comma_test_w_desired_filtered copy.h5py'  # HDF5文件路径

# 运行函数
modify_image_group(h5_file_path)