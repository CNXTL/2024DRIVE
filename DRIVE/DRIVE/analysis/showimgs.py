import os
import h5py
import cv2
import numpy as np
from tqdm import tqdm

def save_image(image_data, filename, format='.jpg'):
    if image_data.dtype != np.uint8:
        image_data = image_data.astype(np.float32) / 255.0 
        image_data = (image_data * 255).astype(np.uint8)  
    

    bgr_image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, bgr_image_data)

def modify_image_group(h5_file_path, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with h5py.File(h5_file_path, 'a') as h5file:
      
        group_key = list(h5file.keys())[0] 
        group = h5file[group_key]
        if 'image' in group:

            image_data = group['image'][:]
   
            image_progress = tqdm(range(image_data.shape[0]), desc="Saving images")
            
            for i in image_progress:
 
                filename = os.path.join(save_dir, f'image_{i:04d}.jpg') 
                save_image(image_data[i], filename) 
                print(f"Saved image {i} to {filename}")


h5_file_path = '/your/dataset_path/comma_test_w_desired_filtered.h5py' # any h5 file you want to examine
modify_image_group(h5_file_path, save_dir)