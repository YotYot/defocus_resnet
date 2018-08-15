import os
from PIL import Image
import numpy as np

def create_bin_files(input_dir, output_dir,prefix):
    database_dir = output_dir
    train_dir = input_dir

    for lbl_dir in os.listdir(train_dir):
        lbl_dir_int = int(lbl_dir)
        lbl_dir_fullpath = os.path.join(train_dir,lbl_dir)
        lbl_dir_filenames = os.listdir(lbl_dir_fullpath)
        lbl_dir_cnt = len(lbl_dir_filenames)
        out = np.zeros((lbl_dir_cnt,3073), dtype=np.uint8)
        for i, img in enumerate(lbl_dir_filenames):
            img_fullpath = os.path.join(lbl_dir_fullpath, img)
            image = Image.open(img_fullpath)
            image = (np.array(image))
            r = image[:, :, 0].flatten()
            g = image[:, :, 1].flatten()
            b = image[:, :, 2].flatten()
            #Saving labels - 1 to have classes from 0 to 14
            label = [lbl_dir_int-1]
            out[i] = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
        output_file = prefix + "_" + lbl_dir+".bin"
        output_file_path = os.path.join(database_dir,output_file)
        out.tofile(output_file_path)


def main():
    create_bin_files(input_dir="/home/yotamg/data/rgb/train", output_dir="/home/yotamg/data/rgb/", prefix="train")
    create_bin_files(input_dir="/home/yotamg/data/rgb/val"  , output_dir="/home/yotamg/data/rgb/", prefix="val")


if __name__ == '__main__':
    main()