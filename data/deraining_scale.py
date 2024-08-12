import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import cv2
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DERAINDataset(Dataset):
    def __init__(self, db_path, img_size, scale_1, scale_2, transform):
        super(DERAINDataset, self).__init__()

        rain_files = sorted(os.listdir(os.path.join(db_path, 'input')))
        tar_files = sorted(os.listdir(os.path.join(db_path, 'target')))

        self.rain_filenames = [os.path.join(db_path, 'input', x) for x in rain_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(db_path, 'target', x) for x in tar_files if is_image_file(x)]


        self.scale_1 = scale_1
        self.scale_2 = scale_2
        self.db_path = db_path
        self.img_size = img_size
        self.transform = transform
        self.sizex = len(self.tar_filenames)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        rain_path = self.rain_filenames[index_]
        rain_img = cv2.imread(rain_path, cv2.IMREAD_COLOR)
        rain_img = cv2.cvtColor(rain_img, cv2.COLOR_BGR2RGB)
        rain_img = cv2.resize(rain_img, self.img_size)
        rain_img = np.array(rain_img).astype('float32') / 255

        h, w, c = rain_img.shape
        rain_img_scale_1 = cv2.resize(rain_img, dsize=(self.scale_1, int(h*(self.scale_1/w))), interpolation=cv2.INTER_CUBIC)
        rain_img_scale_2 = cv2.resize(rain_img, dsize=(self.scale_2, int(h*(self.scale_2/w))), interpolation=cv2.INTER_CUBIC)
        rain_img_scale_2 = rain_img_scale_2[:160, :, :]
        # rain_img = np.transpose(rain_img, (2, 0, 1))
        # rain_img = Image.open(rain_path)
        # tar_img = Image.open(tar_path)
        #

        tar_path = self.tar_filenames[index_]
        tar_img = cv2.imread(tar_path, cv2.IMREAD_COLOR)
        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
        tar_img = cv2.resize(tar_img, self.img_size)
        tar_img = np.array(tar_img).astype('float32') / 255
        # tar_img = np.transpose(tar_img, (2, 0, 1))\

        h, w, c = tar_img.shape
        tar_img_scale_1 = cv2.resize(tar_img, dsize=(self.scale_1, int(h*(self.scale_1/w))), interpolation=cv2.INTER_CUBIC)
        tar_img_scale_2 = cv2.resize(tar_img, dsize=(self.scale_2, int(h*(self.scale_2/w))), interpolation=cv2.INTER_CUBIC)
        tar_img_scale_2 = tar_img_scale_2[:160, :, :]

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        sample = {'rain_img':rain_img, 'rain_img_scale_1':rain_img_scale_1, 'rain_img_scale_2':rain_img_scale_2,
                  'tar_img':tar_img, 'tar_img_scale_1':tar_img_scale_1, 'tar_img_scale_2':tar_img_scale_2, 'filename':filename}

        if self.transform:
            sample = self.transform(sample)

        return sample

        # c, h, w = tar_img.size
        # padw = self.img_size-w if w < self.img_size else 0
        # padh = self.img_size-h if h < self.img_size else 0
        #
        #
        # if padw!= 0 or padh!= 0:
        #     rain_img = F.pad(rain_img, (0, 0, padw, padh), padding_mode='reflect')





if __name__ == "__main__":

    db_path = 'C:\\Users\\yunjeongyong\\Desktop\\intern\\MPRNet-main\\Deraining\\Datasets\\train'
    dataset = DERAINDataset(db_path, (512, 384), scale_1=384, scale_2=224, transform=None)

    for i in range(20):
        object_data = dataset[i]
        rain_img, tar_img, filename = object_data['rain_img'], object_data['tar_img'], object_data['filename']
        print(rain_img)
        print(rain_img.shape)
        print(tar_img)
        print(tar_img.shape)
        print(filename)

    print(0)
