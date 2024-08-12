import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=256):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = int(img_options)
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options

    def __len__(self):
        return self.sizex // 2 # (input, target) 50개, (input2, target2) 50개 --> 50개(length)

    def __getitem__(self, index):
        index_ = (2 * index) % self.sizex
        index_2 = (2 * index + 1) % self.sizex

        ps = self.ps

        inp_path = self.inp_filenames[index_]
        inp_path_2 = self.inp_filenames[index_2]

        tar_path = self.tar_filenames[index_]
        tar_path_2 = self.tar_filenames[index_2]

        inp_img = Image.open(inp_path)
        inp_img_2 = Image.open(inp_path_2)

        tar_img = Image.open(tar_path)
        tar_img_2 = Image.open(tar_path_2)

        inp_img = inp_img.resize((512, 384))
        inp_img_2 = inp_img_2.resize((512, 384))
        tar_img = tar_img.resize((512, 384))
        tar_img_2 = tar_img_2.resize((512, 384))

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            inp_img_2 = TF.pad(inp_img_2, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img_2 = TF.pad(tar_img_2, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        inp_img_2 = TF.to_tensor(inp_img_2)
        tar_img = TF.to_tensor(tar_img)
        tar_img_2 = TF.to_tensor(tar_img_2)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        inp_img_2 = inp_img_2[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]
        tar_img_2 = tar_img_2[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            inp_img_2 = inp_img_2.flip(1)
            tar_img = tar_img.flip(1)
            tar_img_2 = tar_img_2.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            inp_img_2 = inp_img_2.flip(2)
            tar_img = tar_img.flip(2)
            tar_img_2 = tar_img_2.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            inp_img_2 = torch.rot90(inp_img_2, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
            tar_img_2 = torch.rot90(tar_img_2, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            inp_img_2 = torch.rot90(inp_img_2, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
            tar_img_2 = torch.rot90(tar_img_2, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            inp_img_2 = torch.rot90(inp_img_2, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
            tar_img_2 = torch.rot90(tar_img_2, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            inp_img_2 = torch.rot90(inp_img_2.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
            tar_img_2 = torch.rot90(tar_img_2.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            inp_img_2 = torch.rot90(inp_img_2.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))
            tar_img_2 = torch.rot90(tar_img_2.flip(2), dims=(1, 2))

        filename_i = os.path.splitext(os.path.split(inp_path)[-1])[0]
        filename_i_2 = os.path.splitext(os.path.split(inp_path_2)[-1])[0]
        filename_t = os.path.splitext(os.path.split(tar_path)[-1])[0]
        filename_t_2 = os.path.splitext(os.path.split(tar_path_2)[-1])[0]

        sample = {
            'target': tar_img,
            'target2': tar_img_2,
            'input': inp_img,
            'input2': inp_img_2,
            'filename_i': filename_i,
            'filename_i_2': filename_i_2,
            'filename_t': filename_t,
            'filename_t_2': filename_t_2
          }


        return sample


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options

    def __len__(self):
        return self.sizex // 2

    def __getitem__(self, index):
        index_ = (2 * index) % self.sizex
        index_2 = (2 * index + 1) % self.sizex

        ps = self.ps

        inp_path = self.inp_filenames[index_]
        inp_path_2 = self.inp_filenames[index_2]

        tar_path = self.tar_filenames[index_]
        tar_path_2 = self.tar_filenames[index_2]

        inp_img = Image.open(inp_path)
        inp_img_2 = Image.open(inp_path_2)

        tar_img = Image.open(tar_path)
        tar_img_2 = Image.open(tar_path_2)

        inp_img = inp_img.resize((512, 384))
        inp_img_2 = inp_img_2.resize((512, 384))
        tar_img = tar_img.resize((512, 384))
        tar_img_2 = tar_img_2.resize((512, 384))

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps, ps))
            inp_img_2 = TF.center_crop(inp_img_2, (ps, ps))
            tar_img = TF.center_crop(tar_img, (ps, ps))
            tar_img_2 = TF.center_crop(tar_img_2, (ps, ps))

        inp_img = TF.to_tensor(inp_img)
        inp_img_2 = TF.to_tensor(inp_img_2)
        tar_img = TF.to_tensor(tar_img)
        tar_img_2 = TF.to_tensor(tar_img_2)

        filename_i = os.path.splitext(os.path.split(inp_path)[-1])[0]
        filename_i_2 = os.path.splitext(os.path.split(inp_path_2)[-1])[0]
        filename_t = os.path.splitext(os.path.split(tar_path)[-1])[0]
        filename_t_2 = os.path.splitext(os.path.split(tar_path_2)[-1])[0]

        sample = {
            'target': tar_img,
            'target2': tar_img_2,
            'input': inp_img,
            'input2': inp_img_2,
            'filename_i': filename_i,
            'filename_i_2': filename_i_2,
            'filename_t': filename_t,
            'filename_t_2': filename_t_2
          }

        return sample


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size // 2

    def __getitem__(self, index):
        index_ = (2 * index) % self.inp_size
        index_2 = (2 * index + 1) % self.inp_size

        path_inp = self.inp_filenames[index_]
        path_inp_2 = self.inp_filenames[index_2]
        filename_i = os.path.splitext(os.path.split(path_inp)[-1])[0]
        filename_i_2 = os.path.splitext(os.path.split(path_inp_2)[-1])[0]
        inp = Image.open(path_inp)
        inp_2 = Image.open(path_inp_2)

        inp = inp.resize((512, 384))
        inp_2 = inp_2.resize((512, 384))

        inp = TF.to_tensor(inp)
        inp_2 = TF.to_tensor(inp_2)

        sample = {
            'input': inp,
            'input2': inp_2,
            'filename_i': filename_i,
            'filename_i_2': filename_i_2
        }
        return sample

if __name__ == "__main__":

    db_path = 'C:\\Users\\yunjeongyong\\Desktop\\intern\\dataset\\Synthetic_Rain_Datasets\\train'
    dataset = DataLoaderTrain(rgb_dir=db_path,img_options=256)

    for i in range(20):
        sample = dataset[i]
        target, target2, input, input2, filename_i, filename_i_2,filename_t, filename_t_2 = sample['target'], sample['target2'], sample['input'], sample['input2'], sample['filename_i'], sample['filename_i_2'],sample['filename_t'], sample['filename_t_2']

        # 'target': tar_img,
        # 'target2': tar_img_2,
        # 'input': inp_img,
        # 'input2': inp_img_2,
        # 'filename_i': filename_i,
        # 'filename_i_2': filename_i_2,
        # 'filename_t': filename_t,
        # 'filename_t_2': filename_t_2
        print(target)
        print(target.shape)
        print(target2)
        print(target2.shape)
        print(input)
        print(input.shape)
        print(input2)
        print(input2.shape)
        # print(filename)
        # print(filename2)


    print(0)
