import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.LossDisplayer import LossDisplayer
from model.derain_model import DeraiNet

from option.config import Config
from torch import nn, optim
from data.dataset_deraining import get_training_data, get_validation_data
from trainer_feat_loss import train_epoch
from trainer_feat_loss import eval_epoch
from util.image_pool import ImagePool
from torch.utils.tensorboard import SummaryWriter
from utils.util import RandCrop, RandHorizontalFlip, RandRotation, Normalize, ToTensor, RandShuffle

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

# config file
config = Config({
    # device
    "GPU_ID": [2, 3],
    "num_workers": 0,

    "input_nc": 3,
    "input_nc_2": 6,
    "output_nc": 6,
    "output_nc_2": 3,
    "ngf": 64,
    "netG": "resnet_9blocks",
    "netD": "basic",
    "n_layers_D": 3,
    "norm": "instance",
    "init_type": "normal",
    "ndf": 64,
    "init_gain": 0.02,
    "checkpoint_path": None,
    "db_path": "/media/mnt/dataset/train",
    "db_path_val": "/media/mnt/dataset/test/Rain100H",
    "outf": "./outputs/output_02_17",
    "batch_size": 8,
    "pool_size": 50,
    "gan_mode": "vanilla",
    "lr": 0.00001,
    "beta1": 0.5,
    "epoch": 800,
    "lambda_recon": 40.0,
    "lambda_Idt": 40.0,
    "lambda_gan":4.0,
    "snap_path": "./weights_02_17",
    "val_freq": 10,
    "save_freq": 5,

})

# device setting
# config.device = torch.device("cuda:%s" % config.GPU_ID if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     print('Using GPU %s' % config.GPU_ID)
# else:
#     print('Using CPU')

config.device = config.GPU_ID

# data load
train_dataset = get_training_data(rgb_dir=config.db_path, img_options=256)
test_dataset = get_validation_data(rgb_dir=config.db_path_val, img_options=None)
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,drop_last=True, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers,drop_last=True, shuffle=False)

# Loss
#adversarial_loss = torch.nn.BCELoss().cuda()
# criterion_GAN = torch.nn.MSELoss().cuda()
# criterion_GAN = networks.GANLoss(config.gan_mode).cuda()
criterion_l1 = torch.nn.L1Loss().cuda()
criterion_mse = torch.nn.MSELoss().cuda()

#criterion_recon = torch.nn.L1Loss().cuda()



# create model
net = DeraiNet().cuda()
net = torch.nn.DataParallel(net, device_ids=[0, 1])


# optimizer, scheduler
optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

lr_lambda = lambda epoch: 1 - ((epoch - 1) // 100) / (config.epoch / 100)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

# load weights & optimizer
if config.checkpoint_path is not None:
    checkpoint = torch.load(config.checkpoint_path, map_location='cuda:1')
    net.load_state_dict(checkpoint["net"])
    epoch = checkpoint["epoch"]
    start_epoch = 0
else:
    start_epoch = 0


# make directory for saving weights
if not os.path.exists(config.snap_path):
    os.mkdir(config.snap_path)


# train & validation
losses = []
for epoch in range(start_epoch, config.epoch):
    lossG = train_epoch(config, epoch, net, criterion_l1, criterion_mse, optimizer, scheduler, train_loader)

    if (epoch + 1) % config.val_freq == 0:
        lossG_t = eval_epoch(config, epoch, net, criterion_l1, criterion_mse, test_loader)






