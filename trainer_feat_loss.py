import os
import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image as save_image_t
from torch.utils.data import DataLoader
from utils.LossDisplayer import LossDisplayer
import model.networks as networks
import pathlib
from torch import nn, optim
import torchvision.utils as vutils
from util.image_pool import ImagePool
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.util import RandCrop, RandHorizontalFlip, RandRotation, Normalize, ToTensor, RandShuffle


def save_image(t, fp):
    path = pathlib.Path(str(fp))
    path.parent.mkdir(exist_ok=True, parents=True)
    # print('123123123', fp)
    save_image_t(t, path)


def record_tb(images, filename, epoch, summary):
    img_grid = torchvision.utils.make_grid(images)
    summary.add_image(filename, img_grid, epoch)


def train_epoch(config, epoch, net, criterion_l1, criterion_mse, optimizer, scheduler, train_loader):
    losses = []
    net.train()

    disp = LossDisplayer(["loss_total"])
    summary = SummaryWriter()

    for idx, sample in enumerate(tqdm(train_loader)):
        # print(idx)
        # print(f"{idx}/{len(train_loader)}", end="\r")
        target, target2 = sample['target'].cuda(), sample['target2'].cuda()
        input1, input2 = sample['input'].cuda(), sample['input2'].cuda()
        filename_i, filename_i_2 = sample['filename_i'], sample['filename_i_2']
        filename_t, filename_t_2 = sample['filename_t'], sample['filename_t_2']

        optimizer.zero_grad()

        outputs = net(input1, input2, target, target2)

        # l1 loss image
        loss_l1_input1 = criterion_l1(input1, outputs["orirain_input1_output"])  # 파랑
        loss_l1_input2 = criterion_l1(input2, outputs["orirain_input2_output"])

        loss_l1_input1_2 = criterion_l1(target, outputs["norain_input1_output"])
        loss_l1_input2_2 = criterion_l1(target2, outputs["norain_input2_output"])

        loss_l1_output1_2 = criterion_l1(target, outputs["norain_nr_input1_output"])
        loss_l1_output2_2 = criterion_l1(target2, outputs["norain_nr_input2_output"])

        loss_l1 = loss_l1_input1 + loss_l1_input2 + loss_l1_input1_2 + loss_l1_input2_2 + loss_l1_output1_2 + loss_l1_output2_2

        # mse loss feature
        loss_mse_f1 = criterion_mse(outputs["tf_1"], outputs["f_1"])
        loss_mse_f1_2 = criterion_mse(outputs["tf_1"], outputs["ff_1"])
        loss_mse_rf1_2 = criterion_mse(outputs["tf_1"], outputs["rff_1"])

        loss_mse_f2 = criterion_mse(outputs["tf_1_2"], outputs["f_1_2"])
        loss_mse_f2_2 = criterion_mse(outputs["tf_1_2"], outputs["ff_1_2"])
        loss_mse_rf2_2 = criterion_mse(outputs["tf_1_2"], outputs["rff_1_2"])

        loss_mse_twofeature_input1 = criterion_mse(outputs["target_out"], outputs["tworain_input1_f"])
        loss_mse_twofeature_input2 = criterion_mse(outputs["target2_out"], outputs["onerain_input2_f"])

        loss_mse = loss_mse_f1 + loss_mse_f1_2 + loss_mse_rf1_2 + loss_mse_f2 + loss_mse_f2_2 + loss_mse_rf2_2 + loss_mse_twofeature_input1 + loss_mse_twofeature_input2

        loss_total = loss_mse + loss_l1
        loss_val = loss_total.item()
        losses.append(loss_val)

        loss_total.backward()
        optimizer.step()

        # Record loss
        disp.record([loss_total])

        # save image
        if idx % config.save_freq == 0:
            save_image(input1, f"{config.outf}/input_1/total_input_1_{epoch + 1}_{idx}.png")
            save_image(target, f"{config.outf}/target_1/total_target_1_{epoch + 1}_{idx}.png")

            save_image(outputs["norain_input1_output"],
                       f"{config.outf}/step1_output/total_norain_input1_output_{epoch + 1}_{idx}.png")
            save_image(outputs["norain_nr_input1_output"],
                       f"{config.outf}/step2_output/total_norain_nr_input1_output_{epoch + 1}_{idx}.png")
            save_image(outputs["orirain_input1_output"],
                       f"{config.outf}/recon_img1/total_orirain_input1_output_{epoch + 1}_{idx}.png")

            save_image(outputs["tworain_input1_output"],
                       f"{config.outf}/tworain_input1_output/total_tworain_input1_output_{epoch + 1}_{idx}.png")
            save_image(outputs["onerain_input2_output"],
                       f"{config.outf}/onerain_input2_output/total_onerain_input2_output_{epoch + 1}_{idx}.png")

            save_image(input2, f"{config.outf}/input_2/total_input_2_{epoch + 1}_{idx}.png")
            save_image(target2, f"{config.outf}/target_2/total_target_2_{epoch + 1}_{idx}.png")

            save_image(outputs["norain_input2_output"],
                       f"{config.outf}/step1_output2/total_norain_input2_output_{epoch + 1}_{idx}.png")
            save_image(outputs["norain_nr_input2_output"],
                       f"{config.outf}/step2_output2/total_norain_nr_input2_output_{epoch + 1}_{idx}.png")
            save_image(outputs["orirain_input2_output"],
                       f"{config.outf}/recon_img2/total_orirain_input2_output_{epoch + 1}_{idx}.png")

            record_tb(input1, "input1", epoch, summary)
            record_tb(input2, "input2", epoch, summary)
            record_tb(target, "target", epoch, summary)
            record_tb(target2, "target2", epoch, summary)
            record_tb(outputs["norain_input1_output"], "norain_input1_output", epoch, summary)
            record_tb(outputs["norain_nr_input1_output"], "norain_nr_input1_output", epoch, summary)
            record_tb(outputs["orirain_input1_output"], "orirain_input1_output", epoch, summary)

            record_tb(outputs["tworain_input1_output"], "tworain_input1_output", epoch, summary)
            record_tb(outputs["onerain_input2_output"], "onerain_input2_output", epoch, summary)

            record_tb(outputs["norain_input2_output"], "norain_input2_output", epoch, summary)
            record_tb(outputs["norain_nr_input2_output"], "norain_nr_input2_output", epoch, summary)
            record_tb(outputs["orirain_input2_output"], "orirain_input2_output", epoch, summary)

            print(f"Finish save train epoch{epoch + 1}_{idx} image ^^")
            # vutils.save_image(input1, f"{config.outf}/input_1/real_samples_epoch_{epoch + 1}_{idx}.png", normalize=True)
            # vutils.save_image(input2, f"{config.outf}/input_2/real_samples_epoch_{epoch + 1}_{idx}.png", normalize=True)

            # vutils.save_image(pred_img_1,
            #                  f"{config.outf}/pred_car/fake_samples_epoch_{epoch + 1}_{idx}.png",
            #                  normalize=True)
            # vutils.save_image(pred_img_2,
            #                   f"{config.outf}/pred_human/fake_samples_epoch_{epoch + 1}_{idx}.png",
            #                   normalize=True)
            # vutils.save_image(pred_input1,
            #                  f"{config.outf}/gen_rain/fake_samples_epoch_{epoch + 1}_{idx}.png",
            #                  normalize=True)
            # vutils.save_image(pred_input2,
            #                   f"{config.outf}/gen_rain_h/fake_samples_epoch_{epoch + 1}_{idx}.png",
            #                   normalize=True)

    # Step scheduler
    scheduler.step()

    # Record and display loss
    avg_losses = disp.get_avg_losses()
    summary.add_scalar("loss_total", avg_losses[0], epoch)
    # summary.add_scalar("loss_l1", avg_losses[1], epoch)
    # summary.add_scalar("loss_total", avg_losses[2], epoch)

    disp.display()
    disp.reset()

    # save weights
    if (epoch + 1) % config.save_freq == 0:
        weights_file_name = "epoch%d.pth" % (epoch + 1)
        weights_file = os.path.join(config.snap_path, weights_file_name)
        torch.save({
            "net": net.state_dict(),
            "epoch": epoch,
        }, weights_file)
        print('save weights of epoch %d' % (epoch + 1))

    return avg_losses[0]


def eval_epoch(config, epoch, net, criterion_l1, criterion_mse, test_loader):
    with torch.no_grad():

        losses = []
        net.eval()

        disp = LossDisplayer(["loss_total"])
        summary = SummaryWriter()
        for idx, sample in enumerate(tqdm(test_loader)):
            target, target2 = sample['target'].cuda(), sample['target2'].cuda()
            input1, input2 = sample['input'].cuda(), sample['input2'].cuda()
            filename_i, filename_i_2 = sample['filename_i'], sample['filename_i_2']
            filename_t, filename_t_2 = sample['filename_t'], sample['filename_t_2']
    
            outputs = net(input1, input2, target, target2)
    
            # l1 loss image
            loss_l1_input1 = criterion_l1(input1, outputs["orirain_input1_output"])  # 파랑
            loss_l1_input2 = criterion_l1(input2, outputs["orirain_input2_output"])
    
            loss_l1_input1_2 = criterion_l1(target, outputs["norain_input1_output"])
            loss_l1_input2_2 = criterion_l1(target2, outputs["norain_input2_output"])
    
            loss_l1_output1_2 = criterion_l1(target, outputs["norain_nr_input1_output"])
            loss_l1_output2_2 = criterion_l1(target2, outputs["norain_nr_input2_output"])
    
            loss_l1 = loss_l1_input1 + loss_l1_input2 + loss_l1_input1_2 + loss_l1_input2_2 + loss_l1_output1_2 + loss_l1_output2_2
    
            # mse loss feature
            loss_mse_f1 = criterion_mse(outputs["tf_1"], outputs["f_1"])
            loss_mse_f1_2 = criterion_mse(outputs["tf_1"], outputs["ff_1"])
            loss_mse_rf1_2 = criterion_mse(outputs["tf_1"], outputs["rff_1"])
    
            loss_mse_f2 = criterion_mse(outputs["tf_1_2"], outputs["f_1_2"])
            loss_mse_f2_2 = criterion_mse(outputs["tf_1_2"], outputs["ff_1_2"])
            loss_mse_rf2_2 = criterion_mse(outputs["tf_1_2"], outputs["rff_1_2"])
    
            loss_mse_twofeature_input1 = criterion_mse(outputs["target_out"], outputs["tworain_input1_f"])
            loss_mse_twofeature_input2 = criterion_mse(outputs["target2_out"], outputs["onerain_input2_f"])
    
            loss_mse = loss_mse_f1 + loss_mse_f1_2 + loss_mse_rf1_2 + loss_mse_f2 + loss_mse_f2_2 + loss_mse_rf2_2 + loss_mse_twofeature_input1 + loss_mse_twofeature_input2
    
            loss_total = loss_mse + loss_l1
            loss_val = loss_total.item()
            losses.append(loss_val)
            # Record loss
            disp.record([loss_total])
    
            # save image
            if idx % config.save_freq == 0:
                # save_image(input1, f"{config.outf}/test_input_1/total/total_real_samples_epoch_{epoch + 1}_{idx}.png")
                # save_image(pred_img_1,
                #            f"{config.outf}/test_derain/total/total_fake_samples_epoch_{epoch + 1}_{idx}.png")
                # save_image(pred_rain_1,
                #            f"{config.outf}/test_gen_rain/total/total_fake_samples_epoch_{epoch + 1}_{idx}.png")
                # save_image(pred_input1,
                #            f"{config.outf}/test_fake_input/total/total_fake_samples_epoch_{epoch + 1}_{idx}.png")
                #
    
                # seasdasjdkasjdklasjkldasj
                save_image(input1, f"{config.outf}/test_input_1/total_input_1_{epoch + 1}_{idx}.png")
                save_image(target, f"{config.outf}/test_target_1/total_target_1_{epoch + 1}_{idx}.png")
                save_image(outputs["norain_input1_output"],
                           f"{config.outf}/test_step1_output/total_norain_input1_output_{epoch + 1}_{idx}.png")
                save_image(outputs["norain_nr_input1_output"],
                           f"{config.outf}/test_step2_output/total_norain_nr_input1_output_{epoch + 1}_{idx}.png")
                save_image(outputs["orirain_input1_output"],
                           f"{config.outf}/test_recon_img1/total_orirain_input1_output_{epoch + 1}_{idx}.png")
    
                save_image(outputs["tworain_input1_output"],
                           f"{config.outf}/test_tworain_input1_output/total_tworain_input1_output_{epoch + 1}_{idx}.png")
                save_image(outputs["onerain_input2_output"],
                           f"{config.outf}/test_onerain_input2_output/total_onerain_input2_output_{epoch + 1}_{idx}.png")
    
                save_image(input2, f"{config.outf}/test_input_2/total_input_2_{epoch + 1}_{idx}.png")
                save_image(target2, f"{config.outf}/test_target_2/total_target_2_{epoch + 1}_{idx}.png")
                save_image(outputs["norain_input2_output"],
                           f"{config.outf}/test_step1_output2/total_norain_input2_output_{epoch + 1}_{idx}.png")
                save_image(outputs["norain_nr_input2_output"],
                           f"{config.outf}/test_step2_output2/total_norain_nr_input2_output_{epoch + 1}_{idx}.png")
                save_image(outputs["orirain_input2_output"],
                           f"{config.outf}/test_recon_img2/total_orirain_input2_output_{epoch + 1}_{idx}.png")
    
                print(f"Finish save test epoch{epoch + 1}_{idx} image! ^^")

        # Record and display loss
        avg_losses = disp.get_avg_losses()
        summary.add_scalar("loss_total", avg_losses[0], epoch)
    
        disp.display()
        disp.reset()
    
        return avg_losses[0]








