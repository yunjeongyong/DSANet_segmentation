# import torch
# from torch import nn
# import torch.nn.functional as F
# from model.networks_feat_loss import Generator
# from model.networks_feat_loss_decoder import Generator_d
#
#
# class DeraiNet(nn.Module):
#     def __init__(self,):
#         super(DeraiNet, self).__init__()
#         self.netG = Generator()
#         self.netG_i = Generator_d()
#         self.netG_r = Generator_d()
#
#     def forward(self, input1, input2, target, target2):
#         f_1, f_2, half_size = self.netG(input1, align_corners=True)  # pred_img_1은 비 없는거
#         output_i = self.netG_i(input1, f_1, half_size, align_corners=True)
#         output_r = self.netG_r(input1, f_2, half_size, align_corners=True)
#
#         f_1_2, f_2_2, half_size_2 = self.netG(input2, align_corners=True)  # pred_img_1은 비 없는거
#         output_i_2 = self.netG_i(input2, f_1_2, half_size_2, align_corners=True)
#         output_r_2 = self.netG_r(input2, f_2_2, half_size_2, align_corners=True)
#
#         img1_rain2 = output_i + output_r_2
#         img2_rain1 = output_r + output_i_2
#
#         ff_1, ff_2, half_s_ize = self.netG(img1_rain2, align_corners=True)
#         output_ii = self.netG_i(img1_rain2, ff_1, half_s_ize, align_corners=True)
#         output_rr = self.netG_r(img1_rain2, ff_2, half_s_ize, align_corners=True)
#
#         ff_1_2, ff_2_2, half_s_ize_2 = self.netG(img2_rain1, align_corners=True)
#         output_ii_2 = self.netG_i(img2_rain1, ff_1, half_s_ize_2, align_corners=True)
#         output_rr_2 = self.netG_r(img2_rain1, ff_2, half_s_ize_2, align_corners=True)
#
#         recon_img1 = output_ii + output_rr_2
#         recon_img2 = output_rr + output_ii_2
#
#         # target to model
#         ft_1, ft_2, t_half_size = self.netG(target, align_corners=True)
#         output_i_t = self.netG_i(target, ft_1, half_s_ize, align_corners=True)
#
#         ft_1_2, ft_2_2, t_half_size_2 = self.netG(target2, align_corners=True)
#         output_i_t2 = self.netG_i(target2, ft_1_2, t_half_size_2, align_corners=True)
#
#
#
#         # outputs = {
#         #     "": ,
#         #
#         # }
#
#         return {
#             "f_1": f_1,
#             "output_i":output_i,
#             "ff_1":ff_1,
#             "output_ii": output_ii,
#             "recon_img1":recon_img1,
#             "f_1_2":f_1_2,
#             "output_i_2":output_i_2,
#             "ff_1_2":ff_1_2,
#             "output_ii_2":output_ii_2,
#             "recon_img2":recon_img2,
#             "output_i_t":output_i_t,
#             "ft_1_2":ft_1_2,
#             "output_i_t2":output_i_t2,
#             "ft_1":ft_1,
#         }


import torch
from torch import nn
import torch.nn.functional as F
from model.networks_feat_loss import Generator
from model.networks_feat_loss_decoder import Generator_d


class DeraiNet(nn.Module):
    def __init__(self,):
        super(DeraiNet, self).__init__()
        self.Encoder = Generator()
        self.Decoder = Generator_d()


    def forward(self, input1, input2, target, target2):
        f_1, f_2, half_size, input1_out = self.Encoder(input1, align_corners=True)  # pred_img_1은 비 없는거
        tf_1, tf_2, half_size_t, target_out = self.Encoder(target, align_corners=True)
        norain_input1_f = torch.cat([f_1, tf_2], dim=1)
        norain_input1_output = self.Decoder(input1, norain_input1_f, half_size, align_corners=True) #input, out

        input1_rain = input1_out - target_out

        f_1_2, f_2_2, half_size_2, input2_out = self.Encoder(input2, align_corners=True)  # pred_img_1은 비 없는거
        tf_1_2, tf_2_2, half_size_t_2, target2_out = self.Encoder(target2, align_corners=True)
        norain_input2_f = torch.cat([f_1_2, tf_2_2], dim=1)
        norain_input2_output = self.Decoder(input2, norain_input2_f, half_size_2, align_corners=True)  # input, out

        tworain_input1_f = torch.cat([f_1, f_2_2], dim=1)
        tworain_input1_output = self.Decoder(input1, tworain_input1_f, half_size, align_corners=True)

        onerain_input2_f = torch.cat([f_1_2, f_2], dim=1)
        onerain_input2_output = self.Decoder(input2, onerain_input2_f, half_size_2, align_corners=True)

        ################################step2
        ff_1, ff_2, half_size_f, norain_input1_out = self.Encoder(norain_input1_output, align_corners=True)  # pred_img_1은 비 없는거
        rff_1, rff_2, half_size_rf, tworain_input1_out = self.Encoder(tworain_input1_output, align_corners=True)
        norain_nr_input1_f = torch.cat([rff_1, ff_2], dim=1)
        norain_nr_input1_output = self.Decoder(norain_input1_output, norain_nr_input1_f, half_size_f, align_corners=True)  # input, out

        ff_1_2, ff_2_2, half_size_f2, norain_input2_out = self.Encoder(norain_input2_output, align_corners=True)  # pred_img_1은 비 없는거
        rff_1_2, rff_2_2, half_size_rf2, onerain_input2_out = self.Encoder(onerain_input2_output, align_corners=True)
        norain_nr_input2_f = torch.cat([rff_1_2, ff_2_2], dim=1)
        norain_nr_input2_output = self.Decoder(norain_input2_output, norain_nr_input2_f, half_size_f2, align_corners=True)  # input, out

        orirain_input1_f = torch.cat([rff_1, rff_2_2], dim=1)
        orirain_input1_output = self.Decoder(norain_input1_output, orirain_input1_f, half_size_f, align_corners=True)

        orirain_input2_f = torch.cat([rff_1_2, rff_2], dim=1)
        orirain_input2_output = self.Decoder(norain_input2_output, orirain_input2_f, half_size_f2, align_corners=True)









        # outputs = {
        #     "": ,
        #
        # }

        return {
            "f_1": f_1,
            "tf_1":tf_1,
            "f_1_2":f_1_2,
            "tf_1_2": tf_1_2,
            "ff_1":ff_1,
            "rff_1":rff_1,
            "ff_1_2":ff_1_2,
            "rff_1_2":rff_1_2,
            "target_out":target_out,
            "tworain_input1_f":tworain_input1_f,
            "target2_out":target2_out,
            "onerain_input2_f":onerain_input2_f,
            "norain_input1_output":norain_input1_output,
            "norain_input2_output":norain_input2_output,
            "norain_nr_input1_output":norain_nr_input1_output,
            "norain_nr_input2_output":norain_nr_input2_output,
            "orirain_input1_output":orirain_input1_output,
            "orirain_input2_output":orirain_input2_output,
            "tworain_input1_output":tworain_input1_output,
            "onerain_input2_output":onerain_input2_output,
            
        }

