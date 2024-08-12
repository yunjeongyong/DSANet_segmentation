# import torch
# import networks
# import torch.nn as nn
# from util.image_pool import ImagePool
# from base_model import BaseModel
#
#
# # self.pred_Bs = self.netG1(self.Os) # 자동차만 추출
# # self.pred_Rs = self.Os - self.pred_Bs # 비만 추출 = (비+자동차) - 자동차
# # self.pred_Rst = self.netG2(self.pred_Rs) # 비 생성
# # self.pred_Ot = self.pred_Bs + self.pred_Rst # (비+자동차) = 추출한 자동차 + 비 생성
#
# class RainModel(nn.Module):
#     def __init__(self, config, is_Train=True, epoch):
#         super(RainModel,self).__init__()
#
#         self.is_Train = is_Train
#         self.lambda_MSE = float(40.0)
#         self.lambda_GAN = float(4.0)
#         self.lambda_Idt = float(40.0)
#
#         # netG1 for synthetic rain removal
#         # netG2 for real rain generation
#         # netG3 for real rain removal
#         # netG4 for synthetic rain generation
#         self.netG1 = networks.define_G(config.input_nc, config.output_nc, config.ngf, 'unet_128', gpu_ids=config.device)
#         self.netG2 = networks.define_G(config.input_nc, config.output_nc, config.ngf, config.netG,
#                                        gpu_ids=config.device)
#         # self.netG3 = networks.define_G(config.input_nc, config.output_nc, config.ngf, 'unet_128', gpu_ids=self.gpu_ids)
#         # self.netG4 = networks.define_G(config.input_nc, config.output_nc, config.ngf, config.netG,
#         #                                gpu_ids=config.device)
#         self.netD_result_target = networks.define_D(config.output_nc, config.ndf, config.netD, config.n_layers_D, config.norm, config.init_type,
#                                          config.init_gain, gpu_ids=config.device)
#         self.netD_result_c_h = networks.define_D(config.output_nc, config.ndf, config.netD, config.n_layers_D, config.norm, config.init_type,
#                                          config.init_gain, gpu_ids=config.device)
#
#         # if self.is_Train:
#         #     if config.checkpoint_path is not None:
#         #         checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
#         #         self.netG1.load_state_dict(checkpoint["netG1_state_dict"])
#         #         self.netG2.load_state_dict(checkpoint["netG2_state_dict"])
#         #         self.netD_result_target.load_state_dict(checkpoint["netD_result_target_state_dict"])
#         #         self.netD_result_c_h.load_state_dict(checkpoint["netD_result_c_h_state_dict"])
#         #         epoch = checkpoint["epoch"]
#         #     else:
#         #         start_epoch = 0
#         #
#
#
#
#
#
#         self.criterion_MSE = torch.nn.MSELoss()
#         self.criterion_GAN = networks.GANLoss(config.gan_mode).to(config.device)
#         self.criterion_Idt = torch.nn.L1Loss()
#
#         self.Pool_fake_target = ImagePool(config.pool_size)
#         self.Pool_fake_target_2 = ImagePool(config.pool_size)
#         self.Pool_fake_car = ImagePool(config.pool_size)
#         self.Pool_fake_human = ImagePool(config.pool_size)
#
#         parameters_list = [dict(params=self.netG1.parameters(), lr=config.lr)]
#         parameters_list.append(dict(params=self.netG2.parameters(), lr=config.lr))
#
#         self.optimizer_G = torch.optim.Adam(parameters_list, lr=config.lr, betas=(config.beta1, 0.999))
#         self.optimizer_D_target = torch.optim.Adam(self.netD_result_target.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
#         self.optimizer_D_c_h = torch.optim.Adam(self.netD_result_c_h.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
#         self.optimizers = [self.optimizer_G, self.optimizer_D_target, self.optimizer_D_c_h]
#
#
#
#         lr_lambda = lambda epoch: 1 - ((epoch - 1) // 100) / (config.epoch / 100)
#         self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_G, lr_lambda=lr_lambda)
#         self.scheduler_D_target= torch.optim.lr_scheduler.LambdaLR(
#             optimizer=self.optimizer_D_target, lr_lambda=lr_lambda
#         )
#         self.scheduler_D_c_h = torch.optim.lr_scheduler.LambdaLR(
#             optimizer=self.optimizer_D_c_h, lr_lambda=lr_lambda
#         )
#
#     def forward(self, input_image, input_image_2):
#         self.pred_car = self.netG1(input_image)
#         pred_rain = input_image - self.pred_car
#         gen_rain = self.netG2(pred_rain)
#
#         self.pred_human = self.netG1(input_image_2)
#         pred_rain_h = input_image_2 - self.pred_human
#         gen_rain_h = self.netG2(pred_rain_h)
#
#         self.output_image = self.pred_car + gen_rain_h
#         self.output_image_h = self.pred_car + gen_rain
#
#         return self.pred_car, self.pred_human, self.output_image, self.output_image_h
#
#
#
#     def backward_G(self, input_image, input_image_2, target_image, target_image_2):
#
#         mse_loss_c = self.criterion_MSE(self.pred_car, target_image)
#         mse_loss_h = self.criterion_MSE(self.pred_human, target_image_2)
#         total_mse_loss = mse_loss_c + mse_loss_h
#
#         l1_loss_c = self.criterion_Idt(input_image, self.output_image)
#         l1_loss_h = self.criterion_Idt(input_image_2, self.output_image_h)
#         total_l1_loss = l1_loss_c + l1_loss_h
#
#         GAN_output_target = self.criterion_GAN(self.netD_result_target(self.output_image), True)
#         GAN_output_target_2 = self.criterion_GAN(self.netD_result_target(self.output_image_h), True)
#         GAN_pred_car = self.criterion_GAN(self.netD_result_c_h(self.pred_car), True)
#         GAN_pred_human = self.criterion_GAN(self.netD_result_c_h(self.pred_human), True)
#         total_GAN_loss = GAN_output_target + GAN_output_target_2 + GAN_pred_car + GAN_pred_human
#
#         total_loss = self.lambda_MSE * total_mse_loss + self.lambda_GAN * total_GAN_loss + self.lambda_Idt * total_l1_loss
#         return total_loss
#
#     def cal_GAN_loss_D_basic(self, netD, real, fake):
#         pred_real = netD(real)
#         loss_D_real = self.criterion_GAN(pred_real, True)
#         # Fake
#         pred_fake = netD(fake.detach())
#         loss_D_fake = self.criterion_GAN(pred_fake, False)
#         # Combined loss and calculate gradients
#         loss_D = (loss_D_real + loss_D_fake) * 0.5
#         return loss_D
#
#     def backward_D_target(self, input_image, input_image_2):
#         fake_target = self.Pool_fake_target.query(self.output_image)
#         fake_target_2 = self.Pool_fake_target_2.query(self.output_image_h)
#         self.loss_D_target = self.cal_GAN_loss_D_basic(self.netD_result_target, input_image, fake_target)
#         self.loss_D_target_2 = self.cal_GAN_loss_D_basic(self.netD_result_target, input_image_2, fake_target_2)
#         self.loss_D_total_t = self.loss_D_target + self.loss_D_target_2
#         return self.loss_D_total_t
#
#
#     def backward_D_c_h(self, target_image, target_image_2):
#         fake_car = self.Pool_fake_car.query(self.pred_car)
#         fake_human = self.Pool_fake_human.query(self.pred_human)
#         self.loss_D_car = self.cal_GAN_loss_D_basic(self.netD_result_c_h, target_image, fake_car)
#         self.loss_D_human = self.cal_GAN_loss_D_basic(self.netD_result_c_h, target_image_2, fake_human)
#         self.loss_D_total_c_h = self.loss_D_car + self.loss_D_human
#         return self.loss_D_total_c_h
#         # self.loss_D_total_c_h.backward()
#
#     def optimize_parameters(self, input_image, input_image_2, target_image, target_image_2):
#
#         self.pred_car, self.pred_human, self.output_image, self.output_image_h = self.forward(input_image, input_image_2)
#         self.set_requires_grad([self.netD_result_target, self.netD_result_c_h], False)
#         self.optimizer_G.zero_grad()
#         total_loss = self.backward_G(input_image, input_image_2, target_image, target_image_2)
#         total_loss.backward()
#         self.optimizer_G.step()
#
#         self.set_requires_grad([self.netD_result_target, self.netD_result_c_h], True)
#         self.optimizer_D_target.zero_grad()
#         loss_D_total_t = self.backward_D_target(input_image, input_image_2)
#         loss_D_total_t.backward()
#         self.optimizer_D_target.step()
#
#         self.optimizer_D_c_h.zero_grad()
#         loss_D_total_c_h = self.backward_D_c_h(target_image, target_image_2)
#         loss_D_total_c_h.backward()
#         self.optimizer_D_c_h.step()
#
#         return total_loss, loss_D_total_t, loss_D_total_c_h
#
#         #
#         # self.optimizers = [self.optimizer_G, self.optimizer_D_target, self.optimizer_D_c_h]
#         # mse_loss_c = self.criterion_MSE(self.pred_car, target_image)
#         # mse_loss_h = self.criterion_MSE(self.pred_human, target_image_2)
#         # total_mse_loss = mse_loss_c + mse_loss_h
#         #
#         # l1_loss_c = self.criterion_Idt(input_image, self.output_image)
#         # l1_loss_h = self.criterion_Idt(input_image_2, self.output_image_h)
#         # total_l1_loss = l1_loss_c + l1_loss_h
#         #
#         # GAN_output_target = self.criterion_GAN(self.netD_result_target(self.output_image), True)
#         # GAN_output_target_2 = self.criterion_GAN(self.netD_result_target(self.output_image_h), True)
#         # GAN_pred_car = self.criterion_GAN(self.netD_result_c_h(self.pred_car), True)
#         # GAN_pred_human = self.criterion_GAN(self.netD_result_c_h(self.pred_human), True)
#         # total_GAN_loss = GAN_output_target + GAN_output_target_2 + GAN_pred_car + GAN_pred_human
#         #
#         # total_loss = self.lambda_MSE * total_mse_loss + self.lambda_GAN * total_GAN_loss + self.lambda_Idt * total_l1_loss
#         # total_loss.backward()
#         # self.optimizer_G.step()
#
#
#         #
#         # self.set_requires_grad([self.netD_result_target, self.netD_result_c_h], True)
#         # self.optimizer_D_target.zero_grad()
#         # fake_target = self.Pool_fake_target.query(self.output_image)
#         # fake_target_2 = self.Pool_fake_target_2.query(self.output_image_h)
#         # self.loss_D_target = self.cal_GAN_loss_D_basic(self.netD_result_target, input_image, fake_target)
#         # self.loss_D_target_2 = self.cal_GAN_loss_D_basic(self.netD_result_target, input_image_2, fake_target_2)
#         # self.loss_D_total_t = self.loss_D_target + self.loss_D_target_2
#         # self.loss_D_total_t.backward()
#         # self.optimizer_D_target.step()
#         #
#         # self.optimizer_D_c_h.zero_grad()
#         # fake_car = self.Pool_fake_car.query(self.pred_car)
#         # fake_human = self.Pool_fake_human.query(self.pred_human)
#         # self.loss_D_car = self.cal_GAN_loss_D_basic(self.netD_result_c_h, target_image, fake_car)
#         # self.loss_D_human = self.cal_GAN_loss_D_basic(self.netD_result_c_h, target_image_2, fake_human)
#         # self.loss_D_total_c_h = self.loss_D_car + self.loss_D_human
#         # self.loss_D_total_c_h.backward()
#         # self.optimizer_D_c_h.step()
#         #
#         # self.logs.update({'loss_G':total_loss,
#         #                   'loss_D_inp_tar':self.loss_D_total_t,
#         #                   'loss_D_c_h':self.loss_D_total_c_h})
#
#
#
#
#
#
#
#
#
#
#
