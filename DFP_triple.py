from net import skip,skip_mask
from net.losses import ExclusionLoss, plot_image_grid, StdLoss, GradientLoss,MS_SSIM,tv_loss
from net.noise import get_noise
from utils.image_io import *
from utils.segamentation import k_means
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
import torch
from collections import namedtuple

from net.downsampler import Downsampler


matplotlib.use('TkAgg')
All_in_focus_sr_step_1_result = namedtuple("All_in_focus_sr",
                                       ["all_in_focus","all_in_focus_sr", "psnr", "alpha1", "alpha2", "out1", "out2"])
All_in_focus_sr_step_2_result = namedtuple("All_in_focus_sr",["psnr", "alpha1", "alpha2"])
All_in_focus_sr_step_3_result = namedtuple("All_in_focus_sr",
                                       ["all_in_focus","all_in_focus_sr", "psnr", "out1", "out2"])
data_type = torch.cuda.FloatTensor

class Deep_fusion_prior_triple(object):
    def __init__(self, image1_name, image2_name, image1, image2,image3,GT1,GT2,dict, plot_during_training=True,
                 show_every=100,
                 num_iter=4000, factor=4,outpath='',
                 original_reflection=None, original_transmission=None):
        # we assume the reflection is static
        self.image1 = image1
        self.image2 = image2
        self.image3 = image3

        self.GT1 = np_to_torch(GT1)
        self.GT2 = np_to_torch(GT2)
        self.dict = dict
        self.outpath = outpath


        # self.input = input
        self.factor = factor
        self.plot_during_training = plot_during_training
        self.psnrs = []
        self.show_every = show_every
        self.image1_name = image1_name
        self.image2_name = image2_name

        self.num_iter = num_iter
        self.loss_function = None
        self.parameters = None
        self.learning_rate = 0.001
        self.input_depth = 3
        self.reflection_net_input = None
        self.transmission_net_input = None
        self.original_transmission = original_transmission
        self.original_reflection = original_reflection
        self.reflection_net = None
        self.transmission_net = None
        self.total_loss = None
        self.multiscale_loss = None
        self.multiscale = None
        self.reflection_out = None
        self.transmission_out = None
        self.current_result = None
        self.best_result = None
        self._init_all()

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        self.image1_torch = np_to_torch(self.image1).type(torch.cuda.FloatTensor)
        self.image2_torch = np_to_torch(self.image2).type(torch.cuda.FloatTensor)
        self.image3_torch = np_to_torch(self.image3).type(torch.cuda.FloatTensor)
        # self.input_torch = np_to_torch(self.input).type(torch.cuda.FloatTensor)

    def _init_inputs(self):

        data_type = torch.cuda.FloatTensor
        input1 = np_to_pil(self.image1)
        input2 = np_to_pil(self.image2)
        input3 = np_to_pil(self.image3)
        input1 = pil_to_np(input1.resize((input1.size[0] * self.factor, input1.size[1] * self.factor), Image.BICUBIC))
        input2 = pil_to_np(input2.resize((input2.size[0] * self.factor, input2.size[1] * self.factor), Image.BICUBIC))
        input3 = pil_to_np(input3.resize((input3.size[0] * self.factor, input3.size[1] * self.factor), Image.BICUBIC))

        self.input_bicubic_1 = np_to_torch(input1).type(data_type)
        self.input_bicubic_2 = np_to_torch(input2).type(data_type)
        self.input_bicubic_3 = np_to_torch(input3).type(data_type)
        self.all_in_focus_input = np_to_torch((input1 + input2+ input3) / 3).type(data_type)

        self.label1 = np_to_torch(
            cmp_PSF(rgb2y_CWH_nol(self.image1), rgb2y_CWH_nol(self.image2), 5, int(self.dict[0]), int(self.dict[1]),
                    int(self.dict[2]), self.dict[3], self.dict[4])).type(data_type)

        self.label2 = np_to_torch(
            cmp_PSF(rgb2y_CWH_nol(self.image2), rgb2y_CWH_nol(self.image1), 5, int(self.dict[0]), int(self.dict[1]),
                    int(self.dict[2]), self.dict[3], self.dict[4])).type(data_type)
        intersection = self.label1 + self.label2
        intersection[intersection <= 1] = 0
        intersection[intersection > 1] = 1
        self.label1 = abs(self.label1 - intersection)
        self.label2 = abs(self.label2 - intersection)
        self.label3 = abs(1 - self.label1 - self.label2)

        self.current_alpha1 = self.label1
        self.current_alpha2 = self.label2
        self.current_alpha3 = abs(1 - self.label1 - self.label2)

    def _init_parameters(self):
        self.parameters = None

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'
        KERNEL_TYPE = 'lanczos2'
        all_in_focus_net = skip(
            self.input_depth, self.input_depth,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[4, 4, 4, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.reflection_net = all_in_focus_net.type(data_type)

        downsampler = Downsampler(n_planes=self.input_depth, factor=self.factor, kernel_type=KERNEL_TYPE, phase=0.5,
                                  preserve_size=True).type(data_type)
        self.downsampler = downsampler.type(data_type)


    def _init_losses(self):

        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.l1_loss = torch.nn.L1Loss().type(data_type)
        self.exclusion_loss = ExclusionLoss().type(data_type)
        self.blur_loss = StdLoss().type(data_type)
        self.gradientloss = GradientLoss().type(data_type)
        self.ms_ssim_loss = MS_SSIM(max_val=1)

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.parameters = [p for p in self.reflection_net.parameters()]
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        print("Start stage 3: global approaching... ")
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure3(j)
            self._obtain_current_result(j)
            self._plot_closure(j)
            optimizer.step()
        print("Done! ")


    def _optimization_closure3(self, step):
        reg_noise_std = 0.00

        all_in_focus_net_input = self.all_in_focus_input + (self.all_in_focus_input.clone().normal_() * reg_noise_std)
        self.all_in_focus_out_sr = self.reflection_net((self.input_bicubic_1+self.input_bicubic_2+self.input_bicubic_3)/3)
        self.all_in_focus_out =self.all_in_focus_out_sr #self.downsampler(self.all_in_focus_out_sr)

        out_y, _, _ = rgb2y_CWH_nol_torch(self.all_in_focus_out)

        image1 = np_to_pil(torch_to_np(self.image1_torch)).resize(
            (self.all_in_focus_out.shape[3], self.all_in_focus_out.shape[2]), Image.BICUBIC)
        image2 = np_to_pil(torch_to_np(self.image2_torch)).resize(
            (self.all_in_focus_out.shape[3], self.all_in_focus_out.shape[2]), Image.BICUBIC)
        image3 = np_to_pil(torch_to_np(self.image3_torch)).resize(
            (self.all_in_focus_out.shape[3], self.all_in_focus_out.shape[2]), Image.BICUBIC)
        self.image1_torch = np_to_torch(pil_to_np(image1)).type(torch.cuda.FloatTensor)
        self.image2_torch = np_to_torch(pil_to_np(image2)).type(torch.cuda.FloatTensor)
        self.image3_torch = np_to_torch(pil_to_np(image3)).type(torch.cuda.FloatTensor)

        image1_y, _, _ = rgb2y_CWH_nol_torch(self.image1_torch)
        image2_y, _, _ = rgb2y_CWH_nol_torch(self.image2_torch)
        image3_y, _, _ = rgb2y_CWH_nol_torch(self.image3_torch)

        self.input_joint_grads, self.all_in_focus_out_grad = joint_grad_triple(out_y, image1_y, image2_y, image3_y)
        self.total_loss = 0.5*self.l1_loss(self.input_joint_grads, self.all_in_focus_out_grad)


        self.total_loss += self.l1_loss(self.current_alpha1 * self.all_in_focus_out,
                                       self.current_alpha1 * self.image1_torch)
        self.total_loss += self.l1_loss(self.current_alpha2 * self.all_in_focus_out,
                                        self.current_alpha2 * self.image2_torch)
        self.total_loss += self.l1_loss(self.current_alpha3 * self.all_in_focus_out,
                                        self.current_alpha3 * self.image3_torch)

        #self.total_loss +=  0.1*self.gradientloss(self.all_in_focus_out_sr)
        """


        self.total_loss += self.l1_loss(self.label1 * self.all_in_focus_out,
                                           self.label1 * self.image1_torch)
        self.total_loss += self.l1_loss(self.label2 * self.all_in_focus_out,
                                            self.label2 * self.image2_torch)
        """
        self.total_loss.backward()

    def _obtain_current_result(self, j):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        all_in_focus_out = np.clip(torch_to_np(self.all_in_focus_out), 0, 1)
        all_in_focus_sr_out = np.clip(torch_to_np(self.all_in_focus_out_sr), 0, 1)

        out1 = np.clip(torch_to_np(self.current_alpha1), 0, 1)
        out2 = np.clip(torch_to_np(self.current_alpha2), 0, 1)

        GT1 = np.clip(torch_to_np(self.GT1), 0, 1)
        GT2 = np.clip(torch_to_np(self.GT2), 0, 1)


        # print(out1.shape, self.image1.shape)
        psnr1 = compare_psnr(all_in_focus_sr_out,GT1)
        psnr2 = compare_psnr(all_in_focus_sr_out,GT2)


        self.psnrs.append((psnr1 + psnr2) / 2)
        self.current_result = All_in_focus_sr_step_3_result(all_in_focus=all_in_focus_out,
                                                        all_in_focus_sr=all_in_focus_sr_out,
                                                        psnr=((psnr1 + psnr2 ) / 2), out1=out1, out2=out2)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result

    def _plot_closure(self, step):  # Exclusion {:5f} self.exclusion.item(),
        print('Iteration {:5d}    Loss {:5f} PSRN_gt: {:f}'.format(step, self.total_loss.item(),
                                                                   self.current_result.psnr), '\r', end='')
        if self.plot_during_training and step % self.show_every == self.show_every - 1:
            # plot_image_grid("all_in_focus{}".format(step),
            #                 [self.current_result.reflection, self.current_result.transmission])
            # plot_image_grid("learned_mask_{}".format(step),
            #                 [self.current_result.alpha1, self.current_result.alpha2])
            save_image("all_in_focus_{}".format(step), self.current_result.all_in_focus)
            save_image("all_in_focus_sr{}".format(step), self.current_result.all_in_focus_sr)

    def finalize(self):
        outpath = "output/"+self.outpath+"/"
        save_graph("result" + "_psnr", self.psnrs,output_path=outpath)
        save_image("result" + "_all_in_focus", self.best_result.all_in_focus,output_path=outpath)
        save_image("result" + "_all_in_focus_srx4", self.best_result.all_in_focus_sr,output_path=outpath)
        save_image("result" + "_label_foreground", self.best_result.out1,output_path=outpath)
        save_image("result" + "_label_background", self.best_result.out2,output_path=outpath)
        print(self.outpath+" process done!")


if __name__ == "__main__":

    import os
    dict =[[3, 3, 3, 0.010,1]]
    outpath = "/test"
    input_fore = prepare_image('./images/'+'z1.jpg')
    input_back = prepare_image('./images/'+'z3.jpg')
    input_mid = prepare_image('./images/'+'z2.jpg')

    f=1

    input1_pil = np_to_pil(input_fore)
    input1_down = input1_pil.resize((input1_pil.size[0] // f, input1_pil.size[1] // f), Image.BICUBIC)
    input1_bicubic = pil_to_np(input1_down.resize((input1_down.size[0] * 4, input1_down.size[1] * 4), Image.BICUBIC))
    input2_pil = np_to_pil(input_back)
    input2_down = input2_pil.resize((input2_pil.size[0] // f, input2_pil.size[1] // f), Image.BICUBIC)
    input2_bicubic = pil_to_np(input2_down.resize((input2_down.size[0] * 4, input2_down.size[1] * 4), Image.BICUBIC))
    input3_pil = np_to_pil(input_mid)
    input3_down = input3_pil.resize((input3_pil.size[0] // f, input3_pil.size[1] // f), Image.BICUBIC)
    input3_bicubic = pil_to_np(input3_down.resize((input3_down.size[0] * 4, input3_down.size[1] * 4), Image.BICUBIC))
    """
    save_image("Bicubic" + "_Ax4", input1_bicubic, output_path='output/'+outpath+'/')
    save_image("Bicubic" + "_Bx4", input1_bicubic, output_path='output/' + outpath + '/')
    """
    step3 = Deep_fusion_prior_triple('input1', 'input2', pil_to_np(input1_down), pil_to_np(input2_down), pil_to_np(input3_down),input_fore,input_back,dict[0],
                                   plot_during_training=True, num_iter=2000, factor=f,outpath=outpath)
    step3.optimize()
    step3.finalize()