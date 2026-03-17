"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.networks as networks
import util.util as util
import torch.nn.functional as F
from torchinfo import summary


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = label_map.float()

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        background_value = -1.0
        mask = (real_image > (background_value + 1e-5)).float()
        diff = torch.abs(fake_image - real_image)
        l1_loss = (diff * mask).sum() / (mask.sum() + 1e-7)

        # lambda_l1 = getattr(self.opt, 'lambda_l1', 100.0)
        G_losses['L1_River'] = l1_loss * 100.0

        # if not self.opt.no_vgg_loss:
        #     G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
        #         * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netG(input_semantics, z=z)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


class MultiClassDiceLoss(torch.nn.Module):
    def __init__(self, num_classes=16, ignore_index=0):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index  # 极其关键：忽略背景，只算物体的 Dice

    def forward(self, logits, targets):

        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dice_loss = 0.0
        valid_classes = 0

        # 3. 逐个类别计算 Dice
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue  # 跳过背景类 (比如 ID=0)

            p_c = probs[:, c, :, :]
            t_c = targets_one_hot[:, c, :, :]

            # 计算交集和并集
            intersection = (p_c * t_c).sum()
            union = p_c.sum() + t_c.sum()

            # 加上 1e-5 平滑项，防止除以 0 (比如这张图里刚好没有第 c 类物体)
            dice_c = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)

            dice_loss += dice_c
            valid_classes += 1

        # 4. 返回所有有效类别 Dice Loss 的平均值
        return dice_loss / valid_classes


class MultiClassTverskyLoss(torch.nn.Module):
    # num_classes 改成了 3，alpha 设为 0.9 (极度严惩多画线！)
    def __init__(self, num_classes=3, ignore_index=0, alpha=0.9, beta=0.1):
        super(MultiClassTverskyLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.alpha = alpha  # 0.9 意味着多画一个噪点的惩罚是漏掉一个红点的 9 倍！
        self.beta = beta

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        # 将 targets 转为 One-Hot
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        tversky_loss = 0.0
        valid_classes = 0

        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue

            p_c = probs[:, c, :, :]
            t_c = targets_one_hot[:, c, :, :]

            # TP (猜对), FP (多画), FN (漏画)
            TP = (p_c * t_c).sum()
            FP = (p_c * (1.0 - t_c)).sum()
            FN = ((1.0 - p_c) * t_c).sum()

            # Tversky 公式
            tversky_c = 1.0 - (TP + 1e-5) / (TP + self.alpha * FP + self.beta * FN + 1e-5)
            tversky_loss += tversky_c
            valid_classes += 1

        return tversky_loss / valid_classes

class ObjectPlacementModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        self.output_nc = opt.output_nc
        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        self.criterion_Tversky = MultiClassTverskyLoss(num_classes=3, ignore_index=0, alpha=0.9, beta=0.1)

        # set loss functions
        if opt.isTrain:
            from torchinfo import summary
            input_shape = (1, self.opt.label_nc, 256, 256)
            device = next(self.netG.parameters()).device
            dummy_input_segmap = torch.randn(input_shape).to(device)
            # print("---------- Generator Summary ----------")
            # # 注意：SPADE 生成器通常需要 segmap 作为输入
            # summary(self.netG, input_data=dummy_input_segmap)
            #

            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            # if not opt.no_vgg_loss:
                # self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)


            class_weights = torch.ones(self.output_nc, dtype=torch.float32)
            class_weights[0] = 0.05
            class_weights = class_weights.cuda()
            self.criterionCE = torch.nn.CrossEntropyLoss(weight=class_weights)
            self.criterionDice = MultiClassDiceLoss(num_classes=self.output_nc)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)
        #R Channel
        idmap = real_image[:, 0:1, :, :]

        idmap = (idmap * 255).long()
        
        bs, _, h, w = idmap.size()
        input_label = torch.zeros(bs, self.output_nc, h, w, device=idmap.device)
        idmap = torch.clamp(idmap, 0, (self.output_nc - 1))
        # target_id = input_label.scatter_(1, idmap, 1.0)
        target_id_long = idmap.squeeze(1)
        # print("真实标签包含的 ID:", torch.unique(real_image).tolist())
        # G Channel

        rotmap = real_image[:, 1:2, :, :].long()
        bs, _, h, w = rotmap.size()
        input_label = torch.zeros(bs, 24, h, w, device=rotmap.device)
        # rotmap = torch.clamp(idmap, 0, 23) # Fixed potential bug: using rotmap instead of idmap
        rotmap = torch.clamp(rotmap, 0, 23)
        target_rot = input_label.scatter_(1, rotmap, 1.0)
        valid_mask = (target_rot > 0)

        # real_concat = torch.cat([target_id_long, target_rot], dim=1)
        real_concat = idmap.long()
        real_concat = real_concat.squeeze(1)
        # real_concat = target_id
        if mode == 'generator':
            g_loss, generated, real_image_id = self.compute_generator_loss(
                input_semantics, target_id_long)
            return g_loss, generated, real_image_id
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_concat)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_concat)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_concat)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = torch.zeros(bs, nc, h, w, device=label_map.device)
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image']

    def downsample_target(self, pred, target):

        mid_h, mid_w = pred.shape[2:]
        # interpolate 需要 4D 张量 (B, C, H, W)
        if target.dim() == 3:
            # 如果是 (B, H, W)，先升维变 (B, 1, H, W)
            target_4d = target.unsqueeze(1).float()
        else:
            target_4d = target.float()

        # 执行插值
        downsampled = F.interpolate(target_4d, size=(mid_h, mid_w), mode='nearest')
        downsampled = downsampled.long()

        if target.dim() == 3:
            downsampled = downsampled.squeeze(1)

        loss = self.criterionCE(pred, downsampled)
        return loss, downsampled
    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        # fake_image, KLD_loss = self.generate_fake(
        #     input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        fake_result, KLD_loss = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae)
        pred_id = fake_result['id']
        # pred_rot = fake_result['rot']
        pred_mid = fake_result['aux_mid']
        pred_shallow = fake_result['aux_shallow']


        # target_id = real_image[:, 0:16, :, :]
        target_id = real_image
        tversky_loss = self.criterion_Tversky(pred_id, target_id)
        G_losses['G_ID'] = self.criterionCE(pred_id, target_id) + tversky_loss
        # target_rot = real_image[:, 16:40, :, :]
        # G_losses['G_Rot'] = self.criterionCE(pred_rot, target_rot)
        G_losses['mid'], target_id_mid = self.downsample_target(pred_mid, target_id)
        G_losses['shallow'], target_id_shallow = self.downsample_target(pred_shallow, target_id)

        dice_loss_final = self.criterionDice(pred_id, target_id)
        # 对于浅层特征，为了节省计算量，你可以只算 CE；如果想极致对齐，也可以加上 Dice
        dice_loss_mid = self.criterionDice(pred_mid, target_id_mid)
        dice_loss_shallow = self.criterionDice(pred_shallow, target_id_shallow)

        # G_losses['G_ID'] += dice_loss_final
        # G_losses['mid'] += dice_loss_mid
        # G_losses['shallow'] += dice_loss_shallow

        G_losses['mid'] *= .5
        G_losses['shallow'] *= .25

        # G_losses['dice_G_ID'] = dice_loss_final
        # G_losses['dice_mid'] = dice_loss_mid
        # G_losses['dice_shallow'] = dice_loss_shallow

        prob_id = torch.nn.functional.gumbel_softmax(pred_id, dim=1, tau=1.0, hard=True)
        target_id_onehot = F.one_hot(
            target_id.long(), num_classes=self.output_nc
        ).permute(0, 3, 1, 2).float()
        pred_fake, pred_real = self.discriminate(
            input_semantics, prob_id, target_id_onehot)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = torch.zeros(1, device=pred_id.device)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        return G_losses, prob_id, real_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_result, _ = self.generate_fake(input_semantics, real_image)
            pred_id = fake_result['id']
            fake_concat = torch.nn.functional.gumbel_softmax(
                pred_id, dim=1, tau=1.0, hard=True
            ).detach()
        target_id = F.one_hot(
            real_image.long(), num_classes=self.output_nc
        ).permute(0, 3, 1, 2).float()
        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_concat, target_id)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netG(input_semantics, z=z)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
