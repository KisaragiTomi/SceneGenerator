"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))


def get_current_visuals(self):
    visuals = {}
    # 将 ID 图转为可视化的单通道 label
    # 1. 真实 ID
    visuals['real_id'] = util.tensor2label(self.real_image['id_gt'].unsqueeze(1), 16)

    # 2. 预测 ID (取 Argmax)
    if hasattr(self, 'fake_image'):  # fake_image 这里存的是 40 通道概率图
        # 前 16 通道是 ID
        pred_id_logits = self.fake_image[:, :16, :, :]
        pred_id_idx = torch.argmax(pred_id_logits, dim=1, keepdim=True)
        visuals['pred_id'] = util.tensor2label(pred_id_idx, 16)

    return visuals


# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
