import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import if_clight_renderer
import lpips


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = if_clight_renderer.Renderer(self.net)

        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.lpips = lpips.LPIPS(net='vgg')
        self.acc_crit = torch.nn.functional.smooth_l1_loss

    def forward(self, batch):
        ret = self.renderer.render(batch)

        scalar_stats = {}
        loss = 0

        mask = batch['mask_at_box']

        rgb_map = ret['rgb_map'][mask] # ((G * 32 * 32) + (F * 16 * 16), 3)
        rgb_gt = batch['rgb'][mask] # ((G * 32 * 32) + (F * 16 * 16), 3)

        img_mse = self.img2mse(rgb_map, rgb_gt)

        ########################################## LPIPS PREP ##########################################

        # Normalise to [-1, 1]
        rgb_map = (rgb_map[..., [2, 1, 0]] * 2) - 1
        rgb_gt = (rgb_gt[..., [2, 1, 0]] * 2) - 1

        # The tensor needs to be of size (G, 3, H, W) for LPIPS
        patches32 = cfg.train.n32 * 1024
        lpips_map = rgb_map[:patches32, :].view(cfg.train.n32, 32, 32, 3).permute(0, 3, 1, 2)
        lpips_gt = rgb_gt[:patches32, :].view(cfg.train.n32, 32, 32, 3).permute(0, 3, 1, 2)
        img_lpips = self.lpips.forward(lpips_map, lpips_gt) # This returns d, a legnth N tensor

        if cfg.train.n16 != 0:
            patches16 = cfg.train.n16 * 256
            lpips_map = rgb_map[patches32:patches32+patches16, :].view(cfg.train.n16, 16, 16, 3).permute(0, 3, 1, 2)
            lpips_gt = rgb_gt[patches32:patches32+patches16, :].view(cfg.train.n16, 16, 16, 3).permute(0, 3, 1, 2)
            img_lpips_2 = self.lpips.forward(lpips_map, lpips_gt) # This returns d, a legnth N tensor
            img_lpips = torch.cat((img_lpips, img_lpips_2), 1)

        if cfg.train.n8 != 0:
            patches8 = cfg.train.n8 * 64
            lpips_map = rgb_map[patches32+patches16:patches32+patches16+patches8, :].view(cfg.train.n8, 8, 8, 3).permute(0, 3, 1, 2)
            lpips_gt = rgb_gt[patches32+patches16:patches32+patches16+patches8, :].view(cfg.train.n32, 8, 8, 3).permute(0, 3, 1, 2)
            img_lpips_2 = self.lpips.forward(lpips_map, lpips_gt) # This returns d, a legnth N tensor
            img_lpips = torch.cat((img_lpips, img_lpips_2), 1)

        # compute lpips
        print(img_lpips.size())
        img_lpips = torch.mean(img_lpips) # We do the mean between the lpips patches results
        print(img_lpips)

        ########################################## LPIPS PREP ##########################################

        scalar_stats.update({'mse_loss': img_mse, 'lpips_loss': img_lpips})
        loss += (0.2 * img_mse + img_lpips)

        if 'rgb0' in ret:
            img_loss0 = self.img2mse(ret['rgb0'], batch['rgb'])
            scalar_stats.update({'img_loss0': img_loss0})
            loss += img_loss0

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
