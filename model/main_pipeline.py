
from typing import Tuple, Dict, List, Iterator, Union

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from configs.main_config import SystemConfig
from data.data_manager import DataManager
from camera.ray_generator import RayGenerator
from model.model import Model
from utils.metrics import PSNR, SSIM, LPIPS
from utils.tensor_dataclass import td_concat

from data.data_class import RawPixelBundle, RenderOutput


class MainPipeline(nn.Module):
    """
    Pipeline wrapper model for training and testing the model.
    """
    
    def __init__(self, config : SystemConfig, data_manager : DataManager):
        super().__init__()
        self.config = config
        
        # 1. init the RayGenerator
        self.ray_generator = RayGenerator(
            camera=data_manager.camera,
            num_cameras=data_manager.total_image_num,
            config=config.ray_generator
        )  
        # 2. init the model
        self.model = Model(config=config.model)
        
        
    def get_param_groups(self) -> List[Dict[str, Union[Iterator[torch.nn.Parameter], float]]]:
        """Return parameter groups with hyper params."""
        
        # if need to optimize the ray_generator
        #return [
        #    {'params': self.renderer.parameters(), 'lr': self.config.model.lr},
        #    {'params': self.ray_generator.parameters(), 'lr': self.config.ray_generator.opt_lr}
        #]      
        return [
            {'params': self.model.parameters(), 'lr': self.config.model.lr},
        ]
        
    def forward(self, pixel_bundle: RawPixelBundle, global_step: int = 0) -> RenderOutput:
        ray_bundle = self.ray_generator.forward(pixel_bundle)
        rendering_res = self.model.forward(
            ray_bundle,
            background_rgb=torch.ones([1, 3], device=pixel_bundle.poses.device) if self.config.data.white_background
            else torch.zeros([1, 3], device=pixel_bundle.poses.device),
            is_training=True,
            global_step=global_step
        )
        
        return rendering_res         
    
    # important: Loss Function    
    def get_train_loss_dict(self, rendering_res: RenderOutput, pixel_bundle: RawPixelBundle) -> dict:
        """
        Compute training loss and other metrics, return them in a dict.
        :param rendering_res: RenderOutput, rendering result
        :param pixel_bundle: RawPixelBundle, input data
        :return: dict, loss dict
        """
        
        batch_size , sample_points, _ = rendering_res.analytic_normals.shape
        
        # 下面是NeRF的loss
        rgb_loss = nn.functional.l1_loss(rendering_res.rgb, pixel_bundle.rgb_gt, reduction='sum') / \
                   (rendering_res.rgb.size(0) + 1e-5)
                   
        mask1_loss = nn.functional.l1_loss(rendering_res.mask1, pixel_bundle.mask1s[:, 0:1], reduction='sum') / \
                   (rendering_res.rgb.size(0) + 1e-5)
        
        mask2_loss = nn.functional.l1_loss(rendering_res.mask2, pixel_bundle.mask2s[:, 0:1], reduction='sum') / \
                   (rendering_res.rgb.size(0) + 1e-5)       
           
        # 下面是neus的loss         
        gradient_error = (torch.linalg.norm(rendering_res.analytic_normals, ord=2, dim=-1) - 1.0) ** 2
        eikonal_loss = gradient_error.sum() / (batch_size * sample_points)
        
        sdf_loss = nn.functional.l1_loss(rendering_res.rgb_neus, pixel_bundle.rgb_gt, reduction='sum') / \
                   (rendering_res.rgb.size(0) + 1e-5)

        loss = rgb_loss + mask1_loss * self.config.model.mask_weight + mask2_loss * self.config.model.mask_weight + sdf_loss + eikonal_loss * self.config.model.igr_weight
        
        loss_dict = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'mask1_loss': mask1_loss,
            'mask2_loss': mask2_loss,
            'sdf_loss' : sdf_loss,
            'eikonal_loss': eikonal_loss,
            'psnr': PSNR(rendering_res.rgb, pixel_bundle.rgb_gt),
        }


        # 下面是法线损失
        if rendering_res.smoothed_normals is not None:
            smoothed_error = (torch.linalg.norm(rendering_res.analytic_normals - rendering_res.smoothed_normals, ord=2, dim=-1)) ** 2
            smoothed_loss = smoothed_error.sum() / (batch_size * sample_points)
            loss = loss + 0.1 * smoothed_loss
            loss_dict['smoothed_loss'] = smoothed_loss
            loss_dict['loss'] = loss
         
                
        return loss_dict
        
    """Eval Functions"""
    @torch.no_grad()
    def get_eval_dicts(self, img_pixel_bundle: RawPixelBundle, device: torch.device) -> \
            Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, np.ndarray]]:
        """
        Run evaluation on a single image
        :param device: trainer device
        :param img_pixel_bundle: pixel bundle of a single image, [H, W]
        :return: img_dict, metrics_dict, tensor_dict
        """

        flat_pixel_bundle = img_pixel_bundle.flatten()
        has_gt = flat_pixel_bundle.rgb_gt is not None

        chunk = self.config.model.inference_chunk_size
        rets = []
        for i in range(0, flat_pixel_bundle.shape[0], chunk):
            ray_bundle = self.ray_generator.forward(flat_pixel_bundle[i:i + chunk].to(device))
            rendering_res = self.model.forward(
                ray_bundle,
                background_rgb=torch.ones([1, 3], device=device)
                if self.config.data.white_background else torch.zeros([1, 3], device=device),
                is_training=False
            )
            rets.append(rendering_res.to('cpu'))

        torch.set_default_tensor_type('torch.FloatTensor')
        ret = td_concat(rets)
        ret = ret.reshape(img_pixel_bundle.shape)
        rot = torch.linalg.inv(img_pixel_bundle.poses[0, 0, :3, :3])
        
        
        analytic_normals = torch.einsum('...ij,...i->...j', ret.analytic_normals, ret.weights)
        analytic_normals = torch.matmul(rot[None, :], analytic_normals.reshape((-1, 3))[:, :, None]).reshape(
            ret.rgb.shape)
        
        """
        normalized_analytic_normals = torch.einsum('...ij,...i->...j', ret.normalized_analytic_normals,
                                                   ret.weights)
        normalized_analytic_normals = torch.matmul(rot[None, :],
                                                   normalized_analytic_normals.reshape((-1, 3))[:, :, None]).reshape(
            ret.rgb.shape)
        """

        img_dict = {
            'rgb': ret.rgb.detach().cpu().numpy(),
            'analytic_normals': analytic_normals.detach().cpu().numpy(),
            #'normalized_analytic_normals': normalized_analytic_normals.detach().cpu().numpy(),
            #'smoothed_normals': smoothed_normals.detach().cpu().numpy(),
            'mask1': ret.mask1.repeat(1,1,3).detach().cpu().numpy(),
            'mask2': ret.mask2.repeat(1,1,3).detach().cpu().numpy()
        }
        if has_gt:
            img_dict['rgb_gt'] = img_pixel_bundle.rgb_gt.detach().cpu().numpy()
             
        metrics_dict = {
            'psnr': PSNR(ret.rgb, img_pixel_bundle.rgb_gt),
            'ssim': SSIM(ret.rgb, img_pixel_bundle.rgb_gt),
            'lpips': LPIPS(ret.rgb, img_pixel_bundle.rgb_gt),
        } if has_gt else {}
        
        '''
        tensor_dict = {
            'depth': ret.depth.detach().cpu().numpy(),
        }
        '''
        
        tensor_dict = {}
        
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        return img_dict, metrics_dict, tensor_dict