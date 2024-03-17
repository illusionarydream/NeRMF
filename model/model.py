import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_config import *
from model.model_utils import *
from model.render_config import *

from fields.sdf_field import SDFNetwork,SingleVarianceNetwork
from fields.color_network import ColorNetwork
from fields.refract_network import RefractNetwork
from fields.nerf_density_field import NeRF

from data.data_class import RenderOutput
from camera.ray_utils import RayBundle

class Model(nn.Module):
    def __init__(self, config : ModelConfig):
        super().__init__()
        self.config = config
        
        # 1. prepare NeuS model
        
        self.sdf_network = SDFNetwork(config.sdf_network)
        self.deviation_network = SingleVarianceNetwork(
            init_val=config.deviation_network.init_val
        ) 
        
        color_d_in = 9
        self.color_network = ColorNetwork(
            config.sdf_network.d_out_feat,
            d_in=color_d_in,
            d_out=3,
            config=config.color_network
        )       
        
        
        # 2. prepare NeRF Model
        self.refract_network = RefractNetwork(
            256,
            d_in=3,
            d_out=1,
            config=config.refract_network
        )
        
        self.reflect_network = RefractNetwork(
            256,
            d_in=3,
            d_out=1,
            config=config.refract_network
        )
        
        
        self.hidden_dim_normal = 128
        self.normal_net = nn.Sequential(
                nn.Linear(256, self.hidden_dim_normal),
                nn.Linear(self.hidden_dim_normal, 3),
            )
            
        
        self.nerf_network = NeRF(
            d_in=3,
            d_in_view=3,
            config=config.nerf_network
        )
        
    
    """The most important two functions: forward and rendering_rays"""   
    # forward of the model 
    def forward(self, ray_bundle: RayBundle, is_training=False, background_rgb=None, global_step=0) -> RenderOutput:   
        
        # 0. Preparing datas
        rays_o = ray_bundle.origins
        rays_d = ray_bundle.directions
        near = ray_bundle.nears
        far = ray_bundle.fars       
       
        # 1. Extract configs
        device = rays_o.device
        batch_size = len(rays_o)
        n_samples = self.config.renderer.n_samples
        n_importance = self.config.renderer.n_importance_samples
        up_sample_steps = self.config.renderer.up_sample_steps
        
        perturb = is_training
        self.perturb = is_training
        
        geometry_warmup_state = is_training and global_step < self.config.geometry_warmup_end
        
        cos_anneal_ratio = 1.0  # default cosine annealing ratio should be 1.0
        if is_training and self.config.anneal_end > 0:
            cos_anneal_ratio = min([1.0, global_step / self.config.anneal_end])
            

        sample_dist = 2.0 / n_samples  # Assuming the region of interest is a unit sphere
        
        
        # 3. 渲染标准Neus
        z_vals = self.importance_sampler_SDF(
            n_samples,
            near,
            far,
            perturb,
            batch_size,
            n_importance,
            rays_o,
            rays_d,
            up_sample_steps
        )
                    
        render_dict = self.rendering_rays_SDF(rays_o,
                       rays_d,
                       z_vals,
                       sample_dist,
                       cos_anneal_ratio=1.0,
                       background_rgb=None,
                       geometry_warmup_state=geometry_warmup_state)
        
        
        
        # 4. 渲染标准NeRF
        z_vals_nerf = torch.linspace(0.0, 1.0, n_samples, device=device)
        z_vals_nerf = near + (far - near) * z_vals_nerf[None, :]
         
        z_vals_nerf_1 = self.importance_sampler_NeRF(
            n_samples,
            z_vals_nerf,
            perturb,
            batch_size,
            n_importance,
            rays_o,
            rays_d
        )
        
        render_dict_nerf_1 = self.rendering_rays_NeRF(
            rays_o,
            rays_d,
            z_vals_nerf_1,
            sample_dist=sample_dist,
            background_rgb=background_rgb
        )
        
        if geometry_warmup_state:
            rgb_out = render_dict_nerf_1['color']
            smoothed_normals = None
        else:
            
            smoothed_normals = render_dict['smoothed_normal'] 
            # 5. 寻找碰撞点
            hit_point,refract_d,reflect_d = self.hit_point_finder(n_samples,near,far,rays_o,rays_d,render_dict['depths'],z_vals,sample_dist)
            depths = render_dict['depths']
            
            
            # 6. 渲染反射光线
            near = 0.05 * torch.ones((batch_size,1),device=device) 
            far = 3.7 * torch.ones((batch_size,1),device=device)
            
            
            z_vals_fanshe = torch.linspace(0.0, 1.0, n_samples, device=device)
            z_vals_fanshe = near + (far - near) * z_vals_fanshe[None, :]
            
            z_vals_nerf_fanshe = self.importance_sampler_NeRF(
                n_samples,
                z_vals_fanshe,
                perturb,
                batch_size,
                n_importance,
                hit_point,
                reflect_d
            )
            
            render_dict_fanshe = self.rendering_rays_NeRF(
                hit_point,
                reflect_d,
                z_vals_nerf_fanshe,
                sample_dist=sample_dist,
                background_rgb=background_rgb
            )    
            
            
            # 7. 渲染折射光线   
            near = 0 * torch.ones((batch_size,1),device=device) 
            near2 = 0.5 * torch.ones((batch_size,1),device=device) 
            far = 3.7 * torch.ones((batch_size,1),device=device)
            
            
            z_vals_zheshe = torch.linspace(0.0, 1.0, n_samples//2, device=device)
            z_vals_zheshe = near + (far - near) * z_vals_zheshe[None, :]
            
            z_vals_nerf_zheshe = self.importance_sampler_NeRF(
                n_samples//2,
                z_vals_zheshe,
                perturb,
                batch_size,
                n_importance//2,
                hit_point,
                refract_d
            )
            
            
            render_dict_zheshe = self.rendering_rays_NeRF(
                hit_point,
                refract_d,
                z_vals_nerf_zheshe,
                sample_dist=sample_dist,
                background_rgb=background_rgb
            )

            z_vals_zheshe_back = torch.linspace(0.0, 1.0, n_samples//2, device=device)
            z_vals_zheshe_back = near2 + (far - near2) * z_vals_zheshe_back[None, :]
            
            z_vals_nerf_zheshe_back = self.importance_sampler_NeRF(
                n_samples//2,
                z_vals_zheshe_back,
                perturb,
                batch_size,
                n_importance//2,
                hit_point,
                refract_d
            )
            
            render_dict_zheshe_back = self.rendering_rays_NeRF(
                hit_point,
                refract_d,
                z_vals_nerf_zheshe_back,
                sample_dist=sample_dist,
                background_rgb=background_rgb
            )            
            
            rgb_zheshe = render_dict_zheshe['color'] * 0.5 + render_dict_zheshe_back['color'] * 0.5
            

            # 8. 颜色混合
            zheshe_mask = render_dict_nerf_1['masks1']
            fanshe_mask = render_dict_nerf_1['masks2']
            
            rgb_out = render_dict_nerf_1['color'] * (1 - fanshe_mask) + render_dict_fanshe['color'] * fanshe_mask
            rgb_out = rgb_out * ( 1 - zheshe_mask) + rgb_zheshe * zheshe_mask
            
        
        result = RenderOutput(
            rgb=rgb_out,
            mask1=render_dict_nerf_1['masks1'],
            mask2=render_dict_nerf_1['masks2'],
            #depth=render_dict['depths'],
            weights=render_dict['weights'],
            #s_val=render_dict['s_val'],
            analytic_normals=render_dict['analytic_normals'],
            rgb_neus = render_dict['color'],
            smoothed_normals=smoothed_normals,
        )
        
        return result
    # rendering rays 
    def rendering_rays_SDF(self,
                       rays_o,
                       rays_d,
                       z_vals,
                       sample_dist,
                       cos_anneal_ratio=1.0,
                       background_rgb=None,
                       geometry_warmup_state=False
                       ):
        
        device = rays_o.device
        batch_size, n_samples = z_vals.shape
        
        # 1. Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        
        dists = torch.cat([dists, torch.tensor([sample_dist], device=device).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5
        
        # 2. Section midpoints pos+dir
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        
        # 3. through the SDF network    
        sdf_nn_output = self.sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:1 + self.config.sdf_network.d_out_feat]        
        
        alpha, cdf, gradients, inv_s, neus_alpha, res_alpha, res_feat = self.get_alpha(pts, dists, dirs,
                                                                                       cos_anneal_ratio)
        alpha = alpha.reshape(batch_size, n_samples)    
        
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], device=device), 1. - alpha + 1e-7], -1), -1
        )[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)
        neus_weights = weights[:, :n_samples]          
        
        # 4. calculate the depth map and normal
        if self.config.renderer.depth_type == DepthComputationType.SphereTracing:
            hit_points, depths = self.sphere_trace(rays_o, rays_d, 2000, 1e-4, 100)
        elif self.config.renderer.depth_type == DepthComputationType.AlphaBlend:
            with torch.no_grad():
                depths = (mid_z_vals[..., :, None] * neus_weights[:, :, None]).sum(dim=1)
                hit_points = rays_o + rays_d * depths
        elif self.config.renderer.depth_type == DepthComputationType.MaximalWeightPoint:
            with torch.no_grad():
                maximum_idx = torch.argmax(neus_weights, dim=1, keepdim=True)
                depths = torch.gather(mid_z_vals, 1, maximum_idx)
                hit_points = rays_o + rays_d * depths        
                
        # normal
        analytic_normal = gradients
        normalized_normal = F.normalize(analytic_normal, dim=-1, p=2)
        
        # 5. through the color_network
        input_normal = None
        if self.config.renderer.normal_type == NormalComputationType.Analytic:
            input_normal = analytic_normal
        elif self.config.renderer.normal_type == NormalComputationType.NormalizedAnalytic:
            input_normal = normalized_normal
            
        sampled_color = self.color_network(pts, input_normal, dirs, feature_vector).reshape(batch_size, n_samples, 3)
        
        
        # 7. blend with backgroud color
        
        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        
        
        if background_rgb is not None:  # Fixed background
            color = color + background_rgb * (1.0 - weights_sum)     
            
        out =  {
            'color': color,
            'sdf': sdf,
            'analytic_normals': analytic_normal.reshape(batch_size, n_samples, 3),
            'normalized_analytic_normals': normalized_normal.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s.reshape(batch_size, n_samples),
            'weights': weights,
            'depths': depths,
        }  
        
        if geometry_warmup_state == False:
            smoothed_normal = self.normal_net(feature_vector)
            out['smoothed_normal'] = smoothed_normal.reshape(batch_size, n_samples, 3)
            
            
        return out
        
        
        
    def importance_sampler_SDF(self,n_samples,near,far,perturb,batch_size, n_importance, rays_o,rays_d,up_sample_steps):
        # importance_sampler for SDF
        device = rays_o.device
        
        z_vals = torch.linspace(0.0, 1.0, n_samples, device=device)
        z_vals = near + (far - near) * z_vals[None, :] 
        
        # perturb
        if perturb:
            t_rand = (torch.rand([batch_size, 1], device=device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / n_samples
            
        # 2 . Up sample
        # below are SDF up sample
        if n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_samples)

                for i in range(up_sample_steps):
                    new_z_vals = self.up_sample(
                                            rays_o,
                                            rays_d,
                                            z_vals,
                                            sdf,
                                            n_importance // up_sample_steps,
                                            64 * 2 ** i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == up_sample_steps))        
        return z_vals
    
    def importance_sampler_NeRF(self,n_samples,z_vals,perturb,batch_size,n_importance,rays_o,rays_d):
        # importance_sampler for NeRF
        device = rays_o.device
        
        
        # perturb
        if perturb:
            t_rand = (torch.rand([batch_size, 1], device=device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / n_samples   
            
        # 1. Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        
        sample_dist = 1e10  # inf
        dists = torch.cat([dists, torch.tensor([sample_dist], device=device).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5
        
        # 2. Section midpoints pos+dir
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)        
            
        # 3. calculate weights
        density , sampled_color, feature_vector = self.nerf_network(pts,dirs)
        
            
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        
        # 4. calculate importance samplers
        z_samples = sample_pdf(z_vals, weights[:,:-1], n_importance, det=True).detach()
        
        # combine z_samples with z_vals
        batch_size, n_samples = z_vals.shape
        _, n_importance = z_samples.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_samples[..., :, None]
        z_vals = torch.cat([z_vals, z_samples], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)
        
        return z_vals
        
           
    def rendering_rays_NeRF(self,
                       rays_o,
                       rays_d,
                       z_vals,
                       sample_dist,
                       background_rgb=None,
                       ):
        device = rays_o.device
        batch_size, n_samples = z_vals.shape
        
        # 1. Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        
        dists = torch.cat([dists, torch.tensor([sample_dist], device=device).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5
        
        # 2. Section midpoints pos+dir
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        
        # 3. through NeRF 
        density , sampled_color , feature_vector = self.nerf_network(pts,dirs)
            
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)    
        
        sampled_refract_weights = self.refract_network(pts, feature_vector).reshape(batch_size, n_samples, 1)
        refract_weights = (sampled_refract_weights * weights[:, :, None]).sum(dim=1)
        
        sampled_reflect_weights = self.reflect_network(pts, feature_vector).reshape(batch_size, n_samples, 1)
        reflect_weights = (sampled_reflect_weights * weights[:, :, None]).sum(dim=1)
        
        depths = (weights * mid_z_vals).sum(dim=1)
        
        
        # 4. through the normal net
        analytic_normals = self.nerf_network.gradient(pts)
        
        return {
            'color': color,
            'masks1': refract_weights,
            'masks2': reflect_weights,
            'analytic_normals': analytic_normals.reshape(batch_size, n_samples, 3),
            'weights': weights,
            'depths': depths
        }  
        
        
    def rendering_rays_Normal(self,
                       rays_o,
                       rays_d,
                       z_vals,
                       sample_dist,
                       background_rgb=None,):
        device = rays_o.device
        batch_size, n_samples = z_vals.shape
        
        # 1. Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        
        dists = torch.cat([dists, torch.tensor([sample_dist], device=device).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5
        
        # 2. Section midpoints pos+dir
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        
        # 3. through NeRF 
        feature_vector = self.nerf_network.feature(pts)
        smoothed_normals = self.normal_network(pts,feature_vector)
        with torch.no_grad():
            analytic_normals = self.sdf_network.gradient(pts)
        
        
        return {
            'smoothed_normals':smoothed_normals.reshape(batch_size, n_samples, 3),
            'analytic_normals':analytic_normals.reshape(batch_size, n_samples, 3),
        }
        
    def hit_point_finder(self,n_samples,near,far,rays_o,rays_d,depths,z_vals,sample_dist):
        
        epsilon_1 = 1
        epsilon_2 = 1.470
        
        n1_n2 = epsilon_1 / epsilon_2
        
        # 1) find the second depths
        with torch.no_grad():
            hit_point_1 = rays_o + rays_d * depths
            z_vals_left = torch.clamp(far - near - depths,min=0)
            
            
            analytic_normal = self.sdf_network.gradient(hit_point_1)
            analytic_normal = analytic_normal.squeeze()
            normalized_normal = F.normalize(analytic_normal, dim=-1, p=2)
            
            sdf_nn_output = self.sdf_network(hit_point_1)
            feature_vector = sdf_nn_output[:, 1:1 + self.config.sdf_network.d_out_feat]        
            
        #feature_vector = self.nerf_network.feature(hit_point_1)
        smoothed_normals = self.normal_net(feature_vector)
        normalized_smoothed_normals = F.normalize(smoothed_normals, dim=-1, p=2)
        
        
        normal = normalized_smoothed_normals
        #normal = normalized_normal
        L = rays_d
        
        refract_d = self.refract_d_calculator(L,normal)
        
        reflect_d = self.reflect_d_calculator(L,normal)
        
        return hit_point_1,refract_d,reflect_d
        
        
        
    """utils function"""    
    # up sample in Neus
    @staticmethod
    def up_sample(rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s 
        use SDF to get importance sampling
        """
        batch_size, n_samples = z_vals.shape
        device = rays_o.device
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], device=device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples       
        
    # cat z vals
    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf
    
    # get alpha from SDF
    def get_alpha(self, pts, dists, dirs, cos_anneal_ratio=1.):
        device = pts.device
        sdf = self.sdf_network.sdf(pts)
        gradients = self.sdf_network.gradient(pts).squeeze()
        inv_s = self.deviation_network(torch.zeros([1, 3], device=device))[:, :1].clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.expand(sdf.size(0), 1)
        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0, 1)
        return alpha, c, gradients, inv_s, alpha, None, None    
    
    # used in calculate depth
    @torch.no_grad()
    def sphere_trace(self, rays_o, rays_d, num_iterations, convergence_threshold, far):
        
        device = rays_o.device
        pts = rays_o
        depths = torch.zeros((pts.size(0), 1), device=device)
        for _ in range(num_iterations):
            sdf = self.sdf_network.sdf(pts)  # [N, 1]
            converged = (torch.abs(sdf) < convergence_threshold) | (depths > far)  # [N, 1]
            pts = torch.where(converged, pts, pts + sdf * rays_d)  # [N, 3]
            depths = torch.where(converged, depths, depths + sdf)  # [N, 1]
            if converged.all():
                break
        return pts, depths
    
    def refract_d_calculator(self,L,normal):
        """
        cos_2^2>0 return refract_d
        cos_2^2<0 return reflect_d
        """
        batch_size = L.shape[0]
        
        epsilon_1 = 1 # air
        epsilon_2 = 1.470 # glass
    
    
        # 1） checking normal and n1_n2
        cos = torch.sum(normal*L,dim=1)
        normal[cos>0,:] = - normal[cos>0,:]
        n1_n2 = torch.full_like(cos,epsilon_1/epsilon_2)
        n1_n2[cos>0] = epsilon_2 / epsilon_1
        
        cos = torch.sum(normal*L,dim=1)
        
        
        cos_2_check = 1 - n1_n2**2*(1-(cos)**2)
        
        out_dir = torch.full_like(L,1)
        
        # cos_2^2<0 return reflect_d
        
        R = L - 2 * cos[:,None] * normal
        
        out_dir[cos_2_check<0,:] = R[cos_2_check<0,:]
        
        # cos_2^2>0 return refract_d
        
        
        cos_1 = torch.abs(cos[cos_2_check>=0])
        cos_2 = torch.sqrt(cos_2_check[cos_2_check>=0])
        
                
        out_dir[cos_2_check>=0,:] = -cos_2[:,None] * normal[cos_2_check>=0] + n1_n2[cos_2_check>=0,None]*( L[cos_2_check>=0,:] + cos_1[:,None] * normal[cos_2_check>=0,:])
        
        return out_dir
    
    
    def reflect_d_calculator(self,L,normal):
       cos = torch.sum(normal*L,dim=1)
       R = L - 2 * cos[:,None] * normal
       return R