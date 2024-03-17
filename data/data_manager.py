from data.data_config import DataManagerConfig
from data.data_loader import read_data
import torch
from data.data_class import RawPixelBundle
from data.data_config import PixelSamplingStrategy
import numpy as np
from camera.camera_model import CameraModel

from typing import Optional


class PixelSampler(object):
    """
    Sample rays (pixels) from images.
    """
    def __init__(
            self,
            num_image_per_split: list,
            total_image_num: int,
            camera: CameraModel,
            batch_size: int,
            image_idx_rng_seed: int = 42,
            pixel_idx_rng_seed: int = 42,
            strategy: PixelSamplingStrategy = PixelSamplingStrategy.SAME_IMAGE,
            training_view_num_limit: Optional[int] = None,
    ):
        """
        Initialize ray sampler.
        :param num_image_per_split: ["train","val","test"] for deafult
        :param camera: CameraModel
        :param batch_size:
        :param image_idx_rng_seed: set same seed on each process so that they sample the same image
        :param pixel_idx_rng_seed: process rank will be added to ensure each process sample a different batch
        :param strategy: RaySamplingStrategy
        """

        self.batch_size = batch_size
        self.train_image_num = num_image_per_split[0] if training_view_num_limit is None \
            else training_view_num_limit
            
        self.H, self.W = camera.H, camera.W
        
        self.total_image_num = total_image_num
        self.strategy = strategy
        
        self.image_rng = np.random.default_rng(seed=image_idx_rng_seed)
        self.pixel_rng = np.random.default_rng(seed=pixel_idx_rng_seed)
        
        print(f'image_rng_seed: {image_idx_rng_seed}, pixel_rng_seed: {pixel_idx_rng_seed}')

    def sample_batch(self):
        if self.strategy == PixelSamplingStrategy.ALL_IMAGES:
            img_indices = self.image_rng.choice(self.train_image_num, self.batch_size)
        elif self.strategy == PixelSamplingStrategy.SAME_IMAGE:
            img_indices = self.image_rng.choice(self.train_image_num, 1)
            img_indices = np.repeat(img_indices, self.batch_size)
        else:
            raise NotImplementedError
        h_indices = self.pixel_rng.choice(self.H, self.batch_size)
        w_indices = self.pixel_rng.choice(self.W, self.batch_size)

        return img_indices, h_indices, w_indices

class DataManager():
    """
    DataManager Class
    """
    def __init__(self,batch_size:int,seed:int,data_config:DataManagerConfig):
        self.batch_size = batch_size
        self.data_configs = data_config
        self.seed = seed
        
        # read datas
        all_data = read_data(
            basedir=data_config.path,
            half_res=data_config.half_res,
            white_background=data_config.white_background
        )
        
        self.imgs = all_data["imgs"]
        self.poses = all_data["poses"]
        self.mask1s = all_data["mask1s"]
        self.mask2s = all_data["mask2s"]
        self.camera = all_data["camera"]
        self.num_image_per_split = all_data["num_image_per_split"]   #list["train","val","test"] for deafult
        self.total_image_num = all_data["total_image_num"]
        
        
        
        # build image sampler for random pick pixels
        self.sampler = PixelSampler(
            num_image_per_split=self.num_image_per_split,
            total_image_num=self.total_image_num,
            camera=self.camera,
            batch_size=self.batch_size,
            image_idx_rng_seed=seed,
            pixel_idx_rng_seed=seed,
            strategy=data_config.pixel_sampling_strategy
        )
        
    def __getitem__(self, idx):
        H, W = self.camera.H, self.camera.W
        w_indices, h_indices = torch.meshgrid(
            torch.linspace(0, W - 1, W),
            torch.linspace(0, H - 1, H),
            indexing='xy')
        return RawPixelBundle(
            img_indices=torch.ones([H, W, 1], dtype=torch.long) * idx,
            h_indices=h_indices[..., None],
            w_indices=w_indices[..., None],
            rgb_gt=torch.from_numpy(self.imgs[idx]),
            poses=torch.from_numpy(self.poses[idx]).expand((H, W, 4, 4)),
            mask1s=torch.from_numpy(self.mask1s[idx]),
            mask2s=torch.from_numpy(self.mask2s[idx]),
        )
        
    def get_next_batch(self):
        img_indices, h_indices, w_indices = self.sampler.sample_batch()

        return RawPixelBundle(
            img_indices=torch.from_numpy(img_indices)[..., None],
            h_indices=torch.from_numpy(h_indices)[..., None],
            w_indices=torch.from_numpy(w_indices)[..., None],
            rgb_gt=torch.from_numpy(self.imgs[img_indices, h_indices, w_indices]),
            poses=torch.from_numpy(self.poses[img_indices]),
            mask1s=torch.from_numpy(self.mask1s[img_indices, h_indices, w_indices]),
            mask2s=torch.from_numpy(self.mask2s[img_indices, h_indices, w_indices]),
        )
        
    """Functions for render_test_views"""
    def test_view_num(self):
        """get test view nums"""
        return self.num_image_per_split[2]
    
    def get_test_view(self,idx):
        """return a test view according to index."""
        idx += self.num_image_per_split[0] + self.num_image_per_split[1]
        return self[idx]