import random
import numpy as np
import torch
import pathlib
import wandb
import tyro
import pickle
from torchinfo import summary
from tqdm import tqdm
import os
import imageio

from configs.main_config import SystemConfig
from dataclasses import asdict
from data.data_manager import DataManager
from model.main_pipeline import MainPipeline


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
class Trainer:
    def __init__(self, config:SystemConfig):
        """
        init object, load data
        :param config: SystemConfig passed from launcher
        """
        self.config = config
        alpha = config.model.lr_alpha
        warm_up_end = config.model.warm_up_end
        end_iter = config.model.end_iter        
        
        
        # 0. setup cuda and seed everything
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{config.cuda_number}')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.backends.cudnn.benchmark = True  # type: ignore
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
            
        seed_everything(config.seed)
         
        # 1. build datamanager
        self.data_manager = DataManager(
            batch_size=config.model.batch_size,
            seed=config.seed,
            data_config=config.data
        )
        
        # 2. init logger setting
        self.init_logger()
        
        # 3. init Pipeline and model 
        self.pipeline = MainPipeline(
            config=config , 
            data_manager=self.data_manager
        ).to(self.device)
        
        summary(self.pipeline)
        
        # 4. init optimizer
        self.optimizer = torch.optim.Adam(self.pipeline.get_param_groups())       
        
        def lr_lambda(iter_step):
            if iter_step < warm_up_end:
                learning_factor = iter_step / warm_up_end
            else:
                progress = (iter_step - warm_up_end) / (end_iter - warm_up_end)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            return learning_factor

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)       
        # 5. load ckpt
        self.global_step = 0
        self.load_ckpt()
    
    
    """ basic functions"""
    def init_logger(self):
        """Function to init wandb"""
        self.log_dir = pathlib.Path(self.config.base_dir) / self.config.exp_name / self.config.scene_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / 'config.yaml', 'w') as f:
            f.write(tyro.to_yaml(self.config))
        wandb.init(
            project="NeRMF_2",
            name=f'{self.config.exp_name}_{self.config.scene_name}',
            group=self.config.exp_name,
            notes="NeRMF_2 for eccv",
            config=asdict(self.config),
            resume="allow",
            id=str(self.log_dir).replace("/", "_")
        )
        wandb.run.log_code()  # log all files in the current directory      
    
    def load_ckpt(self):
        """Load checkpoint"""

        # compose ckpt paths
        ckpt_path = self.log_dir / "ckpt"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        rng_state_path = self.log_dir / "rng_state"
        rng_state_path.mkdir(parents=True, exist_ok=True)

        if self.config.ckpt_path is not None:
            ckpts = [self.config.ckpt_path]
            print(f'Loading given ckpt from {self.config.ckpt_path}')
        else:
            ckpts = [os.path.join(ckpt_path, f) for f in sorted(os.listdir(ckpt_path)) if 'ckpt' in f]
            print(f'Found {len(ckpts)} ckpts in {ckpt_path}')
        if len(ckpts) > 0:
            try:
                ckpt_path = ckpts[-1]
                print(f'Resume from ckpt: {ckpt_path}')
                self._load_ckpt_file(ckpt_path)
            except EOFError:  # in case last ckpt is corrupted
                ckpt_path = ckpts[-2]
                print(f'Retrying resume from ckpt: {ckpt_path}')
                self._load_ckpt_file(ckpt_path)

            try:
                rng_state_path = rng_state_path / f"step_{self.global_step:07d}.pickle"
                self._load_rng_states(rng_state_path)
            except Exception as e:
                print(e)
                print("rng state resume failed, the results might not be fully reproducible")

    def _load_ckpt_file(self, ckpt_file):
        """Load checkpoint from specific file"""

        ckpt = torch.load(ckpt_file, map_location=self.device)
        self.global_step = ckpt["global_step"]
        self.pipeline.load_state_dict(ckpt["pipeline"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["scheduler"])

    def _load_rng_states(self, rng_state_path):
        """Load rng states from specific file"""

        rng_states = pickle.load(open(rng_state_path, "rb"))
        random.setstate(rng_states["python.random"])
        np.random.set_state(rng_states["np.random"])
        torch.random.set_rng_state(rng_states["torch.random"])
        torch.cuda.set_rng_state(rng_states["torch.cuda.random"])
        self.data_manager.sampler.image_rng.__setstate__(rng_states["ray_generator.image"])
        self.data_manager.sampler.pixel_rng.__setstate__(rng_states["ray_generator.pixel"])
        
    def save_ckpt(self):
        """Save checkpoint"""

        ckpt_path = self.log_dir / "ckpt" / f"step_{self.global_step:07d}.ckpt"
        
        model_states = {
            "global_step": self.global_step,
            "pipeline": self.pipeline.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict()
        }

        torch.save(model_states, ckpt_path)
        # also save rng states
        
        rng_states = {
            "python.random": random.getstate(),
            "np.random": np.random.get_state(),
            "torch.random": torch.random.get_rng_state(),
            "torch.cuda.random": torch.cuda.random.get_rng_state(self.device),
            "ray_generator.image": self.data_manager.sampler.image_rng.__getstate__(),
            "ray_generator.pixel": self.data_manager.sampler.pixel_rng.__getstate__()
        }
        
        rng_state_path = self.log_dir / "rng_state" / f"step_{self.global_step:07d}.pickle"
        pickle.dump(rng_states, open(rng_state_path, 'wb'))
    
    """ functions use for training"""
    def run(self):
        """Core entry point for training"""

        # training
        if not self.config.evaluation_only:
            start_step = self.global_step
            for _ in tqdm(
                    range(start_step, self.config.model.end_iter),
                    desc=f"Training: ",
                    initial=start_step,
                    total=self.config.model.end_iter,
                    dynamic_ncols=True,
            ):
                loss_dict = self.train_iter()
                if self.global_step % self.config.intervals.log_metrics == 0:
                    wandb.log(loss_dict, step=self.global_step)
                self.global_step += 1
                if self.global_step % self.config.intervals.save_ckpt == 0:
                    self.save_ckpt()
                #if self.global_step % self.config.intervals.render_test_views == 0:
                    #self.render_test_views()
                #if self.global_step % self.config.intervals.dump_mesh == 0:
                    #self.dump_mesh()
                #if self.global_step % self.config.intervals.render_video == 0:
                    #self.render_video()

        # final dumps / evaluation
        #self.dump_mesh(resolution=1024)
        self.render_test_views(is_final=True)
        
    def train_iter(self):
        """One training iteration"""

        pixel_bundle = self.data_manager.get_next_batch()
        pixel_bundle = pixel_bundle.to(self.device)
        rendering_out = self.pipeline.forward(pixel_bundle, global_step=self.global_step)  # use the warped model
        loss_dict = self.pipeline.get_train_loss_dict(rendering_out, pixel_bundle)
        loss = loss_dict['loss']

        self.optimizer.zero_grad()
        loss.backward()
        
        
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss_dict
    
    """test functions"""
    def render_test_views(self, is_final=False):
        """Eval on selected images, save all into disk, and log one of them onto wandb"""

        total_test_view = self.data_manager.test_view_num()
        skip_num = self.config.data.testset_skip if not is_final else 1
        start_idx = skip_num

        metrics_dicts = []
        
        for idx in tqdm(range(start_idx, total_test_view),
                        desc=f"Rendering test views on process : "):
            metrics_dict = self.render_single_view(idx)
            metrics_dicts.append(metrics_dict)


        # gather metrics
        output_list = metrics_dicts
        

        # calculate mean metrics & log onto wandb
        gather_output = {}
        image_cnt = 0
        for item in output_list:  # type: ignore
            image_cnt += 1
            for k, v in item.items():
                gather_output.setdefault(k, 0.)
                gather_output[k] += v
        final_output = {}
        for k, v in gather_output.items():
            final_output[f'val/{k}' if is_final else f'val/{k}'] = v / image_cnt
        wandb.log(final_output, step=self.global_step)
            
    @torch.no_grad()
    def render_single_view(self, view_index, is_training_view: bool = False):
        """Render single view"""

        img_pixel_bundle = self.data_manager.get_test_view(view_index)
        # only eval on single gpu
        img_dict, metrics_dict, tensor_dict = self.pipeline.get_eval_dicts(img_pixel_bundle, self.device)
        self.save_dumps(view_index, img_dict, tensor_dict)
        if view_index == 0:
            self.log_images(img_dict)
        return metrics_dict
    
    def save_dumps(self, view_idx, image_dict, tensor_dict):
        """Save dumped images and tensors to disk"""

        dump_dir = self.log_dir / "test_views" / f"step_{self.global_step:07d}"
        dump_dir.mkdir(parents=True, exist_ok=True)

        # dump images
        for k, v in image_dict.items():
            if "normal" in k:
                v = v * 0.5 + 0.5  # scale normal maps from [-1, 1] to [0, 1]
            if v.shape[-1] == 1:
                v = v[..., 0]
            imageio.v3.imwrite(dump_dir / f"{k}_{view_idx:03d}.png", (v * 255).clip(0, 255).astype(np.uint8))

        # dump tensors into npy
        for k, v in tensor_dict.items():
            np.save(dump_dir / f"{k}_{view_idx:03d}.npy", v)  # type: ignore
        
    def log_images(self, image_dict):
        """Log images to wandb"""

        for k, v in image_dict.items():
            if "normal" in k:
                v = v * 0.5 + 0.5  # scale normal maps from [-1, 1] to [0, 1]
            wandb.log({k: wandb.Image((v * 255).clip(0, 255).astype(np.uint8))}, step=self.global_step)