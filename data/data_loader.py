import pathlib
from typing import Optional, Union

import numpy as np
import imageio
import json
import cv2
from tqdm import tqdm

from camera.camera_model import CameraModel


def read_data(
        basedir: Union[pathlib.Path, str],
        splits: Optional[list] = None,
        half_res: bool = False,
        white_background: bool = True
):
    """
    Parse and load datasets data into shared memory.
    :param basedir: dataset data directory.
    :param splits: list of splits to load. Default to ['train', 'val', 'test']
    :param half_res: whether to load half resolution images.
    :param white_background: whether to load images with white background.
    :return: NRDataSHMArrayWriter object
    """
    if splits is None:
        splits = ['train', 'val', 'test']
    if type(basedir) == str:
        basedir = pathlib.Path(basedir)

    # load metadata
    metas = {}
    for s in splits:
        with open(basedir / 'transforms_{}.json'.format(s), 'r') as fp:
            metas[s] = json.load(fp)
    num_image_per_split = [len(metas[s]['frames']) for s in splits]
    total_image_num = sum(num_image_per_split)

    zn = 3.
    zf = 10.
    meta = metas[splits[0]]
    if 'camera_near' in meta:
        zn = meta['camera_near']
    if 'camera_far' in meta:
        zf = meta['camera_far']

    # load first image to get image size
    first_image = imageio.v3.imread(basedir / (metas[splits[0]]['frames'][0]['file_path'] + '.png'))
    H, W = first_image.shape[:2]

    # load camera intrinsic, single camera model
    if 'camera_intrinsics' in meta:
        intrinsics = meta['camera_intrinsics']
        cx = intrinsics[0]
        cy = intrinsics[1]
        fx = intrinsics[2]
        fy = intrinsics[3]
    else:  # fall back to camera_angle_x
        camera_angle_x = float(meta['camera_angle_x'])
        focal = float(.5 * W / np.tan(.5 * camera_angle_x))
        cx = W / 2.
        cy = H / 2.
        fx = focal
        fy = focal

    if half_res:  # half intrinsics
        H = H // 2
        W = W // 2
        cx = cy / 2.
        cy = cy / 2.
        fx = fx / 2.
        fy = fy / 2.
    
    # build CameraModel
    camera=CameraModel(H, W, cx, cy, fx, fy, zn, zf)
    
    # build data array
    imgs = np.zeros((total_image_num, H, W, 3),dtype=np.float32)
    poses = np.zeros((total_image_num, 4, 4),dtype=np.float32)
    mask1s = np.zeros((total_image_num, H, W, 3),dtype=np.float32)
    mask2s = np.zeros((total_image_num, H, W, 3),dtype=np.float32)
    

    global_index = 0
    for s in splits:
        meta = metas[s]
        for frame in tqdm(meta['frames'], desc=f'Loading {s} data'):
            frame_image_ext = frame.get('file_ext', '.png')
            filename = basedir / (frame['file_path'] + frame_image_ext)

            
            mask_path = frame['file_path'].split('_')
            mask1name =   basedir/ "masks" /("RefractMask_" + mask_path[1] + frame_image_ext)
            mask2name =   basedir/ "masks" /("MirrorMask_" + mask_path[1] + frame_image_ext)
            
            # read images
            if frame_image_ext == '.npy':
                img = np.load(filename)
            elif frame_image_ext == '.exr':
                img = imageio.v3.imread(filename)
            else:
                img = imageio.v3.imread(filename) / 255.
                mask1 = imageio.v3.imread(mask1name) / 255.
                mask2 = imageio.v3.imread(mask2name) / 255.
                
            # other handle
            if half_res:
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA)
            if white_background:
                img = img[..., :3] * img[..., 3:] + (1. - img[..., 3:])
            else:
                img = img[..., :3]
                mask1 = mask1[..., :3]
                mask2 = mask2[..., :3]

            mask1s[global_index] = np.array(mask1, dtype=np.float32)
            mask2s[global_index] = np.array(mask2, dtype=np.float32)
            imgs[global_index] = img.astype(np.float32)
            poses[global_index] = np.array(frame['transform_matrix']).astype(np.float32)

            global_index += 1

    return {
        'mask1s':mask1s,
        'mask2s':mask2s,
        'imgs':imgs,
        'poses':poses,
        'camera':camera,
        'num_image_per_split':num_image_per_split,
        'total_image_num':total_image_num
    }