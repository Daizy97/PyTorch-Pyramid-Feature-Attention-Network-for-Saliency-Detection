from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import argparse
import cv2
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import SODModel
from src.dataloader import InfDataloader, SODLoader


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters to train your model.')
    parser.add_argument('--imgs_folder', default='/home/Dataset/ImageNet/val', help='Path to folder containing images', type=str)
    parser.add_argument('--model_path', default='best-model_epoch-204_mae-0.0505_loss-0.1370.pth', help='Path to model', type=str)
    parser.add_argument('--use_gpu', default=True, help='Whether to use GPU or not', type=bool)
    parser.add_argument('--img_size', default=256, help='Image size to be used', type=int)
    parser.add_argument('--bs', default=24, help='Batch Size for testing', type=int)

    # Save saliency map
    parser.add_argument('--asset_dir', default='./assets', type=str)
    parser.add_argument('--save_dir', default='./saves', type=str)
    parser.add_argument('--save_img', dest='save_img', action='store_true')
    parser.add_argument('--targeted', action='store_true', help='Targeted attack if true')
    parser.add_argument('--img_index_start', default=0, type=int)
    parser.add_argument('--sample_size', default=1000, type=int)
    parser.add_argument('--sal_threshold', default=0.1, type=float)

    return parser.parse_args()
def get_path(index, imagenet_path=None):
    image_paths = sorted([os.path.join(imagenet_path, i) for i in os.listdir(imagenet_path)])
    assert len(image_paths) == 50000
    path = image_paths[index]
    return path

def run_inference(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    # Create directory
    if args.save_img:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    # Load the indices
    if args.targeted:
        indices = np.load(os.path.join(args.asset_dir, 'indices_targeted.npy'))
    else:
        indices = np.load(os.path.join(args.asset_dir, 'indices_untargeted.npy'))
    count = 0
    index = args.img_index_start

    while count < args.sample_size:
        path = get_path(indices[index], args.imgs_folder)
        inf_data = InfDataloader(img_path=path, target_size=args.img_size)
        # Since the images would be displayed to the user, the batch_size is set to 1
        # Code at later point is also written assuming batch_size = 1, so do not change
        inf_dataloader = DataLoader(inf_data, batch_size=1, shuffle=False, num_workers=2)

        with torch.no_grad():
            for batch_idx, (img_np, img_tor) in enumerate(inf_dataloader, start=1):
                img_tor = img_tor.to(device)
                pred_masks, _ = model(img_tor)

                # Assuming batch_size = 1
                pred_masks_raw = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
                pred_masks_round_threshold = np.where(pred_masks_raw > args.sal_threshold, 1.0, 0.0)
                if args.save_img:
                    st = "%05d" % indices[index]  # fill 0 to the index
                    cv2.imwrite(os.path.join(args.save_dir, '{}_T={} Saliency Mask.png'.format(st, args.sal_threshold)),
                                pred_masks_round_threshold * 255)
        print("Image:", count)
        count += 1
        index += 1
    print("Finished")

def calculate_mae(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    test_data = SODLoader(mode='test', augment_data=False, target_size=args.img_size)
    test_dataloader = DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=2)

    # List to save mean absolute error of each image
    mae_list = []
    with torch.no_grad():
        for batch_idx, (inp_imgs, gt_masks) in enumerate(tqdm.tqdm(test_dataloader), start=1):
            inp_imgs = inp_imgs.to(device)
            gt_masks = gt_masks.to(device)
            pred_masks, _ = model(inp_imgs)

            mae = torch.mean(torch.abs(pred_masks - gt_masks), dim=(1, 2, 3)).cpu().numpy()
            mae_list.extend(mae)

    print('MAE for the test set is :', np.mean(mae_list))


if __name__ == '__main__':
    rt_args = parse_arguments()
    # calculate_mae(rt_args)
    run_inference(rt_args)

