"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys
import numpy as np
import torch as th
import torch.distributed as dist
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.bratsloader_volume import BRATSVolumes 
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
import pathlib
import nibabel as nib
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    ds = BRATSVolumes(args.data_dir, test_flag=True,
                          normalize=(lambda x: 2*x - 1),
                          mode='test')
    
    datal = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     )
    data_count=0

    average_psnr, average_ssim, average_time, total_num = 0.0, 0.0, 0.0, 0.0
    batch_psnr, batch_ssim,batch_time = [], [], []

    logger.log("sampling...")
    for data in datal:
        HR_image=data[0].to(dist_util.dev())
        LR_image=data[1].to(dist_util.dev())
        _,_,D,H,W=HR_image.shape
        num=16
        volume=th.zeros_like(HR_image)
        step=D//num
        start_time = time.time()
        for i in range (num):
            model_kwargs = {}
            sample_fn = diffusion.p_sample_one_step 

            start_idx = i * step
            end_idx = (i + 1) * step if i < num - 1 else D 
            sub_volume=LR_image[:,:,start_idx:end_idx, :, :]

            sample = sample_fn(
                model,
                sub_volume.shape,
                sub_volume,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            sample=th.clamp(sample,0,1)
            volume[:,:,start_idx:end_idx, :, :] = sample

        infer_time = time.time() - start_time
        batch_time.append(infer_time)

        B, _, D, H, W = sample.size()
        infer_time = time.time() - start_time

        if len(sample.shape) == 5:
            volume = volume.squeeze(dim=1)  
            LR_image=LR_image.squeeze(dim=1)
            HR_image=HR_image.squeeze(dim=1)

        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        for i in range(sample.shape[0]):
            volume=volume.detach().cpu().numpy()[i, :, :, :]
            LR_image=LR_image.detach().cpu().numpy()[i, :, :, :]
            HR_image=HR_image.detach().cpu().numpy()[i, :, :, :]

            psnr = peak_signal_noise_ratio(HR_image, volume, data_range=HR_image.max())
            ssim=structural_similarity(HR_image, volume, data_range=HR_image.max())

            batch_psnr.append(psnr)
            batch_ssim.append(ssim)

            output_name = os.path.join(args.output_dir, f'sample_{data_count}_{i}.nii.gz')
            img = nib.Nifti1Image(volume, np.eye(4))
            nib.save(img=img, filename=output_name)

            output_name_LR = os.path.join(args.output_dir, f'LR_sample_{data_count}_{i}.nii.gz')
            img_LR = nib.Nifti1Image(LR_image, np.eye(4))
            nib.save(img=img_LR, filename=output_name_LR)

            output_name_HR = os.path.join(args.output_dir, f'HR_sample_{data_count}_{i}.nii.gz')
            img_HR = nib.Nifti1Image(HR_image, np.eye(4))
            nib.save(img=img_HR, filename=output_name_HR)
            
            print(f'Saved to {output_name}')
        data_count+=1
        print(f"Currently loading data: {data_count}")

    average_psnr=np.mean(batch_psnr)
    std_psnr=np.std(batch_psnr,ddof=1)
    average_ssim=np.mean(batch_ssim)
    std_ssim=np.std(batch_ssim,ddof=1)
    average_time=np.mean(batch_time)
    print('average_time:{:.3f}s\ttest_psnr:{:.3f}\ttest_ssim:{:.4f}'.format(
        average_time, average_psnr, average_ssim))
    
    print('std_psnr:{:.3f}\tstd_ssim:{:.3f}'.format(
            std_psnr, std_ssim))
    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=1,
        use_ddim=False,
        model_path="",
        data_dir="",
        output_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
