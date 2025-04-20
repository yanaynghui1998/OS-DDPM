"""
Train a diffusion model on images.
"""

import argparse
import sys
import copy
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.bratsloader import BRATSVolumes

from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from improved_diffusion.train_onestep_util import TrainLoop
import torch

def main():
    args = create_argparser().parse_args()
    args_teacher = create_argparser_teacher().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating teacher models and diffusion...")
    teacher_model_real, _ = create_model_and_diffusion(
        **args_to_dict(args_teacher, model_and_diffusion_defaults().keys())
    )
    teacher_model_fake = copy.deepcopy(teacher_model_real)
    teacher_model_real.to(dist_util.dev())
    teacher_model_fake.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    ds = BRATSVolumes(args.data_dir, test_flag=False,
                          normalize=(lambda x: 2*x - 1),
                          mode='train')
    data=torch.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     num_workers=0,
                                     shuffle=True,
                                     )
    logger.log("training...")

    TrainLoop(
        model=model,
        teacher_model_real=teacher_model_real,
        teacher_model_fake=teacher_model_fake,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        teacher_lr=args_teacher.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        logdir=args.logdir,
    ).run_loop()

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        logdir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def create_argparser_teacher():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-6,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
