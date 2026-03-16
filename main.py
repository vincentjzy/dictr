import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np
import os

from utils.datasets import build_train_dataset
from networks.dictr import DICTr
from loss import flow_loss_func, flow_loss_func_unsupervised
from evaluate import validate_speckle, validate_speckle_unsupervised
from experiment import custom, rotation_128, rotation_256, tension, star5, mei_128, realcrack, mei_256, shear
from utils.logger import Logger
from utils import misc
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed


def get_args_parser():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--supervised', default=True, type=bool,
                        help='whether to use supervised loss, set false for unsupervised training and inference')
    parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default=['speckle'], type=str,
                        help='training stage')
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding')
    parser.add_argument('--val_dataset', default=['speckle'], type=str, nargs='+',
                        help='validation dataset')
    parser.add_argument('--v1', default=True, type=bool,
                        help='v1.0 or v2.0 of DICTr, v1.0 deals with 128x128 input and v2.0 deals with 256x256 input, this parameter is for experiment only and does not affect your own model definition')

    # training strategy
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--val_freq', default=5000, type=int)
    parser.add_argument('--save_ckpt_freq', default=5000, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)
    parser.add_argument('--gamma', default=0.9, type=float, help='loss weight for each layer')

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrain model for finetuing or resume from terminated training')
    parser.add_argument('--strict_resume', action='store_true')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # DICTr model
    parser.add_argument('--num_scales', default=2, type=int,
                        help='DICTr use 2 scale features, 1/4 for global match and 1/2 for refinement')
    parser.add_argument('--feature_channels', default=128, type=int,
                        help='DICTr use 128 channels for higher-level description of features, unsupervised DICTr use 256 channels')
    parser.add_argument('--upsample_factor', default=2, type=int,
                        help='DICTr get full resolution result by convex upsampling from 1/2 resolution')
    parser.add_argument('--num_transformer_layers', default=12, type=int,
                        help='DICTr use 12 transformer layer (6 blocks) to enhence image features')
    parser.add_argument('--num_head', default=1, type=int,
                        help='DICTr use single head attention')
    parser.add_argument('--attention_type', default='swin', type=str,
                        help='DICTr use swin transformer')
    parser.add_argument('--ffn_dim_expansion', default=4, type=int,
                        help='Dimension expansion scale in Feed-Forward Networks, follow <Attention Is All You Need>')

    # GMFlow model default setting, you can switch to a smaller window size to carry out
    # the attention mechanism when computational costs become a bottleneck
    # In DICTr, first parameter is for 1/4 scale features and second parameter is for 1/2 scale features
    # For 2D-DIC, flow propagation may be unnecessary since there is no occlusion problem
    parser.add_argument('--attn_splits_list', default=[2, 8], type=int, nargs='+',
                        help='number of splits on feature map edge to form window layout for swin transformer')
    parser.add_argument('--corr_radius_list', default=[-1, 4], type=int, nargs='+',
                        help='radius for feature matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1, 1], type=int, nargs='+',
                        help='self-attention radius for flow propagation, -1 indicates global attention')

    # inference
    parser.add_argument('--exp', action='store_true')
    parser.add_argument('--exp_type', type=str, nargs='+')

    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    return parser


def main(args):
    if not args.exp:
        if args.local_rank == 0:
            print('pytorch version:', torch.__version__)
            print(args)
            # misc.save_args(args)
            misc.check_path(args.checkpoint_dir)
            # misc.save_command(args.checkpoint_dir)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.distributed = True

        # adjust batch size for each gpu
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()

        dist_params = dict(backend='nccl')
        init_dist(args.launcher, **dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        args.gpu_ids = range(world_size)
        device = torch.device('cuda:{}'.format(args.local_rank))

        setup_for_distributed(args.local_rank == 0)

    # model
    model = DICTr(feature_channels=args.feature_channels,
                  num_scales=args.num_scales,
                  upsample_factor=args.upsample_factor,
                  num_head=args.num_head,
                  attention_type=args.attention_type,
                  ffn_dim_expansion=args.ffn_dim_expansion,
                  num_transformer_layers=args.num_transformer_layers,
                  ).to(device)

    if not args.exp:
        print('Model definition:')
        print(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        model_without_ddp = model.module
    else:
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)

            model_without_ddp = model.module
        else:
            model_without_ddp = model

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)
    # if not args.exp:
    #    save_name = '%d_parameters' % num_params
    #    open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_epoch = 0
    start_step = 0
    # resume checkpoints
    if args.resume:
        print('Load checkpoint: %s' % args.resume)

        loc = 'cuda:{}'.format(args.local_rank)
        checkpoint = torch.load(args.resume, map_location=loc)

        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

        model_without_ddp.load_state_dict(weights, strict=args.strict_resume)

        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
                args.no_resume_optimizer:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']

        print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))

    # experiment
    if args.exp:
        if args.v1:
            if 'custom' in args.exp_type:
                custom(model_without_ddp,
                    attn_splits_list=args.attn_splits_list,
                    corr_radius_list=args.corr_radius_list,
                    prop_radius_list=args.prop_radius_list)
            if 'rotation' in args.exp_type:
                rotation_128(model_without_ddp,
                        attn_splits_list=args.attn_splits_list,
                        corr_radius_list=args.corr_radius_list,
                        prop_radius_list=args.prop_radius_list)
            if 'tension' in args.exp_type:
                tension(model_without_ddp,
                        attn_splits_list=args.attn_splits_list,
                        corr_radius_list=args.corr_radius_list,
                        prop_radius_list=args.prop_radius_list)
            if 'star5' in args.exp_type:
                star5(model_without_ddp,
                    attn_splits_list=args.attn_splits_list,
                    corr_radius_list=args.corr_radius_list,
                    prop_radius_list=args.prop_radius_list)
            if 'mei' in args.exp_type:
                mei_128(model_without_ddp,
                    attn_splits_list=args.attn_splits_list,
                    corr_radius_list=args.corr_radius_list,
                    prop_radius_list=args.prop_radius_list)
            if 'realcrack' in args.exp_type:
                realcrack(model_without_ddp,
                        attn_splits_list=args.attn_splits_list,
                        corr_radius_list=args.corr_radius_list,
                        prop_radius_list=args.prop_radius_list)
        if not args.v1:
            if 'rotation' in args.exp_type:
                rotation_256(model_without_ddp,
                        attn_splits_list=args.attn_splits_list,
                        corr_radius_list=args.corr_radius_list,
                        prop_radius_list=args.prop_radius_list)
            if 'mei' in args.exp_type:
                mei_256(model_without_ddp,
                    attn_splits_list=args.attn_splits_list,
                    corr_radius_list=args.corr_radius_list,
                    prop_radius_list=args.prop_radius_list)
            if 'shear' in args.exp_type:
                shear(model_without_ddp,
                        attn_splits_list=args.attn_splits_list,
                        corr_radius_list=args.corr_radius_list,
                        prop_radius_list=args.prop_radius_list)
        return

    # training datset
    train_dataset = build_train_dataset(args)
    print('Number of training images:', len(train_dataset))

    # Multi-processing
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank)
    else:
        train_sampler = None

    shuffle = False if args.distributed else True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=shuffle, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True,
                                               sampler=train_sampler)

    last_epoch = start_step if args.resume and start_step > 0 else -1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr,
        args.num_steps + 10,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='cos',
        last_epoch=last_epoch,
    )

    if args.local_rank == 0:
        summary_writer = SummaryWriter(args.checkpoint_dir)
        logger = Logger(lr_scheduler, summary_writer, args.summary_freq,
                        start_step=start_step, supervised=args.supervised)

    total_steps = start_step
    epoch = start_epoch
    print('Start training')

    while total_steps < args.num_steps:
        model.train()

        # mannual change random seed for shuffling every epoch
        if args.distributed:
            train_sampler.set_epoch(epoch)

        for i, sample in enumerate(train_loader):
            img1, img2, flow_gt, valid = [x.to(device) for x in sample]

            results_dict = model(img1, img2,
                                 attn_splits_list=args.attn_splits_list,
                                 corr_radius_list=args.corr_radius_list,
                                 prop_radius_list=args.prop_radius_list,
                                 )

            flow_preds = results_dict['flow_preds']
            if args.supervised:
                loss, metrics = flow_loss_func(flow_preds, flow_gt, valid,
                                               gamma=args.gamma,
                                               )
            else:
                loss, metrics = flow_loss_func_unsupervised(flow_preds, img1, img2,
                                           gamma=args.gamma,
                                           )

            if isinstance(loss, float):
                continue

            if torch.isnan(loss):
                continue

            metrics.update({'total_loss': loss.item()})

            # more efficient zero_grad
            for param in model_without_ddp.parameters():
                param.grad = None

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            lr_scheduler.step()

            if args.local_rank == 0:
                logger.push(metrics)

            total_steps += 1

            if total_steps % args.save_ckpt_freq == 0 or total_steps == args.num_steps:
                if args.local_rank == 0:
                    checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)
                    torch.save({
                        'model': model_without_ddp.state_dict()
                    }, checkpoint_path)

            if total_steps % args.save_latest_ckpt_freq == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')

                if args.local_rank == 0:
                    torch.save({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': total_steps,
                        'epoch': epoch,
                    }, checkpoint_path)

            if total_steps % args.val_freq == 0:
                print('Start validation')

                val_results = {}

                if 'speckle' in args.val_dataset:
                    if args.supervised:
                        results_dict = validate_speckle(model_without_ddp,
                                                    attn_splits_list=args.attn_splits_list,
                                                    corr_radius_list=args.corr_radius_list,
                                                    prop_radius_list=args.prop_radius_list,
                                                    )
                    else:
                        results_dict = validate_speckle_unsupervised(model_without_ddp,
                                                    attn_splits_list=args.attn_splits_list,
                                                    corr_radius_list=args.corr_radius_list,
                                                    prop_radius_list=args.prop_radius_list,
                                                    )
                    if args.local_rank == 0:
                        val_results.update(results_dict)

                if args.local_rank == 0:
                    logger.write_dict(val_results)

                    # Save validation results
                    val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
                    with open(val_file, 'a') as f:
                        f.write('step: %06d\n' % total_steps)
                        if args.supervised:
                            f.write('supervised training\n')
                            metrics = ['dataset_AEE', 'dataset_AEE_s0_0.5', 'dataset_AEE_s0.5_1', 'dataset_AEE_s1+']
                        else:
                            f.write('unsupervised training\n')
                            metrics = ['Gray', 'Total']

                        eval_metrics = []
                        for metric in metrics:
                            if metric in val_results.keys():
                                eval_metrics.append(metric)

                        metrics_values = [val_results[metric] for metric in eval_metrics]

                        num_metrics = len(eval_metrics)

                        # save as Markdown format
                        f.write(("| {:>20} " * num_metrics + '\n').format(*eval_metrics))
                        f.write(("| {:20.3f} " * num_metrics).format(*metrics_values))

                        f.write('\n\n')

                model.train()

            if total_steps >= args.num_steps:
                print('Training done')

                return

        epoch += 1


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
