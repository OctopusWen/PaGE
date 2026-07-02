"""
Trains the model on both GazeFollow and Video Attention Target.
"""

import argparse
from datetime import datetime
import numpy as np
import os
import random
import math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score

from page.dataloader import GazeDataset, collate_fn
from page.model_factory import get_page_model
from page.utils import gazefollow_auc, gazefollow_l2, vat_auc, vat_l2, CosineLRWithWarmup, make_param_groups

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='page_vitb_inout')
parser.add_argument('--model_ckpt_path', type=str, default=None)
parser.add_argument('--model_ckpt_no_backbone', action="store_true")
parser.add_argument('--gf_data_path', type=str, default=['./data/gazefollow'])
parser.add_argument('--vat_data_path', type=str, default=['./data/vat'])
parser.add_argument('--cp_data_path', type=str, default=['./data/childplay'])
parser.add_argument('--vat_train_frame_sample_every', type=int, default=3)
parser.add_argument('--vat_test_frame_sample_every', type=int, default=6)
parser.add_argument('--cp_train_frame_sample_every', type=int, default=6)
parser.add_argument('--cp_test_frame_sample_every', type=int, default=6)
parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
parser.add_argument('--exp_name', type=str, default='train_all')
parser.add_argument('--log_iter', type=int, default=10, help='how often to log loss during training')
parser.add_argument('--max_epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--inout_loss_lambda', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-2)
parser.add_argument('--warmup_iters', type=float, default=100)
parser.add_argument('--warmup_start_lr', type=float, default=1e-4)
parser.add_argument('--min_lr', type=float, default=1e-7)
parser.add_argument('--n_workers', type=int, default=16)
parser.add_argument('--inout', action="store_true")
parser.add_argument('--heatmap_res', type=int, default=64)
parser.add_argument('--clip_gradient', action="store_true")
parser.add_argument('--eval_every_epochs', type=int, default=3)
parser.add_argument('--tensorboard_log_dir', type=str, default=None, help='TensorBoard log directory (defaults to <exp_dir>/tensorboard)')

args = parser.parse_args()


def main():
    exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(exp_dir, exist_ok=True)

    tb_dir = args.tensorboard_log_dir or os.path.join(exp_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)

    def tb_log(tag, value, step):
        if writer is not None:
            writer.add_scalar(tag, value, step)
            writer.flush()

    model, transforms, head_transforms = get_page_model(args.model)
    if args.model_ckpt_path is not None:
        model.load_page_state_dict(torch.load(args.model_ckpt_path), include_backbone=not args.model_ckpt_no_backbone)
    model.cuda()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_params <= 200000000:
        accumulate_steps = 1
    elif 200000000 < n_params <= 1000000000:
        accumulate_steps = 2
    else:
        accumulate_steps = 5
    if args.batch_size % accumulate_steps != 0:
        raise ValueError(f"Batch size {args.batch_size} must be divisible by the number of gradient accumulation steps {accumulate_steps}.")
    args.batch_size //= accumulate_steps
    print(f"Learnable parameters: {n_params}")

    train_dataset = GazeDataset(['gazefollow', 'videoattentiontarget', 'childplay'], [args.gf_data_path, args.vat_data_path, args.cp_data_path], 'train', transforms, head_transforms, 
                                in_frame_only=not args.inout, sample_rates=[1, args.vat_train_frame_sample_every, args.cp_train_frame_sample_every], preload_imgs=False, heatmap_res=args.heatmap_res)


    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.n_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    gf_eval_dataset = GazeDataset(['gazefollow'], [args.gf_data_path], 'test', transforms, head_transforms, in_frame_only=not args.inout)
    gf_eval_dl = torch.utils.data.DataLoader(
        gf_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.n_workers,
        pin_memory=True,
    )
    # Note this eval dataloader samples frames sparsely for efficiency - for final results, run eval_vat.py which uses sample rate 1
    vat_eval_dataset = GazeDataset(['videoattentiontarget'], [args.vat_data_path], 'test', transforms, head_transforms, in_frame_only=not args.inout, sample_rates=[args.vat_test_frame_sample_every])
    vat_eval_dl = torch.utils.data.DataLoader(
        vat_eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=args.n_workers,
        pin_memory=True,
    )
    # Note this eval dataloader samples frames sparsely for efficiency - for final results, run eval_cp.py which uses sample rate 1
    cp_eval_dataset = GazeDataset(['childplay'], [args.cp_data_path], 'test', transforms, head_transforms, in_frame_only=not args.inout, sample_rates=[args.cp_test_frame_sample_every])
    cp_eval_dl = torch.utils.data.DataLoader(
        cp_eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=args.n_workers,
        pin_memory=True,
    )


    heatmap_loss_fn = nn.BCEWithLogitsLoss()
    inout_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(make_param_groups(model, weight_decay=args.weight_decay), lr=args.lr, fused=True)
    scheduler = CosineLRWithWarmup(optimizer, warmup_iters=args.warmup_iters, warmup_start_lr=args.warmup_start_lr, base_lr=args.lr, max_epochs=args.max_epochs, min_lr=args.min_lr)

    best_min_l2 = 1.0
    best_epoch = None
    global_train_step = 0

    for epoch in range(args.max_epochs):
        # TRAIN EPOCH
        model.train()
        local_train_step = 0
        accumulated_heatmap_loss = 0.0
        accumulated_inout_loss = 0.0
        accumulated_total_loss = 0.0
        for cur_iter, batch in enumerate(train_dl):
            imgs, head_imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps, data_source, aspect_ratios, head_aspect_ratios = batch

            imgs_cuda = [x.cuda(non_blocking=True) for x in imgs]
            head_imgs_cuda = [x.cuda(non_blocking=True) for x in head_imgs] if head_imgs is not None else None
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model.get_logits({"images": imgs_cuda, "head_images": head_imgs_cuda, "bboxes": [[bbox] for bbox in bboxes], "aspect_ratios": aspect_ratios.cuda(), "head_aspect_ratios": head_aspect_ratios.cuda()})
                heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            
                # compute heatmap loss only for in-frame gaze targets (i.e. inout=true)
                loss = heatmap_loss_fn(heatmap_preds[inout.bool()], heatmaps[inout.bool()].cuda())
                accumulated_heatmap_loss += loss.item()
                # compute inout loss for all samples
                if args.inout:
                    inout_preds = torch.stack(preds['inout']).squeeze(dim=1)
                    inout_loss = inout_loss_fn(inout_preds, inout.float().cuda())  
                    accumulated_inout_loss += inout_loss.item()
                    loss = loss + args.inout_loss_lambda * inout_loss
                
            
            accumulated_total_loss += loss.item()
            loss = loss / accumulate_steps
            loss.backward()

            if cur_iter % accumulate_steps == accumulate_steps - 1 or cur_iter == len(train_dl) - 1:
                if args.clip_gradient:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                n_accumulated_iters = cur_iter % accumulate_steps + 1
                tb_log("train/heatmap_loss", accumulated_heatmap_loss / n_accumulated_iters, global_train_step)
                if args.inout:
                    tb_log("train/inout_loss", accumulated_inout_loss / n_accumulated_iters, global_train_step)
                tb_log("train/loss", accumulated_total_loss / n_accumulated_iters, global_train_step)
                tb_log("train/lr", scheduler.get_lr()[0], global_train_step)

                if local_train_step % args.log_iter == 0:
                    n_steps = len(train_dl)
                    if accumulate_steps > 1:
                        n_steps = math.ceil(n_steps / accumulate_steps)
                    print("TRAIN EPOCH {}, iter {}/{}, loss={}".format(
                        epoch, local_train_step, n_steps, round(accumulated_total_loss / n_accumulated_iters, 4)
                    ))
                
                global_train_step += 1
                local_train_step += 1
                accumulated_heatmap_loss = 0.0
                accumulated_inout_loss = 0.0
                accumulated_total_loss = 0.0
                scheduler.step_batch()

        scheduler.step_epoch()

        ckpt_path = os.path.join(exp_dir, 'epoch_{}.pt'.format(epoch))
        torch.save(model.get_page_state_dict(), ckpt_path)
        print("Saved checkpoint to {}".format(ckpt_path))

        # EVAL EPOCH
        if (epoch + 1) % args.eval_every_epochs == 0 or (epoch + 1) == args.max_epochs:
            print("Running evaluation")
            model.eval()
            # Eval GazeFollow
            avg_l2s = []
            min_l2s = []
            aucs = []
            for cur_iter, batch in enumerate(gf_eval_dl):
                imgs, head_imgs, bboxes, gazex, gazey, inout, heights, widths, _, aspect_ratios, head_aspect_ratios = batch
                imgs_cuda = [x.cuda(non_blocking=True) for x in imgs] 
                head_imgs_cuda = [x.cuda(non_blocking=True) for x in head_imgs] if head_imgs is not None else None
                with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds = model({"images": imgs_cuda, "head_images": head_imgs_cuda, "bboxes": [[bbox] for bbox in bboxes], "aspect_ratios": aspect_ratios.cuda(), "head_aspect_ratios": head_aspect_ratios.cuda()})

                heatmap_preds = torch.stack(preds["heatmap"]).squeeze(1).float().cpu()
                for i in range(heatmap_preds.shape[0]):
                    auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
                    avg_l2, min_l2 = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
                    aucs.append(auc)
                    avg_l2s.append(avg_l2)
                    min_l2s.append(min_l2)

            epoch_avg_l2 = np.mean(avg_l2s)
            epoch_min_l2 = np.mean(min_l2s)
            epoch_auc = np.mean(aucs)

            tb_log("eval/GazeFollow AUC", epoch_auc, epoch)
            tb_log("eval/GazeFollow Avg L2", epoch_avg_l2, epoch)
            tb_log("eval/GazeFollow Min L2", epoch_min_l2, epoch)
            print("EVAL EPOCH {} (GazeFollow): AUC={}, Min L2={}, Avg L2={}".format(epoch, round(epoch_auc, 4), round(epoch_min_l2, 4), round(epoch_avg_l2, 4)))

            if epoch_min_l2 < best_min_l2:  # we use GazeFollow metrics to select the best model for the moment-
                best_min_l2 = epoch_min_l2
                best_epoch = epoch

            # Eval VAT
            l2s = []
            aucs = []
            all_inout_preds = []
            all_inout_gts = []
            for cur_iter, batch in enumerate(vat_eval_dl):
                imgs, head_imgs, bboxes, gazex, gazey, inout, heights, widths, _, aspect_ratios, head_aspect_ratios = batch
                imgs_cuda = [x.cuda(non_blocking=True) for x in imgs]
                head_imgs_cuda = [x.cuda(non_blocking=True) for x in head_imgs] if head_imgs is not None else None

                with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds = model({"images": imgs_cuda, "head_images": head_imgs_cuda, "bboxes": [[bbox] for bbox in bboxes], "aspect_ratios": aspect_ratios.cuda(), "head_aspect_ratios": head_aspect_ratios.cuda()})

                heatmap_preds = torch.stack(preds["heatmap"]).squeeze(1).float().cpu()
                if args.inout:
                    inout_preds = torch.stack(preds['inout']).squeeze(dim=1)
                for i in range(heatmap_preds.shape[0]):
                    if inout[i] == 1: # in-frame
                        auc = vat_auc(heatmap_preds[i], gazex[i][0], gazey[i][0], res=args.heatmap_res)
                        l2 = vat_l2(heatmap_preds[i], gazex[i][0], gazey[i][0], res=args.heatmap_res)
                        aucs.append(auc)
                        l2s.append(l2)
                    if args.inout:
                        all_inout_preds.append(inout_preds[i].item())
                        all_inout_gts.append(inout[i])

            epoch_l2 = np.mean(l2s)
            epoch_auc = np.mean(aucs)
            if args.inout:
                epoch_inout_ap = average_precision_score(all_inout_gts, all_inout_preds)
            else:
                epoch_inout_ap = -1.0

            tb_log("eval/VAT AUC", epoch_auc, epoch)
            tb_log("eval/VAT InOut AP", epoch_inout_ap, epoch)
            tb_log("eval/VAT L2", epoch_l2, epoch)
            print("EVAL EPOCH {} (VAT): AUC={}, L2={}, Inout AP={}".format(epoch, round(epoch_auc, 4), round(epoch_l2, 4), round(epoch_inout_ap, 4)))


            # Eval ChildPlay
            l2s = []
            aucs = []
            all_inout_preds = []
            all_inout_gts = []
            for cur_iter, batch in enumerate(cp_eval_dl):
                imgs, head_imgs, bboxes, gazex, gazey, inout, heights, widths, _, aspect_ratios, head_aspect_ratios = batch
                imgs_cuda = [x.cuda(non_blocking=True) for x in imgs]
                head_imgs_cuda = [x.cuda(non_blocking=True) for x in head_imgs] if head_imgs is not None else None

                with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds = model({"images": imgs_cuda, "head_images": head_imgs_cuda, "bboxes": [[bbox] for bbox in bboxes], "aspect_ratios": aspect_ratios.cuda(), "head_aspect_ratios": head_aspect_ratios.cuda()})

                heatmap_preds = torch.stack(preds["heatmap"]).squeeze(1).float().cpu()
                if args.inout:
                    inout_preds = torch.stack(preds['inout']).squeeze(dim=1)
                for i in range(heatmap_preds.shape[0]):
                    if inout[i] == 1: # in-frame
                        auc = vat_auc(heatmap_preds[i], gazex[i][0], gazey[i][0], res=args.heatmap_res)
                        l2 = vat_l2(heatmap_preds[i], gazex[i][0], gazey[i][0], res=args.heatmap_res)  # ChildPlay uses the same L2 metric as VAT
                        aucs.append(auc)
                        l2s.append(l2)
                    if args.inout:
                        all_inout_preds.append(inout_preds[i].item())
                        all_inout_gts.append(inout[i])

            epoch_l2 = np.mean(l2s)
            epoch_auc = np.mean(aucs)
            if args.inout:
                epoch_inout_ap = average_precision_score(all_inout_gts, all_inout_preds)
            else:
                epoch_inout_ap = -1.0

            tb_log("eval/ChildPlay AUC", epoch_auc, epoch)
            tb_log("eval/ChildPlay InOut AP", epoch_inout_ap, epoch)
            tb_log("eval/ChildPlay L2", epoch_l2, epoch)
            print("EVAL EPOCH {} (ChildPlay): AUC={}, L2={}, Inout AP={}".format(epoch, round(epoch_auc, 4), round(epoch_l2, 4), round(epoch_inout_ap, 4)))
    
    # Log final result and clean up the loggers
    print("Completed training. Best GazeFollow Min L2 of {} obtained at epoch {}".format(round(best_min_l2, 4), best_epoch))

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # Enable TF32 for matmul (e.g. Linear, Transformer, ...)
    torch.set_float32_matmul_precision("high")  # Set this to "high" or "medium". By default it's set to 'highest" by PyTorch so that TF32 is disabled
    # Ensure cuDNN Conv TF32 
    torch.backends.cudnn.allow_tf32 = True
    main()
