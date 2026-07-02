import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


from page.dataloader import GazeDataset, collate_fn
from page.model_factory import get_page_model
from page.utils import vat_auc, vat_l2
from sklearn.metrics import average_precision_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--backbone_in_ckpt", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument('--heatmap_res', type=int, default=64)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] device={device}")

    # Build model + transform exactly like training
    model, transforms, head_transforms = get_page_model(args.model_name)

    # Load checkpoint
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_page_state_dict(ckpt, include_backbone=args.backbone_in_ckpt)

    model.to(device)
    model.eval()

    # Use SAME dataset class as training script
    cp_eval_dataset = GazeDataset(['childplay'], [args.data_path], 'test', transforms, head_transforms, in_frame_only=False, sample_rates=[1])
    cp_eval_dl = torch.utils.data.DataLoader(
        cp_eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=args.n_workers
    )

    l2s = []
    aucs = []
    all_inout_preds = []
    all_inout_gts = []

    for batch in tqdm(cp_eval_dl, "Evaluating ChildPlay (train-style)", len(cp_eval_dl)):
        imgs, head_imgs, bboxes, gazex, gazey, inout, heights, widths, _, aspect_ratios, head_aspect_ratios = batch
        imgs_cuda = [x.cuda(non_blocking=True) for x in imgs]
        head_imgs_cuda = [x.cuda(non_blocking=True) for x in head_imgs] if head_imgs is not None else None
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model({"images": imgs_cuda, "head_images": head_imgs_cuda, "bboxes": [[bbox] for bbox in bboxes], "aspect_ratios": aspect_ratios.cuda(), "head_aspect_ratios": head_aspect_ratios.cuda()})
        heatmap_preds = torch.stack(preds["heatmap"]).squeeze(dim=1).float().cpu()  # [B, H, W]
        inout_preds = torch.stack(preds['inout']).squeeze(dim=1)

        for i in range(heatmap_preds.shape[0]):
            if inout[i] == 1: # in-frame
                auc = vat_auc(heatmap_preds[i], gazex[i][0], gazey[i][0], res=args.heatmap_res)
                l2 = vat_l2(heatmap_preds[i], gazex[i][0], gazey[i][0], res=args.heatmap_res)
                aucs.append(auc)
                l2s.append(l2)
            all_inout_preds.append(inout_preds[i].item())
            all_inout_gts.append(inout[i])

    l2 = np.mean(l2s)
    l2_percentiles = np.percentile(l2s, [80, 90, 95, 99])
    auc = np.mean(aucs)
    inout_ap = average_precision_score(all_inout_gts, all_inout_preds)

    print("AUC: ", auc)
    print("InOut AP: ", inout_ap)
    print("L2: ", l2)
    print("Min L2 (80%): ", l2_percentiles[0])
    print("Min L2 (90%): ", l2_percentiles[1])
    print("Min L2 (95%): ", l2_percentiles[2])
    print("Min L2 (99%): ", l2_percentiles[3])

    # visualize histogram of min l2
    bins = 200 
    hist_range = (0.0, 1.0)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Histogram (log y)
    os.makedirs('./visualization/', exist_ok=True)

    counts, edges, _ = ax.hist(
        l2s, bins=bins, range=hist_range,
        edgecolor="black", alpha=0.8, log=True
    )
    ax.set_xlim(*hist_range)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count (log scale)")

    # Cumulative % curve (CDF across bins)
    cum_pct = np.cumsum(counts) / counts.sum() * 100.0
    x_centers = 0.5 * (edges[:-1] + edges[1:])

    ax2 = ax.twinx()
    ax2.plot(x_centers, cum_pct, linewidth=2)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Cumulative samples covered (%)")

    ax.set_title("Histogram (log y) + cumulative % covered")
    plt.tight_layout()
    plt.savefig('./visualization/hist_childplay.png', dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()