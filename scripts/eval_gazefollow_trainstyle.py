import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


from page.dataloader import GazeDataset, collate_fn
from page.model_factory import get_page_model
from page.utils import gazefollow_auc, gazefollow_l2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--backbone_in_ckpt", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_workers", type=int, default=8)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] device={device}")

    # Build model + transform exactly like training
    model, transform, head_transforms = get_page_model(args.model_name)

    # Load checkpoint
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_page_state_dict(ckpt, include_backbone=args.backbone_in_ckpt)

    model.to(device)
    model.eval()

    # Use SAME dataset class as training script
    dataset = GazeDataset(["gazefollow"], [args.data_path], "test", transform, head_transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.n_workers,
        pin_memory=True,
    )

    aucs, avg_l2s, min_l2s = [], [], []

    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating GazeFollow (train-style)"):
        imgs, head_imgs, bboxes, gazex, gazey, _, heights, widths, _, aspect_ratios, head_aspect_ratios = batch
        imgs_cuda = [x.cuda(non_blocking=True) for x in imgs]
        head_imgs_cuda = [x.cuda(non_blocking=True) for x in head_imgs] if head_imgs is not None else None
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model({"images": imgs_cuda, "head_images": head_imgs_cuda, "bboxes": [[bbox] for bbox in bboxes], "aspect_ratios": aspect_ratios.cuda(), "head_aspect_ratios": head_aspect_ratios.cuda()})
        heatmap_preds = torch.stack(preds["heatmap"]).squeeze(dim=1).float().cpu()  # [B, H, W]

        for i in range(heatmap_preds.shape[0]):
            auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
            avg_l2, min_l2 = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
            aucs.append(auc)
            avg_l2s.append(avg_l2)
            min_l2s.append(min_l2)

    epoch_auc = float(np.mean(aucs)) if len(aucs) else float("nan")
    epoch_avg_l2 = float(np.mean(avg_l2s)) if len(avg_l2s) else float("nan")
    epoch_min_l2 = float(np.mean(min_l2s)) if len(min_l2s) else float("nan")
    epoch_min_l2_percentiles = np.percentile(min_l2s, [80, 90, 95, 99])

    print("AUC: ", epoch_auc)
    print("Avg L2: ", epoch_avg_l2)
    print("Min L2: ", epoch_min_l2)
    print("Min L2 (80%): ", epoch_min_l2_percentiles[0])
    print("Min L2 (90%): ", epoch_min_l2_percentiles[1])
    print("Min L2 (95%): ", epoch_min_l2_percentiles[2])
    print("Min L2 (99%): ", epoch_min_l2_percentiles[3])

    # visualize histogram of min l2
    bins = 200 
    hist_range = (0.0, 1.0)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Histogram (log y)
    os.makedirs('./visualization/', exist_ok=True)
    
    counts, edges, _ = ax.hist(
        min_l2s, bins=bins, range=hist_range,
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
    plt.savefig('./visualization/hist_gazefollow.png', dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
