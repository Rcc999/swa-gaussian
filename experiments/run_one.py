"""Train one SWAG configuration with one seed and dump per-sample validation logits.

Usage: run_one.py NAME MODE LR BURN_IN UPDATES_PER_EPOCH SEED [AUGMENT CUTMIX_PROB]
Validation logits are dumped for all 8 dihedral views (S x 8 x N x C) when AUGMENT/CUTMIX_PROB are given.
"""
import pathlib
import sys
import time

PROJECT = pathlib.Path("/Users/rayanecharif/master/MA2/PAI/projects/swa-gaussian")
OUT = pathlib.Path(__file__).parent / "runs"
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch

import solution
from util import seed_setup


def load(split):
    xs = torch.from_numpy(np.load(PROJECT / f"{split}_xs.npz")[f"{split}_xs"])
    meta = np.load(PROJECT / f"{split}_ys.npz")
    return torch.utils.data.TensorDataset(
        xs,
        torch.from_numpy(meta[f"{split}_is_snow"]),
        torch.from_numpy(meta[f"{split}_is_cloud"]),
        torch.from_numpy(meta[f"{split}_ys"]),
    )


def main():
    name, mode, lr, burn_in, updates_per_epoch, seed = sys.argv[1:7]
    augment, cutmix_prob = (sys.argv[7], sys.argv[8]) if len(sys.argv) > 7 else (None, None)
    torch.set_num_threads(2)
    train, val = load("train"), load("val")

    seed_setup()
    torch.manual_seed(int(seed))
    start = time.time()
    model = solution.SWAGPredictor(
        train_images=train.tensors[0],
        weights_dir=PROJECT,
        inference_mode=solution.InferenceMode[mode],
        swag_lr=float(lr),
        swag_burn_in_epochs=int(burn_in),
        swag_updates_per_epoch=int(updates_per_epoch),
        swag_augment=augment == "1",
        swag_cutmix_prob=float(cutmix_prob or 0.0),
        test_time_augmentation="none" if augment is None else "full",
    )
    loader = torch.utils.data.DataLoader(train, batch_size=model.swag_batch_size, shuffle=True, num_workers=0)
    model.run_training(loader)

    val_images, val_snow, val_cloud, val_labels = val.tensors
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_images), batch_size=32)
    model.network.eval()
    with torch.no_grad():
        logits = torch.stack(model._predict_logits_per_sample(val_loader)).numpy()  # S x V x N x C
    if augment is None:
        logits = logits[:, 0]  # S x N x C, as in the first sweep

    OUT.mkdir(exist_ok=True)
    np.savez(
        OUT / f"{name}_s{seed}.npz",
        logits=logits,
        labels=val_labels.numpy(),
        snow=val_snow.numpy(),
        cloud=val_cloud.numpy(),
        num_snapshots=model._swag_num_snapshots,
    )
    print(f"{name} seed {seed}: {model._swag_num_snapshots} snapshots, {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
