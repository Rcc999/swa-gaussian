"""Train one SWAG configuration with one seed, then draw posterior samples at several variance scales.

Usage: run_posterior.py NAME LR SCHEDULE LR_MIN BATCH_SIZE SEED
Writes runs/{NAME}_x{SCALE}_s{SEED}.npz with S x N x C validation logits for every scale in SCALES.
All runs use full SWAG with 5 burn-in epochs and 4 snapshots per epoch (the submitted configuration).
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
from run_one import load
from util import seed_setup

SCALES = [1.0, 2.0, 4.0]


def main():
    name, lr, schedule, lr_min, batch_size, seed = sys.argv[1:]
    torch.set_num_threads(2)
    train, val = load("train"), load("val")

    seed_setup()
    torch.manual_seed(int(seed))
    start = time.time()
    model = solution.SWAGPredictor(
        train_images=train.tensors[0],
        weights_dir=PROJECT,
        swag_lr=float(lr),
        swag_lr_schedule=schedule,
        swag_lr_min=float(lr_min),
        swag_batch_size=int(batch_size),
    )
    loader = torch.utils.data.DataLoader(train, batch_size=model.swag_batch_size, shuffle=True, num_workers=0)
    model.run_training(loader)

    val_images, val_snow, val_cloud, val_labels = val.tensors
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_images), batch_size=32)
    OUT.mkdir(exist_ok=True)
    for scale in SCALES:
        model.swag_variance_scale = scale
        model._posterior_samples = None  # resample from the rescaled posterior
        model.network.eval()
        with torch.no_grad():
            logits = torch.stack(model._predict_logits_per_sample(val_loader))[:, 0].numpy()  # S x N x C
        np.savez(OUT / f"{name}_x{scale:g}_s{seed}.npz", logits=logits, labels=val_labels.numpy(),
                 snow=val_snow.numpy(), cloud=val_cloud.numpy(), num_snapshots=model._swag_num_snapshots)
    print(f"{name} seed {seed}: {model._swag_num_snapshots} snapshots, {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
