"""Train SWAG (submitted configuration) with outlier exposure on synthetic mixed scenes, one seed.

Usage: run_oe.py NAME OE_WEIGHT SEED
Writes runs/{NAME}_s{SEED}.npz with S x 8 x N x C validation logits (all 8 dihedral views),
so analyze.py also reports the "+tta_cycle" and "+tta_full" variants.
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


def main():
    name, oe_weight, seed = sys.argv[1:]
    torch.set_num_threads(2)
    train, val = load("train"), load("val")

    seed_setup()
    torch.manual_seed(int(seed))
    start = time.time()
    model = solution.SWAGPredictor(
        train_images=train.tensors[0],
        weights_dir=PROJECT,
        swag_outlier_exposure_weight=float(oe_weight),
        test_time_augmentation="full",
    )
    loader = torch.utils.data.DataLoader(train, batch_size=model.swag_batch_size, shuffle=True, num_workers=0)
    model.run_training(loader)

    val_images, val_snow, val_cloud, val_labels = val.tensors
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_images), batch_size=32)
    model.network.eval()
    with torch.no_grad():
        logits = torch.stack(model._predict_logits_per_sample(val_loader)).numpy()  # S x 8 x N x C
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / f"{name}_s{seed}.npz", logits=logits, labels=val_labels.numpy(),
             snow=val_snow.numpy(), cloud=val_cloud.numpy(), num_snapshots=model._swag_num_snapshots)
    print(f"{name} seed {seed}: {model._swag_num_snapshots} snapshots, {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
