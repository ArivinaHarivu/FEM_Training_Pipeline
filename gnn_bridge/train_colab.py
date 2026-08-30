"""
Google Colab Training Script for MeshGraphNet (FEM 3D Surrogate)
================================================================

This script is structured in modular blocks ready to run cell-by-cell in Google Colab.
It handles:
1. Environment setup (installing dependencies)
2. Mounting Google Drive or using direct local upload
3. Automatic base-sample-aware train/val/test splitting (if split column is missing)
4. Data loading from FEM pipeline HDF5 files (tet10 quadratic elements)
5. Ground-truth strain derivation via inverse Hooke's law (for L_eps / L_eps_corr)
6. Model instantiation (11-dim node features, 4-dim edge features, 1-dim global)
7. Training with AdaptiveLossBalancer (automatic live loss rebalancing)
8. Plotting loss curves & learning rate schedule
9. Comprehensive evaluation report (MAE, RMSE, Max Error, Relative Error)
10. FEM vs GNN inference speedup benchmark
"""

# ===========================================================================
# CELL 1: Environment Setup
# ===========================================================================
"""
# Run in Colab:
!pip install torch torch_geometric h5py meshio pyyaml pandas matplotlib -q
"""

# ===========================================================================
# CELL 2: Paths & Project Setup
# ===========================================================================
"""
# Run in Colab:
from google.colab import drive
import os, sys

drive.mount('/content/drive')

# Set your project directory on Google Drive:
PROJECT_ROOT = '/content/drive/MyDrive/FEM_GNN_Project'
H5_DIR       = f'{PROJECT_ROOT}/output/raw'         # or output/h5_files
MANIFEST     = f'{PROJECT_ROOT}/output/manifest.csv'
GNN_CODE     = f'{PROJECT_ROOT}/gnn_project_version_2'
BRIDGE_CODE  = f'{PROJECT_ROOT}/Training_Pipeline_version_2'

sys.path.insert(0, GNN_CODE)
sys.path.insert(0, BRIDGE_CODE)
"""

import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("colab_training")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ===========================================================================
# CELL 3: Data Discovery & Auto-Splitting
# ===========================================================================
# Set local / colab paths:
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
H5_DIR = Path(os.environ.get("H5_DIR", "smoke_test_run/output/raw"))
MANIFEST = Path(os.environ.get("MANIFEST", "smoke_test_run/output/manifest.csv"))
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Material properties (Domex 420MC structural steel)
MAT_E = 210.0e9          # Young's modulus [Pa]
MAT_NU = 0.3            # Poisson's ratio [-]
MAT_YIELD = 420.0e6     # Yield strength [Pa]

# Auto-check manifest splits
if MANIFEST.exists():
    manifest_df = pd.read_csv(MANIFEST)
    print(f"Manifest found: {len(manifest_df)} rows")
    if "split" not in manifest_df.columns:
        print("⚠ 'split' column missing. Auto-assigning base-sample-safe 80/10/10 splits...")
        base_ids = manifest_df["base_sample_id"].unique()
        rng = np.random.default_rng(42)
        rng.shuffle(base_ids)
        
        n_train = max(1, int(0.8 * len(base_ids)))
        n_val = max(1, int(0.1 * len(base_ids)))
        
        train_bases = set(base_ids[:n_train])
        val_bases = set(base_ids[n_train:n_train + n_val])
        
        manifest_df["split"] = manifest_df["base_sample_id"].apply(
            lambda bid: "train" if bid in train_bases else ("val" if bid in val_bases else "test")
        )
        manifest_df.to_csv(MANIFEST, index=False)
        print("✓ Created and saved 'split' column to manifest.csv:")
        print(f"  {manifest_df['split'].value_counts().to_dict()}")
else:
    print(f"Manifest not found at {MANIFEST}. Will discover H5 files directly.")


# ===========================================================================
# CELL 4: DataLoaders & Field Normalisation
# ===========================================================================
from gnn_bridge.dataloader import (
    FEMGraphDataset,
    compute_field_stds,
    create_dataloaders,
    h5_to_meshdata,
)

BATCH_SIZE = 4

# Load DataLoaders
if MANIFEST.exists():
    loaders = create_dataloaders(
        h5_dir=H5_DIR,
        manifest_path=MANIFEST,
        batch_size=BATCH_SIZE,
        E=MAT_E,
        nu=MAT_NU,
    )
    train_dataset = FEMGraphDataset(
        h5_dir=H5_DIR,
        manifest_path=MANIFEST,
        split="train",
        E=MAT_E,
        nu=MAT_NU,
    )
else:
    all_dataset = FEMGraphDataset(h5_dir=H5_DIR, E=MAT_E, nu=MAT_NU)
    n = len(all_dataset)
    n_train = max(1, int(0.8 * n))
    n_val = max(1, int(0.1 * n))
    n_test = max(1, n - n_train - n_val)
    
    from torch.utils.data import random_split
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    train_ds, val_ds, test_ds = random_split(
        all_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    loaders = {
        "train": PyGDataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        "val":   PyGDataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False),
        "test":  PyGDataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False),
    }
    train_dataset = all_dataset

for name, loader in loaders.items():
    print(f"  Split '{name}': {len(loader.dataset)} samples ({len(loader)} batches)")

# Compute loss normalisation statistics
field_stds = compute_field_stds(train_dataset)
print(f"\nField standard deviations for loss normalisation:")
for k, v in field_stds.items():
    print(f"  {k:8s}: {v:.6e}")


# ===========================================================================
# CELL 5: Model & Loss Instantiation
# ===========================================================================
from models.meshgraphnet import MeshGraphNet
from models.loss import MeshGraphNetLoss

model = MeshGraphNet(
    node_in_dim=11,       # [x, y, z, is_fixed, is_loaded, is_surface, hops_bc, hops_load, Fx, Fy, Fz]
    edge_in_dim=4,        # [dx, dy, dz, length]
    global_in_dim=1,      # [total_load_magnitude]
    hidden_dim=128,
    num_processor_layers=15,
    stress_net_local_mp_layers=3,
    E=MAT_E,
    nu=MAT_NU,
    yield_strength=MAT_YIELD,
)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nMeshGraphNet: {n_params:,} trainable parameters")

loss_fn = MeshGraphNetLoss(
    field_stds=field_stds,
    w_u=1.0,
    w_sigma=0.5,
    w_eps=0.3,
    w_eps_corr=0.3,
    w_vm=0.2,
)


# ===========================================================================
# CELL 6: Training Loop (With Live Adaptive Loss Balancer)
# ===========================================================================
from gnn_bridge.trainer import Trainer

NUM_EPOCHS = 100
LR = 1e-3

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    train_loader=loaders["train"],
    val_loader=loaders.get("val"),
    lr=LR,
    grad_clip_norm=1.0,
    device=str(device),
    checkpoint_dir=CHECKPOINT_DIR,
    log_dir=LOG_DIR,
    adaptive_loss_weighting=False,
    checkpoint_interval_batches=25,
)

# Auto-resume if previous session was interrupted / disconnected
resume_state = trainer.resume_latest()
if resume_state:
    print(f"✓ Resuming training from Epoch {resume_state['next_epoch']}, Batch {resume_state['start_batch']}!")
else:
    print("Starting fresh training run...")

print(f"\n{'=' * 65}")
print(f"  STARTING TRAINING: {NUM_EPOCHS} epochs, lr={LR}, batch_size={BATCH_SIZE}")
print(f"  Device: {device} | Adaptive Loss Balancer: Disabled (Static Weighted Sum)")
print(f"{'=' * 65}\n")

history = trainer.train(num_epochs=NUM_EPOCHS, resume_state=resume_state)


# ===========================================================================
# CELL 7: Plot Training Curves
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
if history["val_loss"] and any(v != float("inf") for v in history["val_loss"]):
    axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Total Loss")
axes[0].set_title("Training & Validation Loss")
axes[0].legend()
axes[0].set_yscale("log")
axes[0].grid(True, alpha=0.3)

# Learning rate
axes[1].plot(history["lr"], linewidth=2, color="green")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Learning Rate")
axes[1].set_title("Learning Rate Schedule")
axes[1].set_yscale("log")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
curve_path = LOG_DIR / "training_curves.png"
plt.savefig(str(curve_path), dpi=150)
print(f"Saved training curves to {curve_path}")
plt.show()


# ===========================================================================
# CELL 8: Comprehensive Evaluation Report
# ===========================================================================
from gnn_bridge.metrics import evaluate_model, print_evaluation_report

# Load best checkpoint
best_ckpt = CHECKPOINT_DIR / "best_model.pt"
if best_ckpt.exists():
    trainer.load_checkpoint(best_ckpt)
    print(f"\nLoaded best model from {best_ckpt}")

for split_name in ("val", "test"):
    if split_name in loaders:
        results = evaluate_model(model, loaders[split_name], device=str(device))
        print_evaluation_report(results, split_name)

print("\n✓ Colab Training Run Complete!")
