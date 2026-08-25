"""
Google Colab Training Notebook for MeshGraphNet
================================================

This script is designed to be run as a Colab notebook (copy-paste into cells).
It handles:
1. Environment setup (installing dependencies)
2. Mounting Google Drive for data access
3. Data loading from FEM pipeline HDF5 files
4. Model instantiation with 11-dim node features
5. Training with multi-term physics-informed loss
6. Evaluation and visualization

To use:
1. Upload your FEM pipeline output (H5 files + manifest.csv) to Google Drive
2. Clone/upload the gnn_project_version_2 and gnn_bridge code
3. Run this notebook cell by cell

Each section below corresponds to one Colab cell.
"""

# ===========================================================================
# CELL 1: Environment Setup
# ===========================================================================
"""
# Uncomment and run in Colab:

!pip install torch torch_geometric h5py meshio pyyaml pandas matplotlib -q

# If torch_geometric has version issues:
# import torch
# !pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
#     -f https://data.pyg.org/whl/torch-{torch.__version__}.html -q
"""

# ===========================================================================
# CELL 2: Mount Google Drive & Set Paths
# ===========================================================================
"""
# Uncomment in Colab:

from google.colab import drive
drive.mount('/content/drive')

# Adjust these paths to match your Drive structure:
PROJECT_ROOT = '/content/drive/MyDrive/FEM_GNN_Project'
H5_DIR       = f'{PROJECT_ROOT}/output/h5_files'
MANIFEST     = f'{PROJECT_ROOT}/output/manifest.csv'
GNN_CODE     = f'{PROJECT_ROOT}/gnn_project_version_2'
BRIDGE_CODE  = f'{PROJECT_ROOT}/gnn_bridge'
CHECKPOINT_DIR = f'{PROJECT_ROOT}/checkpoints'
LOG_DIR        = f'{PROJECT_ROOT}/logs'
"""

# For local testing, use these paths instead:
import os
import sys

PROJECT_ROOT = r"C:\Projects\Training_Pipeline_version_2"
H5_DIR       = os.path.join(PROJECT_ROOT, "smoke_test_run", "output", "h5_files")
MANIFEST     = os.path.join(PROJECT_ROOT, "smoke_test_run", "output", "manifest.csv")
GNN_CODE     = r"C:\Projects\gnn_project_version_2"
BRIDGE_CODE  = os.path.join(PROJECT_ROOT, "gnn_bridge")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR        = os.path.join(PROJECT_ROOT, "logs")

# Add code to path
sys.path.insert(0, GNN_CODE)
sys.path.insert(0, PROJECT_ROOT)

# ===========================================================================
# CELL 3: Imports
# ===========================================================================
import logging
import time
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
# CELL 4: Load Data
# ===========================================================================
from gnn_bridge.dataloader import (
    FEMGraphDataset,
    compute_field_stds,
    create_dataloaders,
    h5_to_meshdata,
)

# Quick sanity check: load one sample
h5_files = sorted(Path(H5_DIR).glob("*.h5"))
print(f"Found {len(h5_files)} H5 files in {H5_DIR}")

if len(h5_files) > 0:
    sample = h5_to_meshdata(h5_files[0])
    print(f"\nSample graph:")
    print(f"  Nodes:         {sample.x.shape[0]}")
    print(f"  Node features: {sample.x.shape[1]} (expected 11)")
    print(f"  Edges:         {sample.edge_index.shape[1]}")
    print(f"  Edge features: {sample.edge_attr.shape[1]} (expected 4)")
    print(f"  Elements:      {sample.elem_conn.shape[0]}")
    print(f"  Global feat:   {sample.u.shape} → {sample.u.item():.2f}")
    if hasattr(sample, "y_displacement"):
        print(f"  Displacement:  {sample.y_displacement.shape}")
    if hasattr(sample, "y_stress"):
        print(f"  Stress:        {sample.y_stress.shape}")
    if hasattr(sample, "y_von_mises"):
        print(f"  Von Mises:     {sample.y_von_mises.shape}")

# ===========================================================================
# CELL 5: Create DataLoaders
# ===========================================================================
BATCH_SIZE = 4

# Check if manifest has splits
manifest_df = pd.read_csv(MANIFEST)
print(f"\nManifest: {len(manifest_df)} rows")
print(f"Columns: {list(manifest_df.columns)}")

if "split" not in manifest_df.columns:
    print("\n⚠ Manifest does not have 'split' column.")
    print("  Loading all files as training data...")

    # Fallback: load all files, use 80/10/10 random split
    all_dataset = FEMGraphDataset(h5_dir=H5_DIR)
    n = len(all_dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

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
    # Compute field stds from all data
    field_stds_dataset = all_dataset
else:
    print(f"Splits: {manifest_df['split'].value_counts().to_dict()}")
    loaders = create_dataloaders(
        h5_dir=H5_DIR,
        manifest_path=MANIFEST,
        batch_size=BATCH_SIZE,
    )
    field_stds_dataset = FEMGraphDataset(
        h5_dir=H5_DIR,
        manifest_path=MANIFEST,
        split="train",
    )

for name, loader in loaders.items():
    print(f"  {name}: {len(loader.dataset)} samples, {len(loader)} batches")

# ===========================================================================
# CELL 6: Compute Normalisation Statistics
# ===========================================================================
field_stds = compute_field_stds(field_stds_dataset)
print(f"\nField standard deviations:")
for k, v in field_stds.items():
    print(f"  {k}: {v:.6e}")

# ===========================================================================
# CELL 7: Build Model
# ===========================================================================
from models.meshgraphnet import MeshGraphNet
from models.loss import MeshGraphNetLoss

model = MeshGraphNet(
    node_in_dim=11,       # 11-dim: [x,y,z, fixed, loaded, surface, hops_bc, hops_load, Fx, Fy, Fz]
    edge_in_dim=4,        # [dx, dy, dz, length]
    global_in_dim=1,      # [total_load_magnitude]
    hidden_dim=128,
    num_processor_layers=15,
    stress_net_local_mp_layers=3,
    E=210e9,
    nu=0.3,
    yield_strength=420e6,
)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nMeshGraphNet: {n_params:,} trainable parameters")
print(f"Architecture:")
print(f"  Node encoder:     11 → 128")
print(f"  Edge encoder:      4 → 128")
print(f"  Global encoder:    1 → 128")
print(f"  Processor:        15 MPNN layers")
print(f"  Displacement dec: 128 → 3")
print(f"  StressNet:         3 local MP layers")
print(f"  Physics bridge:   B-matrix strain + VM/SF")

loss_fn = MeshGraphNetLoss(
    field_stds=field_stds,
    w_u=1.0,
    w_sigma=0.5,
    w_eps=0.3,
    w_eps_corr=0.3,
    w_vm=0.2,
)

# ===========================================================================
# CELL 8: Train
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
)

print(f"\n{'=' * 60}")
print(f"  TRAINING: {NUM_EPOCHS} epochs, lr={LR}, batch_size={BATCH_SIZE}")
print(f"  Device: {device}")
print(f"{'=' * 60}\n")

history = trainer.train(num_epochs=NUM_EPOCHS)

# ===========================================================================
# CELL 9: Plot Training Curves
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
if history["val_loss"]:
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
plt.savefig(os.path.join(LOG_DIR, "training_curves.png"), dpi=150)
plt.show()

# ===========================================================================
# CELL 10: Evaluate
# ===========================================================================
from gnn_bridge.metrics import evaluate_model, print_evaluation_report

# Load best model
best_ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pt")
if os.path.exists(best_ckpt):
    trainer.load_checkpoint(best_ckpt)
    print("Loaded best model checkpoint")

for split_name in ("val", "test"):
    if split_name in loaders:
        results = evaluate_model(model, loaders[split_name], device=str(device))
        print_evaluation_report(results, split_name)

# ===========================================================================
# CELL 11: Per-Sample Timing (FEM vs GNN)
# ===========================================================================
# If the manifest has solve_time_s, compare with GNN inference time
if "solve_time_s" in manifest_df.columns:
    fem_times = manifest_df["solve_time_s"].dropna()
    avg_fem_s = fem_times.mean()

    # Measure GNN inference time
    test_loader = loaders.get("test", loaders.get("val", loaders["train"]))
    model.eval()
    gnn_times = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            t0 = time.perf_counter()
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            n = batch.num_graphs if hasattr(batch, "num_graphs") else 1
            gnn_times.extend([(t1 - t0) / n] * n)

    avg_gnn_s = np.mean(gnn_times) if gnn_times else 0

    print(f"\n{'=' * 60}")
    print(f"  SPEED COMPARISON")
    print(f"{'=' * 60}")
    print(f"  Avg FEM solve:     {avg_fem_s:.2f}s")
    print(f"  Avg GNN inference: {avg_gnn_s * 1000:.1f}ms")
    if avg_gnn_s > 0:
        print(f"  Speedup:           {avg_fem_s / avg_gnn_s:.0f}x")
    print(f"{'=' * 60}\n")

print("\n✓ Training pipeline complete!")
