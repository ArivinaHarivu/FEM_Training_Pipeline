"""run_training.py — CLI entry point for MeshGraphNet training.

Usage (local):
    python -m gnn_bridge.run_training \
        --h5_dir output/h5_files \
        --manifest output/manifest.csv \
        --epochs 100 \
        --batch_size 4

Usage (Colab):
    See the companion notebook ``train_meshgraphnet.ipynb``.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MeshGraphNet on FEM pipeline data.",
    )
    parser.add_argument("--h5_dir", type=str, required=True,
                        help="Directory containing .h5 files")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to manifest.csv (must have 'split' column)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to GNN config.yaml (optional)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Model hidden dimension (e.g. 64 or 128)")
    parser.add_argument("--num_layers", type=int, default=None,
                        help="Number of processor MPNN layers (e.g. 8)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="'cuda' or 'cpu' (auto-detected if omitted)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--checkpoint_interval_batches", type=int, default=50,
                        help="Save a batch-level checkpoint every N batches (default 50)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt to resume from, or 'auto' to "
                             "automatically find and resume from checkpoint_latest.pt")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, run evaluation only")
    parser.add_argument("--adaptive_loss", action="store_true",
                        help="Enable live adaptive loss-term rebalancing "
                             "(off by default, uses loss_fn's static "
                             "w_u/w_sigma/... weights as-is)")
    parser.add_argument("--loss_balance_momentum", type=float, default=0.9,
                        help="EMA momentum for adaptive loss-term "
                             "rebalancing (higher = smoother/slower to react)")
    args = parser.parse_args()

    # ----- Load config -----
    config = _default_config()
    if args.config:
        with open(args.config) as f:
            config.update(yaml.safe_load(f))

    # Merge config file values into args if not explicitly passed on CLI
    training_cfg = config.get("training", {})
    if args.batch_size is None:
        args.batch_size = training_cfg.get("batch_size", 4)
    if args.epochs is None:
        args.epochs = training_cfg.get("epochs", 100)
    if args.lr is None:
        args.lr = training_cfg.get("lr", 1e-3)
    if args.hidden_dim is None:
        args.hidden_dim = config.get("model", {}).get("hidden_dim", 128)
    if args.num_layers is None:
        args.num_layers = config.get("model", {}).get("num_processor_layers", 15)

    # ----- Device -----
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # ----- Data -----
    mat_cfg = config.get("material", {})
    E = mat_cfg.get("E", 210.0e9)
    nu = mat_cfg.get("nu", 0.3)
    yield_strength = mat_cfg.get("sigma_yield", mat_cfg.get("yield_strength", 420.0e6))

    data_cfg = config.get("data", {})
    manifest_path = args.manifest or data_cfg.get("manifest_path", "output/manifest.csv")
    h5_dir = args.h5_dir or data_cfg.get("h5_dir", "output/raw")

    # ----- Import heavy modules after arg parsing -----
    from gnn_bridge.dataloader import (
        FEMGraphDataset,
        compute_field_stds,
        create_dataloaders,
    )
    from gnn_bridge.trainer import Trainer
    from gnn_bridge.metrics import evaluate_model, print_evaluation_report

    loaders = create_dataloaders(
        h5_dir=h5_dir,
        manifest_path=manifest_path,
        batch_size=args.batch_size,
        E=E,
        nu=nu,
        num_workers=data_cfg.get("num_workers", 0),
        seed=data_cfg.get("seed", 42),
    )

    # Field stats for loss normalisation — reuse the train loader's dataset
    # instead of creating a second FEMGraphDataset (avoids re-reading
    # every H5 file from disk / FUSE a second time).
    field_stds = compute_field_stds(
        loaders["train"].dataset,
        max_samples=data_cfg.get("field_stats_samples", 50),
    )

    # ----- Model -----
    model_cfg = config.get("model", {})

    # Import model — search all candidate locations (Colab root, sibling directory, cwd)
    candidates = [
        Path("/content/gnn_project_version_2"),
        Path(__file__).resolve().parent.parent.parent / "gnn_project_version_2",
        Path(__file__).resolve().parent.parent / "gnn_project_version_2",
        Path.cwd() / "gnn_project_version_2",
        Path.cwd().parent / "gnn_project_version_2",
    ]
    for p in candidates:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    try:
        from models.meshgraphnet import MeshGraphNet
        from models.loss import MeshGraphNetLoss
        logger.info("Loaded MeshGraphNet from gnn_project_version_2")
    except ImportError:
        try:
            from meshgraphnet import MeshGraphNet
            from loss import MeshGraphNetLoss
        except ImportError as e:
            logger.error(
                "Cannot import MeshGraphNet (%s). Ensure gnn_project_version_2 is on sys.path.",
                e,
            )
            sys.exit(1)

    hidden_dim = (
        args.hidden_dim
        if args.hidden_dim is not None
        else model_cfg.get("hidden_dim", 64)
    )
    num_layers = (
        args.num_layers
        if args.num_layers is not None
        else model_cfg.get("num_processor_layers", 8)
    )

    model = MeshGraphNet(
        node_in_dim=model_cfg.get("node_in_dim", 11),
        edge_in_dim=model_cfg.get("edge_in_dim", 4),
        global_in_dim=model_cfg.get("global_in_dim", 1),
        hidden_dim=hidden_dim,
        num_processor_layers=num_layers,
        stress_net_local_mp_layers=model_cfg.get("stress_net_local_mp_layers", 3),
        E=E,
        nu=nu,
        yield_strength=yield_strength,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("MeshGraphNet: %d trainable parameters", n_params)

    loss_fn = MeshGraphNetLoss(
        field_stds=field_stds,
        w_u=config.get("training", {}).get("loss_weights", {}).get("u", 1.0),
        w_sigma=config.get("training", {}).get("loss_weights", {}).get("sigma", 0.5),
        w_eps=config.get("training", {}).get("loss_weights", {}).get("eps", 0.3),
        w_eps_corr=config.get("training", {}).get("loss_weights", {}).get("eps_corr", 0.3),
        w_vm=config.get("training", {}).get("loss_weights", {}).get("vm", 0.2),
    )

    # ----- Trainer -----
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=loaders["train"],
        val_loader=loaders.get("val"),
        lr=args.lr,
        grad_clip_norm=config.get("training", {}).get("grad_clip_norm", 1.0),
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        adaptive_loss_weighting=args.adaptive_loss,
        loss_balance_momentum=args.loss_balance_momentum,
        checkpoint_interval_batches=args.checkpoint_interval_batches,
    )

    resume_state = None
    if args.resume:
        if args.resume.lower() == "auto":
            resume_state = trainer.resume_latest()
            if resume_state is None:
                logger.info("No checkpoint_latest.pt found in %s; starting fresh.", args.checkpoint_dir)
        else:
            resume_state = trainer.load_checkpoint(args.resume)

    # ----- Train -----
    if not args.eval_only:
        logger.info("Starting training for %d epochs...", args.epochs)
        history = trainer.train(num_epochs=args.epochs, resume_state=resume_state)
        logger.info("Training complete. Best val loss: %.6f", trainer._best_val_loss)

    # ----- Evaluate -----
    for split_name in ("val", "test"):
        if split_name in loaders:
            logger.info("Evaluating on %s set...", split_name)
            results = evaluate_model(
                model, loaders[split_name],
                device=str(trainer.device),
            )
            print_evaluation_report(results, split_name)


def _default_config() -> dict:
    """Return the default configuration matching the GNN project."""
    return {
        "model": {
            "node_in_dim": 11,
            "edge_in_dim": 4,
            "global_in_dim": 1,
            "hidden_dim": 128,
            "num_processor_layers": 15,
            "stress_net_local_mp_layers": 3,
        },
        "training": {
            "lr": 1e-3,
            "grad_clip_norm": 1.0,
            "loss_weights": {
                "u": 1.0,
                "sigma": 0.5,
                "eps": 0.3,
                "eps_corr": 0.3,
                "vm": 0.2,
            },
        },
        "material": {
            "E": 210e9,
            "nu": 0.3,
            "yield_strength": 420e6,
        },
    }


if __name__ == "__main__":
    main()
