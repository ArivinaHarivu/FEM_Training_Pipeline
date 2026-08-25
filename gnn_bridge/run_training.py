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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None,
                        help="'cuda' or 'cpu' (auto-detected if omitted)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt to resume from")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, run evaluation only")
    args = parser.parse_args()

    # ----- Load config -----
    config = _default_config()
    if args.config:
        with open(args.config) as f:
            config.update(yaml.safe_load(f))

    # ----- Import heavy modules after arg parsing -----
    from gnn_bridge.dataloader import (
        FEMGraphDataset,
        compute_field_stds,
        create_dataloaders,
    )
    from gnn_bridge.trainer import Trainer
    from gnn_bridge.metrics import evaluate_model, print_evaluation_report

    # ----- DataLoaders -----
    logger.info("Creating DataLoaders from %s", args.h5_dir)
    loaders = create_dataloaders(
        h5_dir=args.h5_dir,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
    )

    if "train" not in loaders:
        logger.error("No training data found. Check manifest and h5_dir.")
        sys.exit(1)

    # ----- Field statistics for loss normalisation -----
    logger.info("Computing field statistics from training data...")
    train_dataset = FEMGraphDataset(
        h5_dir=args.h5_dir,
        manifest_path=args.manifest,
        split="train",
    )
    field_stds = compute_field_stds(train_dataset)

    # ----- Model -----
    model_cfg = config.get("model", {})
    material_cfg = config.get("material", {})

    # Import model — try from gnn_project_version_2 first, fallback to inline
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn_project_version_2"))
        from models.meshgraphnet import MeshGraphNet
        from models.loss import MeshGraphNetLoss
        logger.info("Loaded MeshGraphNet from gnn_project_version_2")
    except ImportError:
        try:
            # Colab: gnn project may be at a different path
            from meshgraphnet import MeshGraphNet
            from loss import MeshGraphNetLoss
        except ImportError:
            logger.error(
                "Cannot import MeshGraphNet. Ensure gnn_project_version_2 "
                "is on sys.path or install the models package."
            )
            sys.exit(1)

    model = MeshGraphNet(
        node_in_dim=model_cfg.get("node_in_dim", 11),
        edge_in_dim=model_cfg.get("edge_in_dim", 4),
        global_in_dim=model_cfg.get("global_in_dim", 1),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_processor_layers=model_cfg.get("num_processor_layers", 15),
        stress_net_local_mp_layers=model_cfg.get("stress_net_local_mp_layers", 3),
        E=material_cfg.get("E", 210e9),
        nu=material_cfg.get("nu", 0.3),
        yield_strength=material_cfg.get("yield_strength", 420e6),
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
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ----- Train -----
    if not args.eval_only:
        logger.info("Starting training for %d epochs...", args.epochs)
        history = trainer.train(num_epochs=args.epochs)
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
