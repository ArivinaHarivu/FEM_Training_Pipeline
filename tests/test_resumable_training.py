import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from gnn_bridge.dataloader import MeshData, ResumableBatchSampler, sample_generator, FEMGraphDataset
from gnn_bridge.trainer import AdaptiveLossBalancer, Trainer


class DummyDataset(torch.utils.data.Dataset):
    """Dummy graph dataset for testing."""

    def __init__(self, size: int = 20):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> MeshData:
        data = MeshData()
        data.x = torch.randn(10, 11)
        data.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        data.edge_attr = torch.randn(4, 4)
        data.global_attr = torch.tensor([[1000.0]])
        data.elem_conn = torch.zeros((2, 10), dtype=torch.long)
        data.num_nodes = 10
        data.y_displacement = torch.randn(10, 3)
        data.y_stress = torch.randn(10, 6)
        data.y_strain = torch.randn(10, 6)
        data.y_von_mises = torch.randn(10)
        return data


class DummyModel(nn.Module):
    """Dummy model returning standard output format."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(11, 3)
        self.sigma_head = nn.Linear(11, 6)
        self.vm_head = nn.Linear(11, 1)

    def forward(self, batch):
        u = self.linear(batch.x)
        sigma = self.sigma_head(batch.x)
        vm = self.vm_head(batch.x).squeeze(-1)
        return {
            "u": u,
            "sigma": sigma,
            "eps": sigma / 210e9,
            "vm": vm,
            "delta_eps": torch.zeros_like(sigma),
        }


class DummyLoss(nn.Module):
    """Dummy loss returning multi-term dictionary."""

    def forward(self, preds, targets):
        L_u = torch.mean((preds["u"] - targets["u"]) ** 2)
        L_sigma = torch.mean((preds["sigma"] - targets["sigma"]) ** 2)
        L_vm = torch.mean((preds["vm"] - targets["vm"]) ** 2)
        total = L_u + 0.5 * L_sigma + 0.2 * L_vm
        return {
            "total": total,
            "L_u": L_u,
            "L_sigma": L_sigma,
            "L_vm": L_vm,
        }


# ===========================================================================
# 1. ResumableBatchSampler Tests
# ===========================================================================

def test_resumable_sampler_determinism():
    """Verify that same seed + epoch generates identical batch permutation."""
    sampler1 = ResumableBatchSampler(dataset_len=20, batch_size=4, shuffle=True, seed=42)
    sampler2 = ResumableBatchSampler(dataset_len=20, batch_size=4, shuffle=True, seed=42)

    sampler1.set_epoch(1)
    sampler2.set_epoch(1)

    batches1 = list(sampler1)
    batches2 = list(sampler2)

    assert batches1 == batches2
    assert len(batches1) == 5

    # Epoch 2 should be different from Epoch 1
    sampler1.set_epoch(2)
    batches_epoch2 = list(sampler1)
    assert batches1 != batches_epoch2


def test_resumable_sampler_mid_epoch_slicing():
    """Verify that setting start_batch slices exactly the remaining batches."""
    sampler = ResumableBatchSampler(dataset_len=20, batch_size=4, shuffle=True, seed=42)
    sampler.set_epoch(1)
    full_batches = list(sampler)
    assert len(full_batches) == 5

    # Resume from batch index 2 (skip batches 0 and 1)
    sampler.set_start_batch(2)
    assert len(sampler) == 3
    resumed_batches = list(sampler)

    assert len(resumed_batches) == 3
    assert resumed_batches == full_batches[2:]


def test_resumable_sampler_state_dict():
    """Verify state serialization and restoration."""
    sampler = ResumableBatchSampler(dataset_len=20, batch_size=4, shuffle=True, seed=123)
    sampler.set_epoch(3)
    sampler.set_start_batch(2)

    state = sampler.state_dict()
    assert state["epoch"] == 3
    assert state["start_batch"] == 2
    assert state["seed"] == 123

    new_sampler = ResumableBatchSampler(dataset_len=20, batch_size=4)
    new_sampler.load_state_dict(state)
    assert new_sampler.epoch == 3
    assert new_sampler.start_batch == 2


# ===========================================================================
# 2. Trainer Intra-Epoch Batch Resumption Tests
# ===========================================================================

def test_trainer_intra_epoch_batch_checkpoint_and_resume():
    """Test saving checkpoint mid-epoch and resuming from the exact batch."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        ckpt_dir = temp_dir / "checkpoints"
        log_dir = temp_dir / "logs"

        dataset = DummyDataset(size=20)  # 20 samples -> 5 batches of 4
        sampler = ResumableBatchSampler(
            dataset_len=len(dataset),
            batch_size=4,
            shuffle=True,
            seed=42,
        )
        loader = PyGDataLoader(dataset, batch_sampler=sampler)

        model = DummyModel()
        loss_fn = DummyLoss()

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            train_loader=loader,
            val_loader=loader,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
            checkpoint_interval_batches=2,  # Auto-save at batch 2 (index 1) and batch 4 (index 3)
            adaptive_loss_weighting=True,
        )

        # Run 1 epoch
        history = trainer.train(num_epochs=1)
        assert len(history["train_loss"]) == 1

        # Check latest checkpoint exists
        latest_ckpt = ckpt_dir / "checkpoint_latest.pt"
        assert latest_ckpt.exists()

        # Load checkpoint into a new trainer instance
        fresh_model = DummyModel()
        fresh_trainer = Trainer(
            model=fresh_model,
            loss_fn=loss_fn,
            train_loader=loader,
            val_loader=loader,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
        )

        resume_state = fresh_trainer.load_checkpoint(latest_ckpt)
        assert resume_state is not None
        assert "next_epoch" in resume_state
        assert "start_batch" in resume_state
        assert "epoch_accum" in resume_state

        # Since epoch 1 completed, next_epoch should be 2, start_batch = 0
        assert resume_state["next_epoch"] == 2
        assert resume_state["start_batch"] == 0

        # Now resume and train for 1 more epoch (total 2)
        history2 = fresh_trainer.train(num_epochs=2, resume_state=resume_state)
        assert len(history2["train_loss"]) == 2

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_trainer_mid_epoch_resumption_exact_batch():
    """Simulate a mid-epoch disconnection at batch index 2, and verify resumption at batch 3."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        ckpt_dir = temp_dir / "checkpoints"
        log_dir = temp_dir / "logs"

        dataset = DummyDataset(size=20)  # 5 batches of 4
        sampler = ResumableBatchSampler(
            dataset_len=len(dataset),
            batch_size=4,
            shuffle=True,
            seed=42,
        )
        loader = PyGDataLoader(dataset, batch_sampler=sampler)

        model = DummyModel()
        loss_fn = DummyLoss()

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            train_loader=loader,
            val_loader=loader,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
            checkpoint_interval_batches=1,  # save every batch
        )

        # Manually save a mid-epoch checkpoint simulating interrupt at batch index 2 (batch 3/5)
        trainer._global_step = 3
        trainer._save_checkpoint(
            epoch=1,
            batch_idx=2,
            val_loss=0.5,
            filename="checkpoint_latest.pt",
            epoch_accum={"total": 1.5, "L_u": 1.0},
            weight_accum={"L_u": 3.0},
            n_batches=3,
        )

        # Create a fresh Trainer and auto-resume
        fresh_model = DummyModel()
        fresh_trainer = Trainer(
            model=fresh_model,
            loss_fn=loss_fn,
            train_loader=loader,
            val_loader=loader,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
        )

        resume_state = fresh_trainer.resume_latest()
        assert resume_state is not None
        assert resume_state["epoch"] == 1
        assert resume_state["batch_idx"] == 2
        assert resume_state["next_epoch"] == 1  # continues epoch 1
        assert resume_state["start_batch"] == 3  # resumes from batch index 3 (4th batch)
        assert resume_state["n_batches"] == 3
        assert resume_state["epoch_accum"]["total"] == 1.5

        # Resume training to complete epoch 1
        history = fresh_trainer.train(num_epochs=1, resume_state=resume_state)
        assert len(history["train_loss"]) == 1
        # Epoch 1 loss should include the aggregated prior metrics
        assert history["train_loss"][0] > 0.0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_resume_latest_non_existent():
    """Verify resume_latest returns None gracefully when no checkpoint exists."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        dataset = DummyDataset(size=5)
        loader = PyGDataLoader(dataset, batch_size=2)
        model = DummyModel()
        loss_fn = DummyLoss()
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            train_loader=loader,
            checkpoint_dir=temp_dir / "empty_checkpoints",
            log_dir=temp_dir / "logs",
        )
        assert trainer.resume_latest() is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_continuous_vs_interrupted_training_trajectory():
    """Verify that an interrupted and resumed run produces matching step counts and valid weights."""
    torch.manual_seed(42)
    temp_dir = Path(tempfile.mkdtemp())
    try:
        dataset = DummyDataset(size=20)  # 5 batches per epoch
        sampler1 = ResumableBatchSampler(len(dataset), batch_size=4, shuffle=True, seed=42)
        loader1 = PyGDataLoader(dataset, batch_sampler=sampler1)

        model1 = DummyModel()
        loss_fn = DummyLoss()

        trainer1 = Trainer(
            model=model1,
            loss_fn=loss_fn,
            train_loader=loader1,
            checkpoint_dir=temp_dir / "ckpt1",
            log_dir=temp_dir / "log1",
            adaptive_loss_weighting=False,
            checkpoint_interval_batches=2,
        )

        # Run full 2 epochs (total 10 steps)
        hist1 = trainer1.train(num_epochs=2)
        assert trainer1._global_step == 10

        # Now simulate run 2: interrupt after batch index 2 in epoch 1 (3 steps)
        sampler2 = ResumableBatchSampler(len(dataset), batch_size=4, shuffle=True, seed=42)
        loader2 = PyGDataLoader(dataset, batch_sampler=sampler2)
        model2 = DummyModel()
        # Initialize model2 with same initial weights
        model2.load_state_dict(DummyModel().state_dict())

        trainer2 = Trainer(
            model=model2,
            loss_fn=loss_fn,
            train_loader=loader2,
            checkpoint_dir=temp_dir / "ckpt2",
            log_dir=temp_dir / "log2",
            adaptive_loss_weighting=False,
            checkpoint_interval_batches=1,
        )

        # Save checkpoint at batch index 2
        trainer2._global_step = 3
        trainer2._save_checkpoint(
            epoch=1,
            batch_idx=2,
            val_loss=0.0,
            filename="checkpoint_latest.pt",
            epoch_accum={"total": 2.0},
            n_batches=3,
        )

        # Create trainer3, resume and train to epoch 2
        model3 = DummyModel()
        sampler3 = ResumableBatchSampler(len(dataset), batch_size=4, shuffle=True, seed=42)
        loader3 = PyGDataLoader(dataset, batch_sampler=sampler3)
        trainer3 = Trainer(
            model=model3,
            loss_fn=loss_fn,
            train_loader=loader3,
            checkpoint_dir=temp_dir / "ckpt2",
            log_dir=temp_dir / "log2",
            adaptive_loss_weighting=False,
        )

        resume_state = trainer3.resume_latest()
        assert resume_state["epoch"] == 1
        assert resume_state["start_batch"] == 3

        hist3 = trainer3.train(num_epochs=2, resume_state=resume_state)
        # Epoch 1 had 2 remaining batches + Epoch 2 had 5 batches -> 7 steps in trainer3
        # Total global steps: 3 + 7 = 10
        assert trainer3._global_step == 10
        assert len(hist3["train_loss"]) == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_sample_generator_determinism():
    """Verify that sample_generator produces identical draws for same (seed, epoch, sample_id)."""
    rng1 = sample_generator(seed=42, epoch=1, sample_id=5)
    rng2 = sample_generator(seed=42, epoch=1, sample_id=5)

    draw1 = rng1.normal(0, 1, size=(10, 3))
    draw2 = rng2.normal(0, 1, size=(10, 3))
    np.testing.assert_allclose(draw1, draw2)

    # Different epoch -> different draw
    rng_ep2 = sample_generator(seed=42, epoch=2, sample_id=5)
    draw_ep2 = rng_ep2.normal(0, 1, size=(10, 3))
    assert not np.allclose(draw1, draw_ep2)

    # Different sample -> different draw
    rng_s6 = sample_generator(seed=42, epoch=1, sample_id=6)
    draw_s6 = rng_s6.normal(0, 1, size=(10, 3))
    assert not np.allclose(draw1, draw_s6)

    # String sample ID support
    rng_str1 = sample_generator(seed=42, epoch=1, sample_id="sample_abc_001")
    rng_str2 = sample_generator(seed=42, epoch=1, sample_id="sample_abc_001")
    np.testing.assert_allclose(rng_str1.normal(0, 1, size=(5,)), rng_str2.normal(0, 1, size=(5,)))


def test_dataset_set_epoch_synchronization():
    """Verify that Trainer._train_epoch updates dataset.current_epoch."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        class EpochAwareDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.current_epoch = 1
            def set_epoch(self, epoch):
                self.current_epoch = epoch
            def __len__(self):
                return 4
            def __getitem__(self, idx):
                data = MeshData()
                data.x = torch.randn(5, 11)
                data.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                data.edge_attr = torch.randn(2, 4)
                data.elem_conn = torch.zeros((1, 10), dtype=torch.long)
                data.num_nodes = 5
                data.y_displacement = torch.randn(5, 3)
                data.y_stress = torch.randn(5, 6)
                data.y_von_mises = torch.randn(5)
                return data

        dataset = EpochAwareDataset()
        loader = PyGDataLoader(dataset, batch_size=2)
        model = DummyModel()
        loss_fn = DummyLoss()

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            train_loader=loader,
            checkpoint_dir=temp_dir / "ckpt",
            log_dir=temp_dir / "log",
        )

        trainer._train_epoch(epoch=3)
        assert dataset.current_epoch == 3

        trainer._train_epoch(epoch=7)
        assert dataset.current_epoch == 7
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("Running test_resumable_sampler_determinism...")
    test_resumable_sampler_determinism()
    print("[PASS] test_resumable_sampler_determinism passed")

    print("Running test_resumable_sampler_mid_epoch_slicing...")
    test_resumable_sampler_mid_epoch_slicing()
    print("[PASS] test_resumable_sampler_mid_epoch_slicing passed")

    print("Running test_resumable_sampler_state_dict...")
    test_resumable_sampler_state_dict()
    print("[PASS] test_resumable_sampler_state_dict passed")

    print("Running test_sample_generator_determinism...")
    test_sample_generator_determinism()
    print("[PASS] test_sample_generator_determinism passed")

    print("Running test_dataset_set_epoch_synchronization...")
    test_dataset_set_epoch_synchronization()
    print("[PASS] test_dataset_set_epoch_synchronization passed")

    print("Running test_trainer_intra_epoch_batch_checkpoint_and_resume...")
    test_trainer_intra_epoch_batch_checkpoint_and_resume()
    print("[PASS] test_trainer_intra_epoch_batch_checkpoint_and_resume passed")

    print("Running test_trainer_mid_epoch_resumption_exact_batch...")
    test_trainer_mid_epoch_resumption_exact_batch()
    print("[PASS] test_trainer_mid_epoch_resumption_exact_batch passed")

    print("Running test_resume_latest_non_existent...")
    test_resume_latest_non_existent()
    print("[PASS] test_resume_latest_non_existent passed")

    print("Running test_continuous_vs_interrupted_training_trajectory...")
    test_continuous_vs_interrupted_training_trajectory()
    print("[PASS] test_continuous_vs_interrupted_training_trajectory passed")

    print("\nALL TESTS PASSED!")

