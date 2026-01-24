"""
Minimal runnable example demonstrating integration of the
Runtime Stability Controller with a PyTorch training loop.

This example intentionally uses a dummy probe and snapshot
manager. It is meant to illustrate usage and API flow, not
the full stability logic.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from runtime_stability_controller.controller import StabilityController
from runtime_stability_controller.probes.base import Probe
from runtime_stability_controller.snapshot import SnapshotManager


# ---------------------------------------------------------------------
# Dummy implementations for demonstration purposes
# ---------------------------------------------------------------------

class DummyProbe(Probe):
    """
    A trivial probe that evaluates model state using
    validation loss on a fixed batch.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.loss_fn = nn.MSELoss()

    def evaluate(self, model):
        model.eval()
        with torch.no_grad():
            pred = model(self.x)
            loss = self.loss_fn(pred, self.y)
        model.train()
        return float(loss.item())


class DummySnapshotManager(SnapshotManager):
    """
    Snapshot manager that stores model and optimizer state in memory.
    """

    def __init__(self):
        self._model_state = None
        self._optimizer_state = None

    def save(self, model, optimizer):
        self._model_state = {k: v.clone() for k, v in model.state_dict().items()}
        self._optimizer_state = optimizer.state_dict()

    def restore(self, model, optimizer):
        model.load_state_dict(self._model_state)
        optimizer.load_state_dict(self._optimizer_state)


# ---------------------------------------------------------------------
# Minimal training script
# ---------------------------------------------------------------------

def main():
    torch.manual_seed(0)

    # Simple linear regression task
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    # Training data
    x_train = torch.randn(32, 1)
    y_train = 2.0 * x_train + 0.1 * torch.randn(32, 1)

    # Fixed validation probe data
    x_val = torch.randn(8, 1)
    y_val = 2.0 * x_val

    probe = DummyProbe(x_val, y_val)
    snapshot_manager = DummySnapshotManager()

    controller = StabilityController(
        probe=probe,
        snapshot_manager=snapshot_manager,
        threshold=1.0,
        smoothing=0.1,
    )

    # Initialize controller
    controller.initialize(model, optimizer)

    # Training loop
    for step in range(20):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()

        # Optimizer proposes update
        optimizer.step()

        # Controller supervision (logic not implemented yet)
        accepted = True  # placeholder for controller.step(model, optimizer)

        if accepted:
            snapshot_manager.save(model, optimizer)

        print(f"Step {step:02d} | loss = {loss.item():.4f}")

    print("Training completed.")


if __name__ == "__main__":
    main()
