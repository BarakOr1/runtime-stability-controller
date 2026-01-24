"""
Minimal runnable example demonstrating integration of the
Runtime Stability Controller with a PyTorch training loop.

This example illustrates how optimizer updates are supervised
at runtime using an external validation-based measurement probe.
The focus is on system integration and API flow rather than
performance or advanced recovery policies.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from runtime_stability_controller.controller import StabilityController
from runtime_stability_controller.probes.base import Probe
from runtime_stability_controller.snapshot import InMemorySnapshotManager


# ---------------------------------------------------------------------
# Validation-based measurement probe
# ---------------------------------------------------------------------

class ValidationLossProbe(Probe):
    """
    Measurement probe that evaluates model state using
    validation loss on a fixed, held-out batch.
    """

    def __init__(self, x_val, y_val):
        self.x_val = x_val
        self.y_val = y_val
        self.loss_fn = nn.MSELoss()

    def evaluate(self, model):
        model.eval()
        with torch.no_grad():
            pred = model(self.x_val)
            loss = self.loss_fn(pred, self.y_val)
        model.train()
        return float(loss.item())


# ---------------------------------------------------------------------
# Minimal training script
# ---------------------------------------------------------------------

def main():
    torch.manual_seed(0)

    # Simple linear regression model
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    # Training data
    x_train = torch.randn(32, 1)
    y_train = 2.0 * x_train + 0.1 * torch.randn(32, 1)

    # Fixed validation probe data
    x_val = torch.randn(8, 1)
    y_val = 2.0 * x_val

    # Runtime stability components
    probe = ValidationLossProbe(x_val, y_val)
    snapshot_manager = InMemorySnapshotManager()

    controller = StabilityController(
        probe=probe,
        snapshot_manager=snapshot_manager,
        threshold=0.5,     # conservative threshold for demonstration
        smoothing=0.1,
    )

    # Initialize controller (reference signal + initial safe snapshot)
    controller.initialize(model, optimizer)

    # Training loop with supervised optimizer steps
    for step in range(20):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()

        accepted = controller.step(model, optimizer)

        print(
            f"Step {step:02d} | "
            f"train_loss={loss.item():.4f} | "
            f"probe={controller.last_measurement:.4f} | "
            f"innovation={controller.last_innovation:.4f} | "
            f"accepted={accepted}"
        )

    print("Training completed.")


if __name__ == "__main__":
    main()
