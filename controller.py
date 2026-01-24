"""
Runtime stability controller.

Supervises optimizer updates using an external measurement probe and
enforces accept / rollback decisions based on an innovation signal.
"""

from runtime_stability_controller.exceptions import StabilityViolation


class StabilityController:
    """
    Runtime stability controller.

    Workflow:
    - initialize(): measure reference signal and store initial safe snapshot
    - step(): run optimizer.step(), measure, compute innovation, accept or rollback
    """

    def __init__(self, probe, snapshot_manager, threshold, smoothing=None):
        """
        Parameters
        ----------
        probe : Probe
            External measurement probe used to evaluate model state.
        snapshot_manager : SnapshotManager
            Manages saving and restoring safe training states.
        threshold : float
            Acceptance threshold for the innovation signal.
        smoothing : float, optional
            Exponential smoothing factor alpha in (0, 1). If None, the reference
            is replaced by the latest accepted measurement.
        """
        self.probe = probe
        self.snapshot_manager = snapshot_manager
        self.threshold = float(threshold)
        self.smoothing = smoothing

        self._reference_value = None

        # Optional diagnostics (useful for debugging / plotting)
        self.last_measurement = None
        self.last_innovation = None
        self.last_accepted = None

    @property
    def reference_value(self):
        return self._reference_value

    def initialize(self, model, optimizer):
        """
        Initialize the controller state (call once before training).
        """
        value = self.probe.evaluate(model)
        self._reference_value = float(value)
        self.snapshot_manager.save(model, optimizer)

        self.last_measurement = self._reference_value
        self.last_innovation = 0.0
        self.last_accepted = True

    def step(self, model, optimizer):
        """
        Supervised optimizer step.

        Assumes gradients are already computed (loss.backward() has been called).

        Returns
        -------
        bool
            True if accepted, False if rollback occurred.
        """
        if self._reference_value is None:
            raise RuntimeError("Controller is not initialized. Call initialize(model, optimizer) first.")

        # Save current safe state *before* applying the proposed update
        # (this enables exact rollback if rejected).
        self.snapshot_manager.save(model, optimizer)

        # Optimizer proposes and applies update
        optimizer.step()

        # Measure after applying the proposed update (candidate state)
        measurement = float(self.probe.evaluate(model))
        innovation = measurement - float(self._reference_value)

        self.last_measurement = measurement
        self.last_innovation = innovation

        # Decision rule (Algorithm 1 style)
        if innovation <= self.threshold:
            # Accept update: update reference signal
            self._update_reference(measurement)
            self.last_accepted = True
            return True

        # Reject: rollback to the last safe snapshot
        self.snapshot_manager.restore(model, optimizer)
        self.last_accepted = False

        # Optional: raise, or just return False. For now, return False.
        # Raising can be enabled later via a policy flag.
        return False

    def _update_reference(self, value):
        """
        Update reference signal after an accepted step.
        """
        if self.smoothing is None:
            self._reference_value = float(value)
            return

        alpha = float(self.smoothing)
        if not (0.0 < alpha < 1.0):
            raise ValueError("smoothing must be in (0, 1)")

        self._reference_value = (1.0 - alpha) * float(self._reference_value) + alpha * float(value)
