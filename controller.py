"""
Runtime stability controller.

Defines the main supervisory logic that evaluates optimizer-proposed
updates using external measurement signals and enforces accept / rollback
decisions at runtime.
"""


class StabilityController:
    """
    Runtime stability controller.

    The controller operates as an external supervisory layer that
    monitors optimizer-proposed updates and intervenes only when
    destabilizing behavior is detected.
    """

    def __init__(self, probe, snapshot_manager, threshold, smoothing=None):
        """
        Parameters
        ----------
        probe : Probe
            External measurement probe used to evaluate proposed updates.
        snapshot_manager : SnapshotManager
            Manages saving and restoring safe training states.
        threshold : float
            Acceptance threshold for the innovation signal.
        smoothing : float, optional
            Optional smoothing factor for reference signal tracking.
        """
        self.probe = probe
        self.snapshot_manager = snapshot_manager
        self.threshold = threshold
        self.smoothing = smoothing

        self._reference_value = None

    def initialize(self, model, optimizer):
        """
        Initialize the controller state.

        Must be called once before training begins.
        """
        value = self.probe.evaluate(model)
        self._reference_value = value
        self.snapshot_manager.save(model, optimizer)

    def step(self, model, optimizer):
        """
        Supervised optimizer step.

        This method evaluates the effect of an optimizer-proposed update
        and decides whether to accept it or rollback to the last safe state.

        Returns
        -------
        bool
            True if the update was accepted, False if rollback was triggered.
        """
        raise NotImplementedError

    def _update_reference(self, value):
        """
        Update the reference signal after an accepted step.
        """
        if self.smoothing is None:
            self._reference_value = value
        else:
            alpha = self.smoothing
            self._reference_value = (1 - alpha) * self._reference_value + alpha * value
