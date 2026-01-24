"""
Snapshot and recovery utilities.

Responsible for saving and restoring model and optimizer state
to enable exact rollback.
"""


class SnapshotManager:
    """
    Manages safe-state snapshots for recovery.
    """

    def save(self, model, optimizer):
        raise NotImplementedError

    def restore(self, model, optimizer):
        raise NotImplementedError

