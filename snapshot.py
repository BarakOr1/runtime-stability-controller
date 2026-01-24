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
        """
        Save a safe snapshot of the current training state.
        """
        raise NotImplementedError

    def restore(self, model, optimizer):
        """
        Restore the most recent safe snapshot.
        """
        raise NotImplementedError
