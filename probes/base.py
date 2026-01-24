"""
Base interface for measurement probes.
"""


class Probe:
    """
    Abstract measurement probe.

    A probe evaluates a proposed model state using information
    external to the training objective.
    """

    def evaluate(self, model):
        """
        Evaluate the proposed model state.

        Returns
        -------
        float
            Scalar measurement value.
        """
        raise NotImplementedError

