class FlightDeployError(RuntimeError):
    """Base class for expected deployment failures."""


class CheckpointError(FlightDeployError):
    """The source checkpoint is missing, damaged, or malformed."""


class UnsupportedCheckpointError(FlightDeployError):
    """No installed adapter can safely convert the checkpoint."""


class BundleIntegrityError(FlightDeployError):
    """A deployment bundle failed schema or checksum validation."""
