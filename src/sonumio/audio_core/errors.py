"""Error taxonomy for independent audio-core processing.

Callers may treat these as explicit data-validation errors rather than
conflating malformed user-controlled buffers with model quality failures.
"""

class AudioContractError(ValueError):
    """Invalid audio shape, value, dtype, layout or argument."""

class AudioFormatError(AudioContractError):
    """The requested media format or encoding is unsupported."""

class AudioLimitError(AudioContractError):
    """An operation exceeds bounded memory, duration or channel limits."""

class AudioIntegrityError(AudioContractError):
    """Audio sample data or its declared integrity metadata changed."""

class AudioProcessingError(RuntimeError):
    """A valid buffer could not be processed using the chosen algorithm."""

class AudioDependencyError(AudioProcessingError):
    """An explicit optional backend is not installed or is unavailable."""
