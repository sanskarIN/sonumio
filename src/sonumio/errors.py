"""Typed user-facing generation errors."""

class GenerationError(ValueError):
    """Invalid request or unavailable backend; safe to expose to API clients."""

class UnsupportedPromptError(GenerationError):
    """No implemented engine can fulfill the prompt."""

class UnavailableModelError(GenerationError):
    """The requested optional model is not enabled or installed."""
