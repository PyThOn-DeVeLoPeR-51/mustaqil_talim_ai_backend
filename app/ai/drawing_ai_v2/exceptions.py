"""Domain exceptions for Drawing AI evaluation."""


class DrawingAIError(RuntimeError):
    """Base exception for Drawing AI failures."""


class DrawingAIValidationError(DrawingAIError):
    """Raised when an input drawing is missing, malformed, or unsupported."""


class DrawingAIConfigurationError(DrawingAIError):
    """Raised when an evaluation request is internally inconsistent."""


class DrawingAIResultError(DrawingAIError):
    """Raised when an evaluator returns an invalid result contract."""
