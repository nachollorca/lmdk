"""Custom exceptions for lmtk."""


class LMTKError(Exception):
    """Base exception for all lmtk errors."""


class ProviderError(LMTKError):
    """Raised when a provider API call fails.

    Attributes:
        status_code: The HTTP status code (0 for local/pre-request errors).
        provider: Name of the provider that raised the error.
        body: Raw response body from the API, if available.
    """

    def __init__(self, status_code: int, message: str, *, provider: str = "", body: str = ""):
        """Initialize with HTTP status code, message, provider name, and optional response body."""
        self.status_code = status_code
        self.provider = provider
        self.body = body
        super().__init__(message)


class AuthenticationError(ProviderError):
    """Raised for 401/403 responses or missing API credentials."""


class RateLimitError(ProviderError):
    """Raised for 429 responses -- too many requests."""


STATUS_TO_ERROR: dict[int, type[ProviderError]] = {
    401: AuthenticationError,
    403: AuthenticationError,
    429: RateLimitError,
}
