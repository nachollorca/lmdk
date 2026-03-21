"""Custom exceptions for lmdk."""


class LMDKError(Exception):
    """Base exception for all lmdk errors."""


class ProviderError(LMDKError):
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


class BadRequestError(ProviderError):
    """Raised for 400 responses -- invalid request."""


class AuthenticationError(ProviderError):
    """Raised for 401 -- missing or incorrect API credentials."""


class BillingError(ProviderError):
    """Raised for 402 responses -- billing issue or payment required."""


class PermissionError(ProviderError):
    """Raised for 403 -- credential is correct but lacks permission."""


class NotFoundError(ProviderError):
    """Raised for 404 responses -- resource not found."""


class RequestTooLargeError(ProviderError):
    """Raised for 413 responses -- request payload too large."""


class RateLimitError(ProviderError):
    """Raised for 429 responses -- too many requests."""


class InternalServerError(ProviderError):
    """Raised for 500 responses -- internal server error."""


class ServiceUnavailableError(ProviderError):
    """Raised for 503 responses -- service overloaded or unavailable."""


class AllModelsFailedError(LMDKError):
    """Raised when every model in a fallback list fails.

    Attributes:
        errors: Mapping of model identifier to the exception it raised.
    """

    def __init__(self, errors: dict[str, Exception]):
        self.errors = errors
        summary = "; ".join(f"{m}: {e}" for m, e in errors.items())
        super().__init__(f"All models failed: {summary}")


STATUS_TO_ERROR: dict[int, type[ProviderError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    402: BillingError,
    403: PermissionError,
    404: NotFoundError,
    413: RequestTooLargeError,
    429: RateLimitError,
    500: InternalServerError,
    503: ServiceUnavailableError,
}
