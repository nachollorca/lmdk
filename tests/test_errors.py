"""Tests for lmdk.errors — exception hierarchy and status-code mapping."""

from lmdk.errors import (
    STATUS_TO_ERROR,
    AllModelsFailedError,
    AuthenticationError,
    InternalServerError,
    LMDKError,
    PermissionError,
    ProviderError,
    RateLimitError,
)

# ---------------------------------------------------------------------------
# Inheritance chain
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_provider_error_is_lmdk_error(self):
        assert issubclass(ProviderError, LMDKError)

    def test_authentication_error_is_provider_error(self):
        assert issubclass(AuthenticationError, ProviderError)

    def test_rate_limit_error_is_provider_error(self):
        assert issubclass(RateLimitError, ProviderError)

    def test_all_models_failed_is_lmdk_error(self):
        assert issubclass(AllModelsFailedError, LMDKError)


# ---------------------------------------------------------------------------
# ProviderError attributes
# ---------------------------------------------------------------------------


class TestProviderError:
    def test_attributes(self):
        err = ProviderError(status_code=500, message="boom", provider="TestProv", body='{"e":1}')
        assert err.status_code == 500
        assert err.provider == "TestProv"
        assert err.body == '{"e":1}'
        assert str(err) == "boom"

    def test_defaults(self):
        err = ProviderError(status_code=0, message="oops")
        assert err.provider == ""
        assert err.body == ""

    def test_catchable_as_lmdk_error(self):
        """ProviderError should be catchable with ``except LMDKError``."""
        try:
            raise ProviderError(status_code=400, message="bad request")
        except LMDKError:
            pass  # expected


# ---------------------------------------------------------------------------
# AllModelsFailedError
# ---------------------------------------------------------------------------


class TestAllModelsFailedError:
    def test_summary_message(self):
        errors = {
            "provider_a:model1": ValueError("timeout"),
            "provider_b:model2": RuntimeError("crash"),
        }
        err = AllModelsFailedError(errors)
        assert "provider_a:model1" in str(err)
        assert "provider_b:model2" in str(err)
        assert err.errors is errors

    def test_single_error(self):
        errors = {"p:m": ValueError("fail")}
        err = AllModelsFailedError(errors)
        assert "p:m" in str(err)


# ---------------------------------------------------------------------------
# STATUS_TO_ERROR mapping
# ---------------------------------------------------------------------------


class TestStatusToError:
    def test_401_maps_to_authentication(self):
        assert STATUS_TO_ERROR[401] is AuthenticationError

    def test_403_maps_to_permission(self):
        assert STATUS_TO_ERROR[403] is PermissionError

    def test_429_maps_to_rate_limit(self):
        assert STATUS_TO_ERROR[429] is RateLimitError

    def test_500_maps_to_internal_server_error(self):
        assert STATUS_TO_ERROR[500] is InternalServerError
