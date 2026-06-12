"""Tests for lmdk.utils — utility helpers."""

import time

import pytest

from lmdk.utils import parallelize_function, render_template, return_if_exception

# ---------------------------------------------------------------------------
# return_if_exception
# ---------------------------------------------------------------------------


class TestReturnIfException:
    def test_returns_value_on_success(self):
        @return_if_exception
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_returns_exception_on_failure(self):
        @return_if_exception
        def fail():
            raise ValueError("boom")

        result = fail()
        assert isinstance(result, ValueError)
        assert str(result) == "boom"

    def test_preserves_function_name(self):
        @return_if_exception
        def my_func():
            pass

        assert my_func.__name__ == "my_func"


# ---------------------------------------------------------------------------
# parallelize_function
# ---------------------------------------------------------------------------


class TestParallelizeFunction:
    def test_preserves_order(self):
        def double(x):
            return x * 2

        params = [{"x": 1}, {"x": 2}, {"x": 3}]
        results = parallelize_function(double, params)
        assert results == [2, 4, 6]

    def test_empty_params_list(self):
        results = parallelize_function(lambda: None, [])
        assert results == []

    def test_catch_exceptions_returns_errors(self):
        def maybe_fail(x):
            if x < 0:
                raise ValueError(f"negative: {x}")
            return x

        params = [{"x": 1}, {"x": -1}, {"x": 2}]
        results = parallelize_function(maybe_fail, params, catch_exceptions=True)

        assert results[0] == 1
        assert isinstance(results[1], ValueError)
        assert results[2] == 2

    def test_without_catch_exceptions_raises(self):
        def fail(x):
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            parallelize_function(fail, [{"x": 1}], catch_exceptions=False)

    def test_actually_runs_in_parallel(self):
        def slow(x):
            time.sleep(0.1)
            return x

        params = [{"x": i} for i in range(5)]
        start = time.monotonic()
        results = parallelize_function(slow, params, max_workers=5)
        elapsed = time.monotonic() - start

        assert results == list(range(5))
        # 5 tasks sleeping 0.1s each, with 5 workers should take ~0.1s not ~0.5s
        assert elapsed < 0.3


# ---------------------------------------------------------------------------
# render_template
# ---------------------------------------------------------------------------


def test_render_template_string():
    template = "Hello {{ name }}!"
    result = render_template(template=template, name="World")
    assert result == "Hello World!"


def test_render_template_path(tmp_path):
    template_file = tmp_path / "template.jinja"
    template_file.write_text("Hello {{ name }} from file!")

    result = render_template(path=template_file, name="World")
    assert result == "Hello World from file!"


def test_render_template_path_string(tmp_path):
    template_file = tmp_path / "template.jinja"
    template_file.write_text("Hello {{ name }}!")

    result = render_template(path=str(template_file), name="World")
    assert result == "Hello World!"


def test_render_template_no_args():
    with pytest.raises(ValueError, match="Must provide either 'template' or 'path'"):
        render_template()


def test_render_template_both_args():
    with pytest.raises(ValueError, match="Provide either 'template' or 'path', not both"):
        render_template(template="foo", path="bar.jinja")


def test_render_template_cleaning():
    # Test existing cleaning logic
    template = "Value: {{ val }}"
    result = render_template(template=template, val="{{ dirty }}")
    assert result == "Value: dirty"


def test_render_template_preserves_curly_brackets_when_disabled():
    template = "History:\n{{ HISTORY }}"
    inner = "<template>\nReview: {{ REVIEW }}\n</template>"
    result = render_template(
        template=template,
        HISTORY=inner,
        strip_curly_brackets=False,
    )
    assert result == "History:\n<template>\nReview: {{ REVIEW }}\n</template>"


def test_render_template_still_strips_whitespace_when_curly_brackets_preserved():
    template = "Value: {{ val }}"
    result = render_template(
        template=template,
        val="  {{ nested }}  ",
        strip_curly_brackets=False,
    )
    assert result == "Value: {{ nested }}"
