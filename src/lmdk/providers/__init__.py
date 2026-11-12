"""LLM provider implementations.

- ``shapes/``: wire formats only (``chat``, ``responses``, ``messages``,
  ``systemone``), always addressed via ``@<address>``.
- ``vendors/``: hosted defaults on top of a shape (``openai``, ``anthropic``,
  ``mistral``, ...), where ``@`` is optional and overrides the default location.
"""
