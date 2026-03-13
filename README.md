# lmtk

Language Models, as God intended.

What it offers:
- Basic interface to inference different Language Model APIs
- Minimal dependencies: calls are made through REST, not third party packages
- Stateless `get_response` or Stateful `Chat`
- Can stream text or use grammar for structured outputs (but not both)
- Managed config and secrets (API Keys)
- TUI to chat with LLMs from the terminal

What it does NOT offer:
- Tools (function calling)
- Agents or agentic workflows
- The TUI oes not grant access to your file system, terminal, etc.
- Multimodality. Just text in, text out
