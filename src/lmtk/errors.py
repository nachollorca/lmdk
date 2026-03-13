"""Error registry."""


# TODO: check and learn about vanilla python errors and what is what
class MissingAPIKeyError(RuntimeError):
    def __init__(self, env_var_name: str) -> None:
        super().__init__(f"{env_var_name} is not set.")
