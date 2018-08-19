import contextlib
from typing import TypeVar


class Config:
    def __init__(self):
        self.stack = [{}]

    @property
    def current(self) -> dict:
        return self.stack[-1]


CONFIG = Config()
T = TypeVar('T')


def get(key: str, default: T = None) -> T:
    if key not in CONFIG.current:
        CONFIG.current[key] = default
    return CONFIG.current[key]


@contextlib.contextmanager
def use(config: dict) -> dict:
    CONFIG.stack.append(config)
    try:
        yield CONFIG.current
    finally:
        CONFIG.stack.pop()


def current() -> dict:
    return CONFIG.current
