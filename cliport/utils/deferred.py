from dataclasses import dataclass, field
from typing import Any, Callable, List


@dataclass
class DeferredExecutor:
    """
    Defer execution of functions.

    Helpful if we have functions that require state of current variables
    but only need execution if certain constraints are satisfied.
    """

    functions: List[Callable[[], Any]] = field(default_factory=list)

    def add(self, func: Callable[..., Any], *args, **kwargs):
        """Add a function to the list of functions to execute."""
        self.functions.append(lambda: func(*args, **kwargs))

    def execute(self) -> None:
        """Execute all functions in the list."""
        for func in self.functions:
            func()
        self.functions = []


deferred = DeferredExecutor()
