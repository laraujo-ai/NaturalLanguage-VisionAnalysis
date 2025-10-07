from typing import Protocol, Any


class BaseRetrieveService(Protocol):
    def retrieve(self, query: Any, top_k: int) -> Any:
        ...
