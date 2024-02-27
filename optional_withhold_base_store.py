from typing import Callable, Sequence, List, Optional, Union, Iterator, Tuple

from langchain_core.stores import BaseStore, K, V


class OptionalWithholdBaseStore(BaseStore):
    def __init__(self, underlying_base_store: BaseStore, withhold_predicate: Callable[[K, V], bool]) -> None:
        self.underlying_base_store = underlying_base_store
        self.withhold_predicate = withhold_predicate

    def mget(self, keys: Sequence[K], *args, **kwargs) -> List[Optional[V]]:
        return [None if self.withhold_predicate(key, value) else value
                for key, value in zip(keys, self.underlying_base_store.mget(keys, *args, **kwargs))]  # noqa

    def mset(self, key_value_pairs: Sequence[Tuple[K, V]], *args, **kwargs) -> None:
        return self.underlying_base_store.mset(key_value_pairs, *args, **kwargs)  # noqa

    def mdelete(self, keys: Sequence[K], *args, **kwargs) -> None:
        return self.underlying_base_store.mdelete(keys, *args, **kwargs)  # noqa

    def yield_keys(self, *args, prefix: Optional[str] = None, **kwargs) -> Union[Iterator[K], Iterator[str]]:
        return self.underlying_base_store.yield_keys(*args, prefix=prefix, **kwargs)  # noqa

    @classmethod
    def always_withhold(cls, underlying_base_store: BaseStore) -> 'OptionalWithholdBaseStore':
        return OptionalWithholdBaseStore(underlying_base_store, lambda k, v: True)
