from typing import Any, Optional, Hashable, TypeVar, Generic, LiteralString


class ReturnUpdates:
    def __init__(
        self, next_fn: Optional[LiteralString], *remove: LiteralString, **replace: Any
    ):
        self.remove = remove
        self.replace = replace
        self.next_fn = next_fn


CondVal = TypeVar("CondVal", bound=Hashable)


class ConditionalReturns(Generic[CondVal]):
    def __init__(
        self,
        /,
        operand: CondVal,
        default: ReturnUpdates,
        *,
        true: Optional[ReturnUpdates] = None,
        **value_retn: ReturnUpdates,
    ):
        self.op = operand
        self.value_retn = value_retn
        self.true_retn = true
        self.default_retn = default

    def evaluate(self):
        if (self.op in self.value_retn) and isinstance(self.op, str):
            return self.value_retn[self.op]
        if (self.true_retn is not None) and self.op:
            return self.true_retn
        return self.default_retn
