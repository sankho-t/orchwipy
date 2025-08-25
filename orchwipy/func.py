import dataclasses
import functools
import inspect
import json
import logging
import os
import traceback
import uuid
from typing import Callable, Optional, Any, Protocol
from pathlib import Path

import yaml

from .plot_graph import PlotGraph
from .types import ReturnUpdates, ConditionalReturns

log = logging.getLogger(__name__)

OrchwipyMaxStack = int(os.getenv("OrchwipyMaxStack", 25))


class SupportsOrchwipyFn(Protocol):
    __name__: str

    def __call__(self, **kwds) -> ConditionalReturns | ReturnUpdates | dict: ...


@dataclasses.dataclass
class Functions(PlotGraph):
    """Base class for the pipelined functions."""

    _: dataclasses.KW_ONLY
    beginner: str
    """Function name of the first function to be called"""

    functions: dict[str, Callable[[dict], dict]] = dataclasses.field(
        default_factory=dict
    )
    default_next: dict[str, str] = dataclasses.field(default_factory=dict)
    debugging: bool = dataclasses.field(default=False)
    """Enabling this flag will store the currently executing function's arguments
    in `debug_args_fname` pickle"""

    debug_args_fname: str = dataclasses.field(default="orchwipy_pipeline_last_args.pk")

    log_args: Optional[Path] = dataclasses.field(default=None)
    running_name: str | None = dataclasses.field(default=None)

    def __post_init__(self):
        self.synthetic_args.append("_original_input")

    def _goto_next(self, *, data: dict, caller: str, force: Optional[str] = None):
        stack = data["$stack"] = [*data.get("$stack", []), caller]
        if len(stack) > OrchwipyMaxStack:
            log.error("Exceeded call stack size %d", len(stack))
            return

        if not force:
            to = self.default_next.get(caller)
            if to:
                return self.do_call(dest=to, data=data)
        else:
            return self.do_call(dest=force, data=data)

    def _register(self, *, handler: Callable[[dict], dict], name: str, **kwargs):
        self.functions[name] = handler

    def prologue(self, *, dest: str, data: dict):
        return {**data, "$stack": data.get("$stack", [])}

    def do_call(self, *, dest: str, data: dict):
        data_in = self.prologue(dest=dest, data=data)
        return self.functions[dest](data_in)

    def always_on_exit(self, *, input_data: dict):
        pass

    def epilogue(self, *, dest: str, input_data: dict, output_data: dict):
        output_data["$original_input"] = input_data.get("$original_input", input_data)

    def special_keys(self):
        yield "$stack"
        yield "$original_input"

    def overwrite_keys(self):
        yield "$stack"

    def save_args(self, event: dict, fname: str):
        if self.log_args:
            with open(self.log_args, "a") as log_out:
                yaml.safe_dump(data={**event, "$fname": fname}, stream=log_out)
                log_out.write("---\n")

    def new_request_id(self, data: dict):
        return str(uuid.uuid4())

    def call_wrap(self, fn: SupportsOrchwipyFn, **args):
        self.running_name = fn.__name__
        try:
            return fn(**args)
        finally:
            self.running_name = None

    def fn(
        self,
        default_next: Optional[str] = None,
        next_key: str = "$next",
        terminates=False,
        **kwargs,
    ):
        def decorator_begin(func: SupportsOrchwipyFn):
            ts = traceback.extract_stack(None)[-2]
            assert isinstance(ts.lineno, int)

            nbranches = self.new_function(
                lineno=ts.lineno,
                default_next=default_next,
                next_key=next_key,
                fname=ts.filename,
                may_terminate=terminates,
            )

            log.debug("Register Function %s, branches: %d", func.__name__, nbranches)

            @functools.wraps(func)
            def handler(event: dict, context=None):

                req_id = self.new_request_id(event)
                if "$request_id" not in event:
                    log.debug("New request id", extra=dict(req_id=req_id))
                    event["$request_id"] = req_id

                self.save_args(event=event, fname=func.__name__)

                log.debug(
                    "Call Function %s",
                    func.__name__,
                    extra=dict(eventKeys=list(event.keys())),
                )
                event["$original_input"] = event.get("$original_input", dict(event))
                hidden = {
                    k: v for k, v in event.items() if k not in list(self.special_keys())
                }
                args = dict(**hidden)
                if ("_original_input" in inspect.getfullargspec(func).args) or (
                    "_original_input" in inspect.getfullargspec(func).kwonlyargs
                ):
                    args["_original_input"] = event["$original_input"]

                func_args = inspect.getfullargspec(func)
                callback_args: list[str] = []

                for callback in (
                    fname
                    for fname in [*func_args.args, *func_args.kwonlyargs]
                    if fname.startswith("callback_")
                ):
                    args[callback] = functools.partial(
                        self.__getattribute__(callback),
                        _original_input=event["$original_input"],
                    )
                    callback_args.append(callback)

                if self.debugging:
                    import pickle

                    args["$fname"] = func.__name__
                    with open(self.debug_args_fname, "wb") as f:
                        try:
                            pickle.dump(obj=args, file=f)
                        except BaseException:
                            log.warning("Cannot save function arguments", exc_info=True)

                try:
                    out_or_cond: dict[str, Any] | ReturnUpdates | ConditionalReturns = (
                        self.call_wrap(func, **args)
                    )
                except PauseExecution:
                    log.debug("Pausing execution")
                except Exception:
                    args_no_cb = {
                        k: v for k, v in args.items() if not k.startswith("callback_")
                    }

                    log.error("Failed", exc_info=True)
                    try:
                        self.pipeline_fail(input_data=event)
                    except Exception:
                        log.error("Pipeline fail exception", exc_info=True)
                    raise
                else:
                    next_fn: str | None = None
                    out = (
                        out_or_cond.evaluate()
                        if isinstance(out_or_cond, ConditionalReturns)
                        else out_or_cond
                    )

                    if isinstance(out, ReturnUpdates):
                        if "*" in out.remove:
                            args.clear()
                        else:
                            for k in out.remove:
                                if k in args:
                                    del args[k]
                        args.update(out.replace)
                        next_fn = out.next_fn
                        out = {k: v for k, v in args.items() if not k in callback_args}

                    for k in self.overwrite_keys():
                        out[k] = event[k]

                    out[next_key or "$next"] = next_fn or out.get(
                        next_key, self.default_next.get(func.__name__)
                    )

                    self.epilogue(dest=func.__name__, input_data=event, output_data=out)

                    if next_key:
                        if next_fn := out.get(next_key):
                            del out[next_key]
                            return self._goto_next(
                                data=out, caller=func.__name__, force=str(next_fn)
                            )

                    return self._goto_next(data=out, caller=func.__name__)

                finally:
                    self.always_on_exit(input_data=event)

            self._register(name=func.__name__, handler=handler, **kwargs)

            if default_next:
                self.default_next[func.__name__] = default_next
            return handler

        return decorator_begin

    def run(self, **data: Any):
        if self.log_args:
            self.log_args.write_text("")

        return self._goto_next(
            data={**data, "$stack": [], "$original_input": data},
            caller="",
            force=self.beginner,
        )

    def run_from(self, *, step: str, **data: Any):
        return self._goto_next(
            data={**data, "$stack": [], "$original_input": data},
            caller="",
            force=step,
        )

    def resume(self, fname: Path):
        data = json.loads(fname.read_text())

        return self._goto_next(
            data=data, caller=data["$seqnName"], force=data.get("$next")
        )

    def run_from_last_fail(self):
        import pickle

        with open(self.debug_args_fname, "rb") as f:
            args = pickle.load(file=f)
            step = args["$fname"]
            del args["$fname"]

            return self.run_from(step=step, **args)

    def pipeline_fail(self, input_data: dict):
        pass

    def set_log_level(self, level: int):
        log.setLevel(level=level)


class PauseExecution(BaseException):
    pass


class JSONEncodeTruncated(json.JSONEncoder):
    def __init__(
        self,
        *,
        skipkeys: bool = False,
        ensure_ascii: bool = True,
        check_circular: bool = True,
        allow_nan: bool = True,
        sort_keys: bool = False,
        indent: int | str | None = None,
        separators: tuple[str, str] | None = None,
        default: Callable[..., Any] | None = None,
    ) -> None:
        self.string_truncate_length = 20

        super().__init__(
            skipkeys=skipkeys,
            ensure_ascii=ensure_ascii,
            check_circular=check_circular,
            allow_nan=allow_nan,
            sort_keys=sort_keys,
            indent=indent,
            separators=separators,
            default=default,
        )

    def default(self, o: Any) -> Any:
        if isinstance(o, str):
            o2 = o[: self.string_truncate_length]
            if not o == o2:
                return f"{o2}...(+{len(o) - len(o2)})"
            return o
        return super().default(o)
