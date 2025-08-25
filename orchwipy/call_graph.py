import ast
import json
import dataclasses
import logging
import warnings
from collections import defaultdict, deque
from functools import cached_property, cache
from itertools import groupby, chain
from pathlib import Path
from pprint import pp
from typing import Any, Iterable, Optional, Literal

from .types import ReturnUpdates, ConditionalReturns

log = logging.getLogger(__name__)


@dataclasses.dataclass
class GraphDynData:
    _: dataclasses.KW_ONLY
    to_next_dyn: defaultdict[
        str,
        list[
            tuple[
                str,
                tuple[str, ...],
                tuple[str, ...],
                Optional[tuple[int, str, str | bool]],
            ]
        ],
    ] = dataclasses.field(default_factory=lambda: defaultdict(list))
    requires: dict[str, tuple[tuple[str, bool], ...]] = dataclasses.field(
        default_factory=dict
    )
    synthetic_args: list[str] = dataclasses.field(default_factory=list)
    node_props: dict[str, dict] = dataclasses.field(default_factory=dict)

    def init_dyn_nodes(self):
        for fro, to_ in self.to_next_dyn.items():
            for to, rem, repl, cond in to_:
                if "*" in rem:
                    yield (fro, cond, to, repl)

    def explore_dyn_nodes(self):
        explored: set[str] = set()
        edge_attrib: dict[str, Any] = {}

        explore_now = deque(
            [
                (fro, cond, to, edge_attrib)
                for (fro, cond, to, repl) in self.init_dyn_nodes()
            ][:0]
        )

        def explore_fn(from_next: str):
            cond_compress: dict[int, str | None | bool | Literal[False]] = {}
            for to, rem, repl, cond in self.to_next_dyn.get(from_next, []):
                if not cond:
                    explore_now.append((from_next, cond, to, {}))
                else:
                    ln, name, target = cond
                    if not target == "*":
                        if cond_compress.get(ln) == None:
                            cond_compress[ln] = target
                        elif cond_compress.get(ln):
                            cond_compress[ln] = False

            for to, rem, repl, cond in self.to_next_dyn.get(from_next, []):
                if cond:
                    ln, name, target = cond
                    if t2 := cond_compress.get(ln):
                        if target != "*":
                            explore_now.append((from_next, cond, to, {}))
                        else:
                            explore_now.append(
                                (from_next, (ln, name, t2), to, dict(style="dashed"))
                            )
                    else:
                        explore_now.append((from_next, cond, to, {}))

        for fn in iter(set(fro for fro, _, _, _ in self.init_dyn_nodes())):
            explore_fn(from_next=fn)

        cond_node_names: dict[tuple[int, str, str], int] = {}

        while explore_now:
            exp = explore_now.popleft()
            repexp = f"{exp[0]} {exp[1]} {exp[2]}"
            if repexp in explored:
                continue

            explored.add(repexp)
            fro, cond, to, edge_attrib = exp
            if cond:
                cond_node_index = cond_node_names.get(cond, len(cond_node_names) + 1)
                cond_node = f"cond{cond_node_index}"
                self.node_props[cond_node] = dict(
                    shape="box",
                    style="filled",
                    line=cond[0],
                    label=self.repr_cond(*cond),
                    operand=cond[1],
                    match=cond[2],
                )
                yield fro, cond_node, {}
                yield cond_node, to, edge_attrib
                cond_node_names[cond] = cond_node_index
            else:
                yield fro, to, {}

            explore_fn(to)

    @staticmethod
    def repr_cond(ln: int, op: str, target: str | bool):
        if isinstance(target, bool):
            return ("" if target else "!") + f"{op} ?"
        return f"{op} = {target} ?"

    def check_graph_at(self, /, prev_callers: list[str], fn: str, *attributes: str):
        require_default = dict(self.requires[fn])
        missing = set(require_default.keys()).difference(attributes)
        optional = set([att for att, opt in require_default.items() if opt])
        if not prev_callers:
            missing.clear()
        if not_passing := missing.difference(optional):
            warnings.warn(
                "%s : caller history %s -> %s" % (str(not_passing), prev_callers, fn),
                MissingArgument,
            )
        if default_passing := list(missing.intersection(optional)):
            warnings.warn(
                "%s : caller history %s -> %s"
                % (str(default_passing), prev_callers, fn),
                DefaultArgument,
            )
        ok = not bool(not_passing)

        for to, rem, repl, _ in self.to_next_dyn.get(fn, []):
            if "*" in rem:
                passing = set(repl)
            else:
                passing = set(attributes).difference(rem).union(repl)
            if to in [*prev_callers, fn]:
                warnings.warn("Infinite recursion possible due to %s <- %s" % (to, fn))
                this_ok = True
            else:
                this_ok = self.check_graph_at([*prev_callers, fn], to, *passing)
            ok = ok and this_ok
        return ok

    def check_graph(self, *start_from: str, warn_default_passing: bool = False):

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("once")

            for caller in start_from:
                self.check_graph_at([], caller)

        have_missing = False

        for wx in w:
            if issubclass(wx.category, MissingArgument) and warn_default_passing:
                have_missing = True
                warnings.showwarning(
                    category=wx.category,
                    message=wx.message,
                    lineno=0,
                    filename="",
                    line="",
                )
        if have_missing:
            return False

        for wx in w:
            warnings.showwarning(
                category=wx.category,
                message=wx.message,
                lineno=0,
                filename="",
                line="",
            )

        return True

    def edges_from_dyn(self):
        for fro, to, ea in self.explore_dyn_nodes():
            yield (fro, to, ea)


@dataclasses.dataclass
class CallGraph(GraphDynData):
    _: dataclasses.KW_ONLY

    terminal: list[str] = dataclasses.field(default_factory=list)

    def new_function_from_code(
        self,
        *,
        code: ast.Module,
        lineno: int,
        default_next: str | None,
        next_key: str,
        may_terminate=False,
    ):

        n_branches = 0
        finder = FindFn(orchwipy_call_line=lineno)
        finder.visit(code)

        assert finder.found, "Cannot find function in line %d" % lineno

        fn_parse = ParseFunction(line=lineno, next_key_in_dict=next_key)
        fn_parse.visit(finder.found)

        assert (
            fn_parse.return_dicts
            or fn_parse.return_updates
            or fn_parse.return_cond_updates
        ), ("No return statements for function in line %d" % lineno)

        req: list[tuple[str, bool]] = []

        for argname, argrhs in fn_parse.args.items():
            req.append((argname, len(argrhs) > 1))

        this_fn = str(finder.found.name)
        self.requires[this_fn] = tuple(
            x for x in req if not x[0] in self.synthetic_args
        )

        for line, (next_fn, drop, update) in fn_parse.return_updates.items():
            next_fn = next_fn or default_next
            if not next_fn:
                if not may_terminate:
                    raise ValueError("Cannot determine next function (line: %d)" % line)
            else:
                self.to_next_dyn[this_fn].append((next_fn, drop, update, None))
                n_branches += 1

        for line, (cond, def_ret, cond_ret) in fn_parse.return_cond_updates.items():
            for op, target in chain(cond_ret.items(), [(None, def_ret)]):
                cond_name = (
                    "*" if op is None else (op if isinstance(op, bool) else f"'{op}'")
                )
                (next_fn, drop, update) = target
                next_fn = next_fn or default_next
                cond_repr = (line, ast.unparse(cond), cond_name)
                if not next_fn:
                    if not may_terminate:
                        raise ValueError(
                            "Cannot determine next function (line: %d)" % line
                        )
                else:
                    self.to_next_dyn[this_fn].append((next_fn, drop, update, cond_repr))
                    n_branches += 1

        if not n_branches and not may_terminate:
            raise ValueError("No valid returns")

        if may_terminate:
            self.terminal.append(this_fn)
        return n_branches

    def new_function(
        self,
        *,
        fname: str,
        lineno: int,
        default_next: str | None,
        next_key: str,
        may_terminate=False,
    ):

        code = ast.parse(
            source=open(fname, "r").read(), filename=fname, type_comments=True
        )
        try:
            return self.new_function_from_code(
                code=code,
                lineno=lineno,
                default_next=default_next,
                next_key=next_key,
                may_terminate=may_terminate,
            )
        except AssertionError as exc:
            arg0 = f"Error building call graph from {fname}: {exc.args[0]}"
            raise AssertionError(arg0, *exc.args[1:])


@dataclasses.dataclass
class FindFn(ast.NodeVisitor):
    orchwipy_call_line: int
    found: ast.FunctionDef | None = dataclasses.field(default=None)

    def visit_FunctionDef(self, node):
        for dec in node.decorator_list:
            if dec.lineno == self.orchwipy_call_line:
                self.found = node


RetUpdRep = tuple[str | None, tuple[str, ...], tuple[str, ...]]


@dataclasses.dataclass
class ParseFunction(ast.NodeVisitor):
    line: int
    next_key_in_dict: str
    top_fndef: bool = dataclasses.field(default=False)
    return_dicts: dict[int, ast.Dict] = dataclasses.field(default_factory=dict)
    return_updates: dict[int, RetUpdRep] = dataclasses.field(default_factory=dict)
    return_cond_updates: dict[
        int, tuple[ast.expr, RetUpdRep, dict[str | bool, RetUpdRep]]
    ] = dataclasses.field(default_factory=dict)

    args: dict[str, tuple[ast.expr] | tuple[ast.expr, Any]] = dataclasses.field(
        default_factory=dict
    )

    def visit_Return(self, node):
        parser = ParseReturn(line=self.line, next_key_in_dict=self.next_key_in_dict)
        parser.visit(node=node)

        def return_upd_to_tuple(x: ReturnUpdates):
            return (x.next_fn, tuple(sorted(x.remove)), tuple(sorted(x.replace)))

        if parser.const_retn:
            self.return_updates[node.lineno] = return_upd_to_tuple(parser.const_retn)
        elif parser.cond_retn:
            x = self.return_cond_updates[node.lineno] = (
                parser.cond_retn.op,
                return_upd_to_tuple(parser.cond_retn.default_retn),
                {},
            )

            if tr := parser.cond_retn.true_retn:
                x[2][True] = return_upd_to_tuple(tr)

            for k, v in parser.cond_retn.value_retn.items():
                x[2][k] = return_upd_to_tuple(v)

        else:
            raise ValueError(
                "Only constant dictionary returns permitted (line: %d)" % node.lineno
            )

    def visit_FunctionDef(self, node):
        if self.top_fndef:
            return
        self.top_fndef = True

        parse_args = ParseFunctionArguments()
        try:
            parse_args.visit(node.args)
        except AssertionError:
            raise ValueError(f"Error in line {node.lineno}")

        self.args = parse_args.args

        return self.generic_visit(node=node)

    def visit_AsyncFunctionDef(self, node):
        return

    def visit_ClassDef(self, node):
        return


@dataclasses.dataclass
class ParseFunctionArguments(ast.NodeVisitor):
    args: dict[str, tuple[ast.expr] | tuple[ast.expr, ast.Constant]] = (
        dataclasses.field(default_factory=dict)
    )

    def visit_arguments(self, node):
        assert node.kwarg, f"Missing **kwarg"
        assert not node.args, f"Only keyword arguments supported"

        for arg, default in zip(node.kwonlyargs, node.kw_defaults):
            self.args[arg.arg] = (
                (arg.annotation, default) if default is not None else (arg.annotation,)
            )

        for arg in node.args[: -1 * len(node.defaults)]:
            self.args[arg.arg] = (arg.annotation,)

        for arg, default in zip(node.args[-1 * len(node.defaults) :], node.defaults):
            self.args[arg.arg] = (arg.annotation, default)

        for arg in node.args[: -1 * len(node.defaults)] if node.defaults else node.args:
            self.args[arg.arg] = (arg.annotation,)


@dataclasses.dataclass
class ParseReturn(ast.NodeVisitor):
    line: int
    next_key_in_dict: str
    const_retn: Optional[ReturnUpdates] = dataclasses.field(default=None)
    cond_retn: Optional[ConditionalReturns[ast.expr]] = dataclasses.field(default=None)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.line = node.func.lineno
            if node.func.id == "ReturnUpdates":
                parsed_retn = ParseReturnUpdates()
                parsed_retn.visit(node)
                self.const_retn = parsed_retn.ru
            elif node.func.id == "ConditionalReturns":
                parsed_cond = ParseCondReturn(line=self.line)
                parsed_cond.visit(node)
                self.cond_retn = parsed_cond.cond_retn

    def visit_Dict(self, node):
        invalid_keys = [k.value for k in node.keys if not isinstance(k, ast.Constant)]
        if invalid_keys:
            raise ValueError(
                "Only keys as string constants allowed in return dictionary (line: %d)"
                % self.line
            )

        next_fn = None
        for i, key in enumerate(node.keys):
            if isinstance(key, ast.Constant) and key.value == self.next_key_in_dict:
                next_fn_ = node.values[i]
                if not isinstance(next_fn_, ast.Constant) or not isinstance(
                    next_fn_.value, str
                ):
                    raise ValueError(
                        "Next function should be a literal string (line: %d), (key: %s)"
                        % (self.line, key.value)
                    )
                next_fn = next_fn_.value

        self.const_retn = ReturnUpdates(
            next_fn,
            "*",
            **{
                k.value: True
                for k in node.keys
                if isinstance(k, ast.Constant) and not k.value == self.next_key_in_dict
            },
        )


@dataclasses.dataclass
class ParseCondReturn(ast.NodeVisitor):
    line: int
    cond_retn: Optional[ConditionalReturns[ast.expr]] = dataclasses.field(default=None)

    def visit_Call(self, node):
        op, def_arg = node.args[:2]
        assert isinstance(op, ast.expr)

        assert (
            isinstance(def_arg, ast.Call)
            and isinstance(def_arg.func, ast.Name)
            and def_arg.func.id == "ReturnUpdates"
        ), ("Conditional returns should be ReturnUpdates in line %d" % self.line)

        def_parse = ParseReturnUpdates()
        def_parse.visit(def_arg)

        assert def_parse.ru, "Expected condition default return in line %d" % self.line
        self.cond_retn = ConditionalReturns(op, def_parse.ru)

        for kw in node.keywords:
            assert (
                isinstance(kw.value, ast.Call)
                and isinstance(kw.value.func, ast.Name)
                and kw.value.func.id == "ReturnUpdates"
            ), ("Conditional returns should be ReturnUpdates in line %d" % kw.lineno)

            def_parse = ParseReturnUpdates()
            def_parse.visit(node=kw)

            assert def_parse.ru, "Expected condition return in line %d" % kw.lineno
            assert self.cond_retn
            if kw.arg == "true":
                self.cond_retn.true_retn = def_parse.ru
            else:
                self.cond_retn.value_retn[str(kw.arg)] = def_parse.ru


@dataclasses.dataclass
class ParseReturnUpdates(ast.NodeVisitor):
    ru: ReturnUpdates | None = dataclasses.field(default=None)

    def visit_Call(self, node):
        next_fn: str | None = None
        rem: list[str] = []

        if node.args:
            next_name = node.args[0]
            if isinstance(next_name, ast.Constant):
                assert isinstance(next_name.value, str)
                next_fn = next_name.value
            for rem_arg in node.args[1:]:
                assert isinstance(
                    rem_arg, ast.Constant
                ), f"{rem_arg} should be a string value"
                rem.append(rem_arg.value)

        kw: dict[str, bool] = {}

        for keyword in node.keywords:
            assert keyword.arg, f"Use explicit argument names instead of {keyword.arg}"
            kw[keyword.arg] = True

        self.ru = ReturnUpdates(next_fn, *rem, **kw)


class MissingArgument(UserWarning):
    pass


class DefaultArgument(UserWarning):
    pass
