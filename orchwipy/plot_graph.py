import dataclasses
import io
import logging
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, TypeVarTuple, Any

import networkx as nx

log = logging.getLogger(__name__)

from .call_graph import CallGraph


@dataclasses.dataclass
class PlotGraph(CallGraph):
    _: dataclasses.KW_ONLY

    def call_graph(self):
        graph: nx.DiGraph = nx.DiGraph()
        node_requires = {fname: dict(v) for fname, v in self.requires.items()}
        edges = list(self.edges_from_dyn())
        graph.add_nodes_from(node_requires.items())

        unique_cond: set[tuple[int, str, str]] = set()
        for prop in self.node_props.values():
            if prop.get("operand") and prop.get("match"):
                unique_cond.add((prop["line"], prop["operand"], prop["match"]))

        cond_colors = cond_node_color(list(unique_cond))

        for node, prop in self.node_props.items():
            if prop.get("operand") and prop.get("match"):
                ln, operand, match = prop["line"], prop["operand"], prop["match"]
                color = cond_colors[(ln, operand, match)]
                prop.update(dict(color=color))
                del prop["match"]
                del prop["operand"]
            graph.add_node(node, **prop)
        graph.add_edges_from(edges)
        return graph

    def save_dot(self, f: io.TextIOWrapper, *, top_node: str | None = None):
        # agraph = nx.nx_agraph.to_agraph(self.call_graph)
        for ln in self.digraph_to_dot(self.call_graph(), top_node=top_node):
            f.write(f"{ln}\n")

    @staticmethod
    def digraph_to_dot(g: nx.DiGraph, top_node: str | None):

        def to_dot_str(x: Any):
            if isinstance(x, str) and " " in x:
                return f'"{x}"'
            return f"{x}"

        def format_attribs(attribs: dict[str, Any]):
            return ",".join(f"{k}={to_dot_str(v)}" for k, v in attribs.items())

        def node_lines():
            nodes = list(g.nodes.keys())
            if top_node:
                nodes = sorted(nodes, key=lambda x: int(x == top_node), reverse=True)

            for node in g.nodes.keys():
                attribs: dict[str, Any] = g.nodes[node]
                attrib_str = format_attribs(attribs)
                yield "%s %s;" % (
                    to_dot_str(node),
                    f"[{attrib_str}]" if attrib_str else "",
                )

        def edge_lines():
            for from_, to in g.edges.keys():
                attribs: dict[str, Any] = g.edges[(from_, to)]
                attrib_str = format_attribs(attribs)
                yield to_dot_str(from_) + " -> " + to_dot_str(to) + (
                    " %s;" % (f"[{attrib_str}]" if attrib_str else "")
                )

        yield 'strict digraph "" {'
        yield from node_lines()
        yield from edge_lines()
        yield "}"


try:
    import networkx as nx
except ModuleNotFoundError:
    if not TYPE_CHECKING:
        GraphableFunctions = CallGraph
    else:
        raise
else:
    GraphableFunctions = PlotGraph


def cond_node_color(props: list[tuple[int, str, str]], *sat_value_alpha):
    keys = set([x[0] for x in props])
    parts = len(keys) + 1

    divs: list[float] = [0]
    for i in range(len(keys)):
        divs.append(divs[-1] + 1.0 / parts)
    divs = divs[1:]
    sat_value_alpha = sat_value_alpha or (0.3, 0.9, 0.5)
    colors = ["%.3f %.3f %.3f %.3f" % (div, *sat_value_alpha) for div in divs]
    color_keyed = dict(zip(keys, colors))

    return {k: color_keyed[k[0]] for k in props}
