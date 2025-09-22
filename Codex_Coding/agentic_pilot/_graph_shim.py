"""Minimal LangGraph-compatible shim for environments without the dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

START = "__start__"
END = "__end__"


class MemorySaver:
    """No-op checkpointer placeholder used for compatibility."""

    def __init__(self) -> None:  # pragma: no cover - intentionally trivial
        self.store: Dict[str, Any] = {}


@dataclass
class _ConditionalRoute:
    selector: Callable[[Any], str]
    mapping: Dict[str, str]


class StateGraph:
    """Simplified directed graph runner that mimics LangGraph APIs used in tests."""

    def __init__(self, state_type: Type[Any]):
        self.state_type = state_type
        self._nodes: Dict[str, Callable[[Any], Any]] = {}
        self._edges: Dict[str, List[str]] = {}
        self._conditional: Dict[str, _ConditionalRoute] = {}
        self._start: Optional[str] = None

    def add_node(self, name: str, fn: Callable[[Any], Any]) -> None:
        self._nodes[name] = fn

    def add_edge(self, source: str, target: str) -> None:
        if source == START:
            self._start = target
            return
        self._edges.setdefault(source, []).append(target)

    def add_conditional_edges(
        self,
        source: str,
        selector: Callable[[Any], str],
        mapping: Dict[str, str],
    ) -> None:
        self._conditional[source] = _ConditionalRoute(selector, mapping)

    def compile(self, checkpointer: Optional[MemorySaver] = None) -> "CompiledStateGraph":
        if self._start is None:
            raise ValueError("Graph requires a start node")
        return CompiledStateGraph(self._start, dict(self._nodes), dict(self._edges), dict(self._conditional))


class CompiledStateGraph:
    """Execute the simplified graph sequentially."""

    def __init__(
        self,
        start: str,
        nodes: Dict[str, Callable[[Any], Any]],
        edges: Dict[str, List[str]],
        conditional: Dict[str, _ConditionalRoute],
    ) -> None:
        self._start = start
        self._nodes = nodes
        self._edges = edges
        self._conditional = conditional

    def invoke(self, state: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        current = self._start
        while current and current != END:
            node_fn = self._nodes[current]
            state = node_fn(state)

            if current in self._conditional:
                route = self._conditional[current]
                key = route.selector(state)
                current = route.mapping.get(key, END)
                continue

            next_edges = self._edges.get(current, [])
            current = next_edges[0] if next_edges else END
        return state