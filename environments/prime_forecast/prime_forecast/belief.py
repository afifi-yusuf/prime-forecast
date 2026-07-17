"""Structured linguistic belief state (BLF §C.1), ported from haruspex."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel


class BeliefUpdate(BaseModel):
    """The updated_belief payload the agent attaches to every tool call."""

    p: float | None = None
    confidence: str | None = None
    update_reasoning: str | None = None
    evidence_for: list[str] | None = None
    evidence_against: list[str] | None = None
    key_uncertainties: list[str] | None = None
    base_rate_anchor: str | None = None


@dataclass
class BeliefState:
    p: float = 0.5
    base_rate_anchor: str = ""
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    key_uncertainties: list[str] = field(default_factory=list)
    confidence: str = "low"  # low | medium | high
    update_reasoning: str = ""
    searches_tried: list[str] = field(default_factory=list)
    step: int = 0

    def to_prompt_str(self, max_steps: int = 10) -> str:
        lines = [
            f"Current belief state (step {self.step}/{max_steps}):",
            f"  Probability: {self.p:.3f}",
        ]
        if self.base_rate_anchor:
            lines.append(f"  Base rate anchor: {self.base_rate_anchor}")
        if self.evidence_for:
            lines.append("  Evidence FOR resolution (pushes p up):")
            for e in self.evidence_for:
                lines.append(f"    - {e}")
        if self.evidence_against:
            lines.append("  Evidence AGAINST resolution (pushes p down):")
            for e in self.evidence_against:
                lines.append(f"    - {e}")
        if self.key_uncertainties:
            lines.append("  Key uncertainties:")
            for u in self.key_uncertainties:
                lines.append(f"    - {u}")
        lines.append(f"  Confidence: {self.confidence}")
        if self.searches_tried:
            lines.append(f"  Searches tried: {self.searches_tried}")
        if self.update_reasoning:
            lines.append(f"  Last update: {self.update_reasoning}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "p": self.p,
            "base_rate_anchor": self.base_rate_anchor,
            "evidence_for": list(self.evidence_for),
            "evidence_against": list(self.evidence_against),
            "key_uncertainties": list(self.key_uncertainties),
            "confidence": self.confidence,
            "searches_tried": list(self.searches_tried),
            "step": self.step,
        }
        if self.update_reasoning:
            d["update_reasoning"] = self.update_reasoning
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "BeliefState":
        if not d or not isinstance(d, dict):
            return cls()
        fields = cls.__dataclass_fields__
        clean = {k: v for k, v in d.items() if k in fields}
        if "p" in clean:
            try:
                clean["p"] = max(0.05, min(0.95, float(clean["p"])))
            except (TypeError, ValueError):
                clean["p"] = 0.5
        return cls(**clean)

    def merge_update(self, update: dict | None) -> None:
        """Apply an updated_belief payload from a tool call."""
        if not update or not isinstance(update, dict):
            return
        if "p" in update:
            try:
                self.p = max(0.05, min(0.95, float(update["p"])))
            except (TypeError, ValueError):
                pass
        for key in ("base_rate_anchor", "confidence", "update_reasoning"):
            if update.get(key):
                setattr(self, key, str(update[key]))
        for key in ("evidence_for", "evidence_against", "key_uncertainties", "searches_tried"):
            if update.get(key) and isinstance(update[key], list):
                # BLF accumulates evidence lists across steps.
                existing = getattr(self, key)
                for item in update[key]:
                    s = str(item).strip()
                    if s and s not in existing:
                        existing.append(s)
        self.step += 1

    def evidence_char_count(self) -> int:
        return sum(len(x) for x in self.evidence_for + self.evidence_against + self.key_uncertainties)

    def compact_if_needed(self, threshold: int = 2000) -> bool:
        """Truncate long evidence lists to keep context bounded."""
        if self.evidence_char_count() <= threshold:
            return False
        for key in ("evidence_for", "evidence_against", "key_uncertainties"):
            items = getattr(self, key)
            if len(items) > 4:
                setattr(self, key, items[:4] + [f"[... {len(items) - 4} more items truncated]"])
        return True


@dataclass
class SearchResult:
    index: int
    title: str
    url: str
    snippet: str
    body: str = ""

    def to_snippet_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet[:800],
        }


@dataclass
class SearchBatch:
    query: str
    results: list[SearchResult] = field(default_factory=list)


@dataclass
class SearchStore:
    """Per-episode store for search results (BLF Algorithm 1)."""

    searches: list[SearchBatch] = field(default_factory=list)

    def add_search(self, query: str, results: list[SearchResult]) -> int:
        self.searches.append(SearchBatch(query=query, results=results))
        return len(self.searches) - 1

    def get_results(self, search_index: int, result_indices: list[int]) -> list[SearchResult]:
        if not (0 <= search_index < len(self.searches)):
            return []
        batch = self.searches[search_index]
        out = []
        for i in result_indices:
            if 0 <= i < len(batch.results):
                out.append(batch.results[i])
        return out
