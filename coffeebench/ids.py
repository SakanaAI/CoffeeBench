"""Deterministic, environment-owned ID generator.

CoffeeBench originally minted listing / offer / deal / message / invoice /
journal-entry IDs with `uuid.uuid4()`, which is unseeded randomness. That makes
two runs of the *same seed* produce different IDs and, more importantly, makes a
checkpoint+resume run non-identical to an uninterrupted run at the byte level —
even though the economics are the same.

Replacing the UUIDs with a single monotonic counter per prefix makes every ID a
pure function of *how many* IDs of that kind were minted before it. Because the
counter is captured in the checkpoint and the post-checkpoint action sequence is
deterministic (seeded env RNG + deterministic scripted agents + deterministic
serial tool-apply order), an uninterrupted run and a checkpoint-resumed run mint
the *same* IDs. Two branches loaded from the same checkpoint likewise mint the
same IDs. This is the substrate the checkpoint equality check relies on, and it
also gives future decision-lineage work (Valto) stable, greppable identifiers.

The generator is owned by the `Marketplace` and shared with the `Environment`
so there is a single counter namespace across the whole run.
"""

from __future__ import annotations


class IdGenerator:
    """Monotonic per-prefix counter. `next("lst")` → `lst_000001`, ...

    Serialized/restored via `state()` / `IdGenerator.from_state()` so the
    counter continues exactly across a checkpoint boundary.
    """

    __slots__ = ("_counters",)

    def __init__(self, counters: dict[str, int] | None = None):
        self._counters: dict[str, int] = dict(counters or {})

    def next(self, prefix: str) -> str:
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        return f"{prefix}_{n:06d}"

    def state(self) -> dict[str, int]:
        """JSON-serializable snapshot of the counters."""
        return dict(self._counters)

    def restore(self, counters: dict[str, int]) -> None:
        self._counters = {str(k): int(v) for k, v in (counters or {}).items()}

    @classmethod
    def from_state(cls, counters: dict[str, int] | None) -> "IdGenerator":
        return cls(counters or {})
