"""
SkyNomad Sec 4.4 — Lifetime / survival analysis for time-to-eviction.

Kaplan–Meier style hazard, cumulative hazard, survival, and expected remaining (Lbar).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LifetimeStats:
    """Survival stats from observed lifetimes (hours) and censoring flags.

    lifetimes_hours: observed duration until eviction or censor
    events: True if eviction observed, False if right-censored (still alive / migrated)
    """

    lifetimes_hours: list[float] = field(default_factory=list)
    events: list[bool] = field(default_factory=list)

    def add(self, lifetime_hours: float, *, evicted: bool) -> None:
        self.lifetimes_hours.append(max(0.0, float(lifetime_hours)))
        self.events.append(bool(evicted))

    def n(self) -> int:
        return len(self.lifetimes_hours)

    def compute_at_risk(self, t: float) -> int:
        """Number of observations still at risk just before time t."""
        return sum(1 for L in self.lifetimes_hours if L >= t)

    def _event_times(self) -> list[float]:
        """Unique eviction times (sorted)."""
        times = sorted({L for L, e in zip(self.lifetimes_hours, self.events) if e and L > 0})
        return times

    def compute_hazard(self, t: float) -> float:
        """Discrete hazard at time t: d_t / n_t (evictions at t / at-risk at t)."""
        at_risk = self.compute_at_risk(t)
        if at_risk <= 0:
            return 0.0
        deaths = sum(
            1
            for L, e in zip(self.lifetimes_hours, self.events)
            if e and abs(L - t) < 1e-9
        )
        # Also count events in a small bin if exact match is rare
        if deaths == 0:
            deaths = sum(
                1
                for L, e in zip(self.lifetimes_hours, self.events)
                if e and abs(L - t) <= 1e-6
            )
        return deaths / at_risk

    def compute_cumulative_hazard(self, t: float) -> float:
        """Nelson–Aalen style cumulative hazard up to t."""
        H = 0.0
        for et in self._event_times():
            if et > t:
                break
            H += self.compute_hazard(et)
        return H

    def compute_survival(self, t: float) -> float:
        """Kaplan–Meier survival S(t) = Π (1 - d_i/n_i) for t_i <= t."""
        if self.n() == 0:
            # No data: optimistic default survival decay
            return max(0.0, min(1.0, 1.0 - 0.05 * max(0.0, t) / 24.0))

        s = 1.0
        for et in self._event_times():
            if et > t:
                break
            h = self.compute_hazard(et)
            s *= max(0.0, 1.0 - h)
        return max(0.0, min(1.0, s))

    def expected_remaining(
        self,
        age_hours: float = 0.0,
        *,
        horizon_hours: float = 168.0,
        step_hours: float = 1.0,
    ) -> float:
        """Expected remaining lifetime Lbar given current age (hours).

        Integrates conditional survival: Lbar ≈ ∫_0^H S(age+u)/S(age) du
        """
        s_age = max(self.compute_survival(age_hours), 1e-9)
        acc = 0.0
        u = 0.0
        while u < horizon_hours:
            s_u = self.compute_survival(age_hours + u + step_hours)
            cond = s_u / s_age
            if cond <= 1e-6:
                break
            acc += cond * step_hours
            u += step_hours
        # Floor so ranking never divides by zero
        return max(acc, step_hours)

    @classmethod
    def from_exponential(cls, mean_lifetime_hours: float = 24.0, n: int = 50) -> "LifetimeStats":
        """Seed synthetic exponential-like samples for demos/tests."""
        import random

        rng = random.Random(int(mean_lifetime_hours * 1000) % 10_000)
        stats = cls()
        for _ in range(n):
            # Inverse CDF of Exp(1/mean)
            u = max(1e-9, rng.random())
            L = -mean_lifetime_hours * __import__("math").log(u)
            # 80% observed evictions, 20% censored
            evicted = rng.random() < 0.8
            if not evicted:
                L = min(L, mean_lifetime_hours)  # censored early
            stats.add(L, evicted=evicted)
        return stats
