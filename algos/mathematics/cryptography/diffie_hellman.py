#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.click
"""
Diffie-Hellman Key Exchange — Step-by-Step Demo

Demonstrates the protocol with clear separation between private peer state
and public bus traffic. After each step, an inventory shows exactly what
Alice, Bob, and a MITM observer each know — making it visible that the
eavesdropper can never derive the shared secret without solving the
discrete logarithm problem.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import click

SEED = 42
P = 9973
G = 5

Knowledge = dict[str, int | str]


@dataclass
class ExchangeState:
    """Tracks what each party knows at every step of the protocol."""

    p: int
    g: int
    alice: Knowledge = field(default_factory=dict)
    bob: Knowledge = field(default_factory=dict)
    bus: Knowledge = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)

    def _fmt_knowledge(self, who: str, k: Knowledge) -> str:
        items = ", ".join(f"{name} = {val}" for name, val in k.items())
        return f"  {who:>5} knows: {{{items}}}"

    def snapshot(self) -> str:
        return "\n".join(
            [
                self._fmt_knowledge("Alice", self.alice),
                self._fmt_knowledge("Bob", self.bob),
                self._fmt_knowledge("MITM", self.bus),
            ]
        )


def run_exchange(p: int, g: int, a: int, b: int) -> ExchangeState:
    """
    Run the Diffie-Hellman protocol, recording each party's knowledge
    after every step.
    """
    st = ExchangeState(p=p, g=g)
    thin = "─" * 60

    # Step 0: public parameters
    pub: Knowledge = {"p": p, "g": g}
    st.alice.update(pub)
    st.bob.update(pub)
    st.bus.update(pub)
    st.steps.append(
        f"PUBLIC PARAMETERS (agreed in the open)\n{thin}\n"
        f"  p = {p} (prime modulus)\n"
        f"  g = {g} (generator)\n\n" + st.snapshot()
    )

    # Step 1: Alice picks secret, publishes A
    big_a = pow(g, a, p)
    st.alice["a"] = a
    st.alice["A"] = big_a
    st.bus["A"] = big_a
    st.bob["A"] = big_a
    st.steps.append(
        f"STEP 1 — Alice picks secret a, publishes A = g^a mod p\n{thin}\n"
        f"  a = {a}  (kept secret)\n"
        f"  A = {g}^{a} mod {p} = {big_a}  (sent on the bus)\n\n" + st.snapshot()
    )

    # Step 2: Bob picks secret, publishes B
    big_b = pow(g, b, p)
    st.bob["b"] = b
    st.bob["B"] = big_b
    st.bus["B"] = big_b
    st.alice["B"] = big_b
    st.steps.append(
        f"STEP 2 — Bob picks secret b, publishes B = g^b mod p\n{thin}\n"
        f"  b = {b}  (kept secret)\n"
        f"  B = {g}^{b} mod {p} = {big_b}  (sent on the bus)\n\n" + st.snapshot()
    )

    # Step 3: both sides compute shared secret
    s_alice = pow(big_b, a, p)
    s_bob = pow(big_a, b, p)
    st.alice["s"] = s_alice
    st.bob["s"] = s_bob
    st.bus["s"] = "???"
    st.steps.append(
        f"STEP 3 — Both sides compute the shared secret\n{thin}\n"
        f"  Alice:  s = B^a mod p = {big_b}^{a} mod {p} = {s_alice}\n"
        f"  Bob:    s = A^b mod p = {big_a}^{b} mod {p} = {s_bob}\n"
    )

    return st


def format_exchange(st: ExchangeState) -> str:
    """Render the full exchange as a single string."""
    w = 64
    sep = "═" * w

    header = f"\n{sep}\n{'DIFFIE-HELLMAN KEY EXCHANGE':^{w}}\n{sep}\n"
    body = "\n\n".join(st.steps)
    return f"{header}\n{body}\n"


# --- CLI ---


@click.command()
def main() -> None:
    """Step-by-step Diffie-Hellman key exchange demo."""
    rng = random.Random(SEED)
    a = rng.randint(2, P - 2)
    b = rng.randint(2, P - 2)

    st = run_exchange(P, G, a, b)
    click.echo_via_pager(format_exchange(st))


if __name__ == "__main__":
    main()
