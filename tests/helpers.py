"""Synthetic structures for the tests (no network, no external files)."""

import math

AA3 = ["ALA", "GLY", "SER", "LEU", "LYS", "GLU", "ASP", "VAL"]


def make_pdb(plddts, first_number=1, chain="A", seq=None):
    """A helix-like CA/N/C/O trace with one residue per pLDDT value (B-factor = pLDDT)."""
    lines, serial = [], 1
    for i, b in enumerate(plddts):
        name = seq[i] if seq else AA3[i % len(AA3)]
        t = i * 100.0 / 180.0 * math.pi
        x, y, z = 2.3 * math.cos(t), 2.3 * math.sin(t), 1.5 * i
        for atom, el, dx in (("N", "N", -0.5), ("CA", "C", 0.0), ("C", "C", 0.5), ("O", "O", 0.9)):
            lines.append(f"ATOM  {serial:5d}  {atom:<3s} {name} {chain}{first_number + i:4d}    "
                         f"{x + dx:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           {el}")
            serial += 1
    lines += ["TER", "END"]
    return "\n".join(lines) + "\n"
