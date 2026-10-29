"""Read AlphaFold-type models (PDB or mmCIF) and their per-residue pLDDT.

AlphaFold DB stores pLDDT in the B-factor column (PDB) / ``B_iso_or_equiv``
(mmCIF), identical for all atoms of a residue. We read the first model and
the first polymer chain, drop hydrogens, waters and ligands, and keep the
residues in file order: residue *i* of this list corresponds to row/column
*i* of the PAE matrix (0-based), whatever the residue numbering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..errors import INVALID_STRUCTURE, FujisanError


@dataclass
class Residue:
    index: int                  # 0-based position in the chain (= PAE index)
    number: int                 # residue sequence number in the file
    icode: str
    name: str                   # three-letter code
    one_letter: str
    plddt: float                # B-factor of CA (pLDDT for AlphaFold models)
    atoms: list = field(default_factory=list)   # (name, element, x, y, z)


@dataclass
class Structure:
    path: str
    chain: str
    residues: list

    @property
    def sequence(self) -> str:
        return "".join(r.one_letter for r in self.residues)

    def __len__(self) -> int:
        return len(self.residues)


def read_structure(path: str | Path) -> Structure:
    """First model, first polymer chain of a PDB or mmCIF file."""
    import gemmi

    path = str(path)
    try:
        st = gemmi.read_structure(path)
    except Exception as exc:
        raise FujisanError(INVALID_STRUCTURE, f"{path}: {exc}") from exc
    if len(st) == 0:
        raise FujisanError(INVALID_STRUCTURE, f"{path}: no model")
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    model = st[0]
    chain = next((ch for ch in model if len(ch) > 0), None)
    if chain is None:
        raise FujisanError(INVALID_STRUCTURE, f"{path}: no protein chain")
    residues = []
    for res in chain:
        info = gemmi.find_tabulated_residue(res.name)
        if info is None or not info.is_amino_acid():
            continue
        atoms = [(a.name, a.element.name, a.pos.x, a.pos.y, a.pos.z) for a in res
                 if not a.has_altloc() or a.altloc == "A"]
        ca = res.find_atom("CA", "*")
        if ca is None or not atoms:
            continue
        letter = info.one_letter_code.upper()
        residues.append(Residue(len(residues), res.seqid.num, res.seqid.icode.strip(), res.name,
                                letter if letter.isalpha() else "X", float(ca.b_iso), atoms))
    if not residues:
        raise FujisanError(INVALID_STRUCTURE, f"{path}: no amino-acid residues with a CA atom")
    return Structure(path, chain.name, residues)


def write_pdb(residues: list, path: str | Path, chain: str = "A") -> None:
    """Write the given residues as PDB ATOM records (input for TM-align)."""
    lines, serial = [], 1
    for r in residues:
        for name, element, x, y, z in r.atoms:
            atom = name if len(name) == 4 else f" {name:<3s}"
            lines.append(f"ATOM  {serial % 100000:5d} {atom:4s} {r.name:>3s} {chain}{r.number % 10000:4d}"
                         f"{(r.icode or ' ')[:1]}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00{r.plddt:6.2f}"
                         f"          {element:>2s}")
            serial += 1
    lines += ["TER", "END"]
    Path(path).write_text("\n".join(lines) + "\n")


def read_fasta(path: str | Path) -> tuple[str, str]:
    """(identifier, sequence) of the first record of a FASTA file."""
    name, seq = None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                break
            name = line[1:].split()[0] if line[1:].strip() else "query"
        elif line:
            seq.append(line)
    sequence = "".join(seq).replace(" ", "").upper().rstrip("*")
    if name is None or not sequence:
        from ..errors import INVALID_INPUT
        raise FujisanError(INVALID_INPUT, f"{path}: no FASTA record")
    return name, sequence
