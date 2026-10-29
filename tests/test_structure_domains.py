import numpy as np
import pytest

from FUJISAN.errors import LENGTH_MISMATCH, NO_CONFIDENT_RESIDUES, NO_DOMAIN, FujisanError
from FUJISAN.features import domains as dm
from FUJISAN.features.structure import read_structure

from helpers import make_pdb


def _structure(tmp_path, plddts, first=1, suffix=".pdb"):
    p = tmp_path / "m.pdb"
    p.write_text(make_pdb(plddts, first_number=first))
    if suffix == ".cif":
        import gemmi
        st = gemmi.read_structure(str(p))
        st.setup_entities()
        c = tmp_path / "m.cif"
        st.make_mmcif_document().write_file(str(c))
        p = c
    return read_structure(p)


@pytest.mark.parametrize("suffix", [".pdb", ".cif"])
def test_plddt_per_residue_in_order(tmp_path, suffix):
    pl = [91.5, 65.0, 70.0, 88.25]
    st = _structure(tmp_path, pl, first=101, suffix=suffix)
    assert [r.plddt for r in st.residues] == pytest.approx(pl)
    assert [r.number for r in st.residues] == [101, 102, 103, 104]
    assert [r.index for r in st.residues] == [0, 1, 2, 3]
    assert st.sequence == "AGSL"


def test_pae_index_maps_to_residue_order_not_number(tmp_path):
    # numbering starts at 101: PAE index 0 must be residue 101 (no off-by-one)
    st = _structure(tmp_path, [90.0] * 40, first=101)
    doms = dm.segment(st, np.zeros((40, 40)), clusters=[list(range(0, 35)), list(range(35, 40))])
    assert [r.number for r in doms[0].residues][:2] == [101, 102]
    assert doms[0].residues[-1].number == 135
    assert [r.number for r in doms[1].residues] == [136, 137, 138, 139, 140]


def test_plddt_filter_and_minimum_size(tmp_path):
    pl = [90.0] * 29 + [70.0] + [69.99] * 10 + [95.0] * 29
    st = _structure(tmp_path, pl)
    clusters = [list(range(0, 40)), list(range(40, 69))]
    doms = dm.segment(st, np.zeros((69, 69)), clusters=clusters)
    # domain 1: 29 residues at 90 + 1 at exactly 70.00 kept (>= 70), 10 below dropped -> 30 = usable
    assert doms[0].n_assigned == 40 and doms[0].n_kept == 30 and doms[0].usable
    assert all(r.plddt >= 70 for r in doms[0].residues)
    # domain 2: 29 residues -> too small
    assert doms[1].n_kept == 29 and not doms[1].usable
    usable = dm.usable_domains(doms, tmp_path / "w", "prot")
    assert [d.number for d in usable] == [1]
    assert (tmp_path / "w" / "prot_domain1.pdb").exists()
    st2 = read_structure(usable[0].pdb_path)
    assert len(st2) == 30 and [r.number for r in st2.residues][-1] == 30


def test_no_usable_domain(tmp_path):
    st = _structure(tmp_path, [90.0] * 20)
    doms = dm.segment(st, np.zeros((20, 20)), clusters=[list(range(20))])
    with pytest.raises(FujisanError) as e:
        dm.usable_domains(doms, tmp_path, "p")
    assert e.value.reason == NO_DOMAIN


def test_no_confident_residue(tmp_path):
    st = _structure(tmp_path, [50.0] * 40)
    with pytest.raises(FujisanError) as e:
        dm.segment(st, np.zeros((40, 40)), clusters=[list(range(40))])
    assert e.value.reason == NO_CONFIDENT_RESIDUES


def test_length_mismatch(tmp_path):
    st = _structure(tmp_path, [90.0] * 10)
    with pytest.raises(FujisanError) as e:
        dm.segment(st, np.zeros((12, 12)))
    assert e.value.reason == LENGTH_MISMATCH


def test_pae_to_domains_defaults_and_two_blocks():
    assert (dm.PAE_POWER, dm.PAE_CUTOFF, dm.GRAPH_RESOLUTION, dm.PAE_LIBRARY) == (1.0, 5.0, 1.0, "igraph")
    n = 80
    pae = np.full((n, n), 25.0)
    pae[:50, :50] = 1.0
    pae[50:, 50:] = 1.0
    np.fill_diagonal(pae, 0.0)
    c1 = dm.cluster_pae(pae)
    assert sorted(map(len, c1), reverse=True) == [50, 30]
    assert dm.cluster_pae(pae) == c1                # seeded: reproducible
