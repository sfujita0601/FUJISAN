import pytest

from FUJISAN.errors import NO_ACCESSION, FujisanError
from FUJISAN.features import blast as bl

LINE = ("protein1\tprotein2\tprotein2\t39.394\t396\t3.62e-99\t289\t1\t392\t3\t397\t396\t408\tprotein2\n")


def test_outfmt_fields_are_explicit():
    assert bl.OUTFMT_FIELDS[3:7] == ["pident", "length", "evalue", "bitscore"]


def test_parse_outfmt6_values():
    h = bl.parse_outfmt6(LINE)[0]
    assert h.bitscore == 289.0 and h.pident == 39.394
    assert (h.qlen, h.slen, h.evalue) == (396, 408, 3.62e-99)


def test_bitscore_and_identity_features():
    h = bl.parse_outfmt6(LINE)[0]
    assert bl.bitscore_feature(h) == 289.0      # raw bit score of the best HSP
    assert h.pident == 39.394                    # Full_SeqID


def test_first_row_is_used_when_several():
    text = LINE + LINE.replace("\t289\t", "\t100\t")
    hits = bl.parse_outfmt6(text)
    assert len(hits) == 2 and hits[0].bitscore == 289.0


@pytest.mark.parametrize("tid, acc", [
    ("sp|P12345|ABC_HUMAN", "P12345"),
    ("tr|A0A0B4J2F0|A0A0B4J2F0_HUMAN", "A0A0B4J2F0"),
    ("tr|Q67890|NAME", "Q67890"),
    ("P12345", "P12345"),
    ("P12345.2", "P12345"),
    ("UniRef100_Q9HBA0", "Q9HBA0"),
    ("AF-Q9HBA0-F1", "Q9HBA0"),
])
def test_accession_parser_accepts(tid, acc):
    assert bl.parse_uniprot_accession(tid) == acc


@pytest.mark.parametrize("tid", ["P12345-2", "sp|P12345-2|ABC_HUMAN", "mylab_contig_0042", "P11295,O06897|RHEA:23228",
                                 "gi|123456|ref|NP_000001.1|", "UPI0000000001", "BL_ORD_ID:12", "12345"])
def test_accession_parser_rejects(tid):
    with pytest.raises(FujisanError) as e:
        bl.parse_uniprot_accession(tid)
    assert e.value.reason == NO_ACCESSION


def test_hit_accession_reports_reason_of_the_id():
    h = bl.parse_outfmt6(LINE)[0]
    h.sseqid, h.sacc, h.stitle = "P00509-2", "P00509-2", "isoform-like title"
    with pytest.raises(FujisanError) as e:
        bl.hit_accession(h)
    assert "isoform" in e.value.detail


def test_missing_blast_db(tmp_path):
    with pytest.raises(FujisanError) as e:
        bl.check_blast_db(str(tmp_path / "nodb"))
    assert e.value.reason == "BLAST database not found"
    (tmp_path / "db.pin").write_text("")
    bl.check_blast_db(str(tmp_path / "db"))


def test_pair_and_search_use_the_same_blast_call(monkeypatch):
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        return LINE if "-query" in cmd else ""

    monkeypatch.setattr(bl, "_run", fake_run)
    monkeypatch.setattr(bl, "find_makeblastdb", lambda b: "makeblastdb")
    bl.blast_pair("MKV", "MKVL", "blastp")
    bl.blast_search("MKV", "/db/swissprot", "blastp", max_hits=100)
    make, pair, search = calls
    assert make[0] == "makeblastdb" and "-dbtype" in make
    assert "-subject" not in pair and "-db" in pair                     # one-sequence database
    strip = lambda c: [x for i, x in enumerate(c) if c[i - 1] not in ("-query", "-db", "-max_target_seqs")]
    assert strip(pair) == strip(search)                                 # identical options otherwise
    assert "-dbsize" not in pair and "-dbsize" not in search            # BLAST's own database size by default
    assert pair[pair.index("-max_hsps") + 1] == "1"


def test_dbsize_only_when_requested(monkeypatch):
    calls = []
    monkeypatch.setattr(bl, "_run", lambda cmd: calls.append(cmd) or "")
    bl.blast_search("MKV", "/db/x", "blastp", max_hits=5, dbsize=12345)
    assert calls[0][calls[0].index("-dbsize") + 1] == "12345"
