"""``fujisan`` command line: feature generation and prediction with the updated FUJISAN model.

    fujisan pair        five features + probability for one protein pair
    fujisan search      one query -> BLASTP against a database -> AFDB models of the hits -> ranking
    fujisan batch       several queries (AFDB IDs or local files), one sub-directory each
    fujisan explain     per-pair feature contributions (SHAP, log-odds) of a feature table
    fujisan cache-info  size of the AFDB download cache

Only the ``updated`` model (BitScore, Full_SeqID, Dmn_RMSD, Dmn_SeqID, Dmn_TM)
is supported; the pocket features of the ``paper2024`` model are not generated.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

from . import __version__
from .errors import INVALID_INPUT, MODEL_MISMATCH, FujisanError
from .features import blast as bl
from .features import pipeline as pp
from .features import tmalign as tm
from .features.alphafold import normalize_afdb_id
from .prediction import FEATURES, add_probabilities, get_model
from .utils.cache import Cache, human_size
from .utils.metadata import run_metadata, write_metadata


# --------------------------------------------------------------------------- shared options

def _common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("tools and storage")
    g.add_argument("--blastp-bin", default="blastp", help="BLAST+ blastp executable (default: blastp on PATH)")
    g.add_argument("--tmalign-bin", default="TMalign", help="TM-align executable (default: TMalign on PATH)")
    g.add_argument("--cache-dir", type=Path, default=None,
                   help="AFDB download cache (default: $FUJISAN_CACHE_DIR or ~/.cache/fujisan)")
    keep = g.add_mutually_exclusive_group()
    keep.add_argument("--keep-downloads", dest="keep_downloads", action="store_true", default=True,
                      help="keep AFDB structures / PAE in the cache (default)")
    keep.add_argument("--no-keep-downloads", dest="keep_downloads", action="store_false",
                      help="download into a temporary cache that is deleted at the end of the run")
    g.add_argument("--refresh-afdb", action="store_true", help="ignore cached AFDB files and download again")
    g.add_argument("--keep-intermediates", action="store_true",
                   help="keep domain PDB files and raw TM-align output (in <output>/intermediates)")
    g.add_argument("--blast-dbsize", type=int, default=None,
                   help="optional BLAST -dbsize override (residues) for all BLAST runs; default: not set "
                        "(BLAST's own database size)")
    g.add_argument("--debug", action="store_true", help="show Python tracebacks")


def _search_opts(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("BLAST search")
    g.add_argument("--blast-db", required=True, help="BLAST protein database prefix prepared by the user")
    g.add_argument("--blast-db-label", default="", help='label stored in run_metadata.json, e.g. "Swiss-Prot 2026_01"')
    g.add_argument("--max-hits", type=int, default=100, help="BLAST hits to analyse per query (default 100; 0 = all)")
    g.add_argument("--threads", type=int, default=1, help="BLAST threads (default 1)")
    g.add_argument("--workers", type=int, default=1,
                   help="targets processed in parallel (default 1; AFDB requests stay throttled)")


def _tools(args) -> pp.Tools:
    blastp = bl.find_blastp(args.blastp_bin)
    if args.command == "pair":
        bl.find_makeblastdb(blastp)
    tmalign = tm.find_tmalign(args.tmalign_bin)
    return pp.Tools(blastp, tmalign, getattr(args, "threads", 1), getattr(args, "workers", 1), args.debug,
                    args.blast_dbsize)


def _tool_info(t: pp.Tools) -> dict:
    return {"blast_version": bl.blast_version(t.blastp), "tmalign": t.tmalign,
            "tmalign_version": tm.tmalign_version(t.tmalign)}


def _cache(args):
    if args.keep_downloads:
        return Cache(args.cache_dir), None
    tmp = tempfile.mkdtemp(prefix="fujisan_cache_")
    return Cache(tmp), tmp


def _storage_banner(args, cache) -> None:
    n = "all" if args.max_hits <= 0 else f"up to {args.max_hits}"
    pp.log(f"Processing {n} BLAST hits per query.")
    if args.keep_downloads:
        pp.log("AFDB structures and PAE files will be cached.")
    else:
        pp.log("AFDB files go to a temporary cache that is deleted at the end (--no-keep-downloads).")
    pp.log("Large analyses may consume substantial disk space.")
    pp.log(f"Cache directory: {cache.root}")


def _cache_footer(cache, tmp) -> None:
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
        pp.log("Temporary AFDB cache removed.")
    else:
        info = cache.info()
        pp.log(f"FUJISAN cache: {human_size(info['total_bytes'])}")
        pp.log(f"Cache directory: {info['cache_dir']}")


def _protein(args, n: str, cache):
    """Protein n (1 or 2) of `pair`, or the query of `search` (n = 'query')."""
    pre = "query_" if n == "query" else ""
    suf = "" if n == "query" else n
    afdb = getattr(args, f"{pre}afdb_id{suf}")
    structure, pae, fasta = (getattr(args, f"{pre}structure{suf}"), getattr(args, f"{pre}pae{suf}"),
                             getattr(args, f"{pre}fasta{suf}"))
    if afdb and (structure or pae):
        raise FujisanError(INVALID_INPUT, f"protein {n}: give either an AFDB ID or local files, not both")
    if afdb:
        return pp.protein_from_afdb(afdb, cache, args.refresh_afdb)
    if not (structure and pae):
        raise FujisanError(INVALID_INPUT, f"protein {n}: --{pre}afdb-id{suf}, or --{pre}structure{suf} and "
                                          f"--{pre}pae{suf} (and optionally --{pre}fasta{suf}) are required")
    return pp.protein_from_local(structure, pae, fasta)


# --------------------------------------------------------------------------- commands

def cmd_pair(args) -> int:
    tools = _tools(args)
    get_model()
    cache, tmp = _cache(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = pp.WorkDir(out.parent, args.keep_intermediates)
    try:
        s1, s2 = _protein(args, "1", cache), _protein(args, "2", cache)
        if s1.label == s2.label:
            s2.label += "_2"
        p1 = pp.prepare_protein(s1, work.path / "protein1")
        p2 = pp.prepare_protein(s2, work.path / "protein2")
        row = pp.pair_features(p1, p2, tools, keep_dir=work.path if args.keep_intermediates else None)
        row = add_probabilities([row])[0]
        row["ProteinID1"], row["ProteinID2"] = row["QueryID"], row["TargetID"]
        cols = ["ProteinID1", "ProteinID2"] + FEATURES + ["FUJISAN_probability"] + \
            [c for c in pp.FEATURE_COLUMNS if c not in ("QueryID", "TargetID", "BlastSubjectID") + tuple(FEATURES)]
        pp.write_tsv(out, [row], cols)
        meta = run_metadata("pair", vars(args), _tool_info(tools))
        meta["proteins"] = {"1": vars(s1) | {"sequence": len(s1.sequence)}, "2": vars(s2) | {"sequence": len(s2.sequence)}}
        write_metadata(out.with_name(out.stem + ".run_metadata.json"), meta)
        print("\t".join(cols[:8]))
        print("\t".join(str(row[c]) for c in cols[:8]))
        pp.log(f"Written: {out}")
    finally:
        work.cleanup()
        _cache_footer(cache, tmp)
    return 0


def _check_search_inputs(args, tools) -> None:
    bl.check_blast_db(args.blast_db)
    get_model()


def cmd_search(args) -> int:
    tools = _tools(args)
    _check_search_inputs(args, tools)
    cache, tmp = _cache(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    _storage_banner(args, cache)
    work = pp.WorkDir(out, args.keep_intermediates)
    try:
        src = _protein(args, "query", cache)
        query = pp.prepare_protein(src, work.path / "query")
        store = pp.TargetStore(cache, work.path, args.refresh_afdb)
        summary = pp.run_search(query, args.blast_db, args.max_hits, tools, store, out,
                                keep_dir=work.path / "tmalign" if args.keep_intermediates else None)
        meta = run_metadata("search", vars(args), _tool_info(tools))
        meta.update(blast_db=str(Path(args.blast_db).resolve()), blast_db_label=args.blast_db_label,
                    max_hits=args.max_hits, query={**vars(src), "sequence": len(src.sequence)}, summary=summary)
        write_metadata(out / "run_metadata.json", meta)
        pp.log(f"[{src.label}] predicted {summary['n_predicted']}, skipped {summary['n_skipped']}; outputs in {out}")
    finally:
        work.cleanup()
        _cache_footer(cache, tmp)
    return 0


def _read_queries(path: Path) -> list:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise FujisanError(INVALID_INPUT, f"{path}: no rows")
    cols = set(rows[0])
    if not ({"afdb_id", "query_id", "structure"} & cols):
        raise FujisanError(INVALID_INPUT, f"{path}: needs a column 'afdb_id' or 'query_id' "
                                          "(AFDB IDs), or 'query_id' + 'structure' + 'pae' (+ 'fasta') for local files")
    base = path.parent
    out = []
    for r in rows:
        r = {k: (v or "").strip() for k, v in r.items()}
        if r.get("structure"):
            res = lambda p: str((base / p) if p and not Path(p).is_absolute() else p) if p else None
            out.append({"query_id": r.get("query_id") or Path(r["structure"]).stem, "afdb": None,
                        "structure": res(r.get("structure")), "pae": res(r.get("pae")), "fasta": res(r.get("fasta"))})
        else:
            out.append({"query_id": r.get("query_id") or r.get("afdb_id"), "afdb": r.get("afdb_id") or r.get("query_id")})
    return out


def cmd_batch(args) -> int:
    tools = _tools(args)
    _check_search_inputs(args, tools)
    queries = _read_queries(Path(args.queries))
    cache, tmp = _cache(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pp.log(f"{len(queries)} queries.")
    _storage_banner(args, cache)
    work = pp.WorkDir(out, args.keep_intermediates)
    store = pp.TargetStore(cache, work.path, args.refresh_afdb)
    summary_rows = []
    try:
        for i, q in enumerate(queries, start=1):
            qid = "".join(c if c.isalnum() or c in "._-" else "_" for c in q["query_id"])
            qdir = out / qid
            pp.log(f"=== query {i}/{len(queries)}: {qid}")
            row = {"query_id": qid, "status": "", "n_hits": "", "n_predicted": "", "n_skipped": "",
                   "top_target": "", "top_probability": "", "error": ""}
            try:
                if q["afdb"]:
                    src = pp.protein_from_afdb(normalize_afdb_id(q["afdb"]), cache, args.refresh_afdb)
                else:
                    src = pp.protein_from_local(q["structure"], q["pae"], q["fasta"], label=qid)
                query = pp.prepare_protein(src, work.path / "queries" / qid)
                s = pp.run_search(query, args.blast_db, args.max_hits, tools, store, qdir,
                                  keep_dir=work.path / "tmalign" / qid if args.keep_intermediates else None)
                meta = run_metadata("batch", vars(args), _tool_info(tools))
                meta.update(blast_db=str(Path(args.blast_db).resolve()), blast_db_label=args.blast_db_label,
                            max_hits=args.max_hits, query={**vars(src), "sequence": len(src.sequence)}, summary=s)
                write_metadata(qdir / "run_metadata.json", meta)
                row.update(status="ok", **s)
            except FujisanError as exc:
                row.update(status="failed", error=str(exc))
                pp.log(f"[{qid}] FAILED: {exc}")
            except Exception as exc:
                if args.debug:
                    traceback.print_exc()
                row.update(status="failed", error=f"Unexpected error: {type(exc).__name__}: {exc}")
                pp.log(f"[{qid}] FAILED: {row['error']}")
            summary_rows.append(row)
            pp.write_tsv(out / "batch_summary.tsv", summary_rows, list(row))
        meta = run_metadata("batch", vars(args), _tool_info(tools))
        meta.update(blast_db=str(Path(args.blast_db).resolve()), blast_db_label=args.blast_db_label,
                    max_hits=args.max_hits, n_queries=len(queries),
                    n_ok=sum(r["status"] == "ok" for r in summary_rows))
        write_metadata(out / "run_metadata.json", meta)
    finally:
        work.cleanup()
        _cache_footer(cache, tmp)
    return 0


def cmd_explain(args) -> int:
    import pandas as pd

    from . import interpretation as it
    from .models import MODEL_CONFIGS
    if args.model not in MODEL_CONFIGS:
        raise FujisanError(INVALID_INPUT, f"unknown model {args.model!r}")
    sep = "," if str(args.input).endswith(".csv") else "\t"
    # only empty cells are missing: IDs such as "NA" stay text; non-numeric features are skipped later
    df = pd.read_csv(args.input, sep=sep, keep_default_na=False, na_values=[""])
    id_cols = args.id_columns.split(",") if args.id_columns else None
    rename = dict(kv.split("=", 1) for kv in args.rename) if args.rename else None
    try:
        records = it.explain_table(df, args.model, id_cols=id_cols, rename=rename,
                                   reference_probability_col=args.reference_probability,
                                   source={"file": Path(args.input).name})
    except ValueError as exc:
        raise FujisanError(INVALID_INPUT, str(exc)) from exc
    bad = [(r["pair"]["pair_id"], e) for r in records for e in it.validate_record(r)]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    it.write_jsonl(out, records)
    if args.tsv:
        it.records_to_frame(records).to_csv(args.tsv, sep="\t", index=False)
    n_ok = sum(r["status"] == "ok" for r in records)
    pp.log(f"{n_ok} pairs explained, {len(records) - n_ok} skipped (see skip_reason); written: {out}")
    pp.log(f"SHAP scale: {it.SHAP_SCALE} (contributions are log-odds, not probability changes)")
    if bad:
        for pid, e in bad[:10]:
            pp.log(f"INVALID {pid}: {e}")
        raise FujisanError(MODEL_MISMATCH, f"{len(bad)} records failed validation")
    return 0


def cmd_cache_info(args) -> int:
    info = Cache(args.cache_dir).info()
    print(f"Cache directory      {info['cache_dir']}")
    print(f"Number of AFDB entries {info['afdb_entries']}")
    print(f"Structure size       {human_size(info['structure_bytes'])}")
    print(f"PAE size             {human_size(info['pae_bytes'])}")
    print(f"Other files          {human_size(info['other_bytes'])}")
    print(f"Total size           {human_size(info['total_bytes'])}")
    return 0


# --------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fujisan", description="Feature generation and prediction with the updated "
                                "5-feature FUJISAN model (BitScore, Full_SeqID, Dmn_RMSD, Dmn_SeqID, Dmn_TM).",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Run 'fujisan <command> --help' for the options of a command.")
    p.add_argument("--version", action="version", version=f"fujisan {__version__}")
    sub = p.add_subparsers(dest="command", metavar="command")

    pair = sub.add_parser("pair", help="features and FUJISAN probability for one protein pair",
                          description="Five features (BitScore, Full_SeqID, Dmn_RMSD, Dmn_SeqID, Dmn_TM) and the "
                                      "FUJISAN probability for one pair. Each protein is given as an AFDB ID / "
                                      "UniProt accession, or as local structure (PDB or mmCIF) + PAE JSON files.")
    for n in ("1", "2"):
        g = pair.add_argument_group(f"protein {n}")
        g.add_argument(f"--afdb-id{n}", help="UniProt accession or AFDB ID (e.g. Q9HBA0 or AF-Q9HBA0-F1)")
        g.add_argument(f"--fasta{n}", help="FASTA (optional; default: sequence of the structure)")
        g.add_argument(f"--structure{n}", help="structure file (.pdb or .cif), pLDDT in the B-factor column")
        g.add_argument(f"--pae{n}", help="PAE JSON (AlphaFold DB / AlphaFold 2 / ColabFold / AlphaFold 3)")
    pair.add_argument("--output", "-o", required=True, help="output TSV")
    _common(pair)
    pair.set_defaults(func=cmd_pair)

    search = sub.add_parser("search", help="one query -> BLAST -> AFDB -> features -> ranking",
                            description="BLASTP one query against a user-provided database, fetch the AFDB model of "
                                        "each hit, compute the five features and rank the hits by FUJISAN probability.")
    g = search.add_argument_group("query")
    g.add_argument("--query-afdb-id", help="UniProt accession or AFDB ID of the query")
    g.add_argument("--query-fasta", help="query FASTA (local mode, optional)")
    g.add_argument("--query-structure", help="query structure (.pdb / .cif, local mode)")
    g.add_argument("--query-pae", help="query PAE JSON (local mode)")
    _search_opts(search)
    search.add_argument("--output", "-o", required=True, help="output directory")
    _common(search)
    search.set_defaults(func=cmd_search)

    batch = sub.add_parser("batch", help="several queries, one output directory each",
                           description="Run `search` for every query of a TSV. Columns: 'afdb_id' (or 'query_id' "
                                       "holding AFDB IDs); for local files 'query_id', 'structure', 'pae' and "
                                       "optionally 'fasta' (paths relative to the TSV).")
    batch.add_argument("--queries", required=True, help="queries TSV")
    _search_opts(batch)
    batch.add_argument("--output", "-o", required=True, help="output directory")
    _common(batch)
    batch.set_defaults(func=cmd_batch)

    ex = sub.add_parser("explain", help="per-pair feature contributions (SHAP, log-odds) as JSONL",
                        description="Probability, raw score and per-feature SHAP contributions (log-odds) for each "
                                    "pair of a feature table (e.g. predictions.tsv / features.tsv of `pair` or "
                                    "`search`). Uses the same model, feature order and prediction function as "
                                    "prediction. Rows with missing features are reported as skipped, not filled in.")
    ex.add_argument("--input", "-i", required=True, help="feature table (TSV, or CSV if the name ends in .csv)")
    ex.add_argument("--output", "-o", required=True, help="output JSONL (schema fujisan.interpretation)")
    ex.add_argument("--tsv", help="optional flat TSV of the same records")
    ex.add_argument("--model", default="updated", help="model key (default: updated)")
    ex.add_argument("--id-columns", help="query,target ID columns (default: ProteinID1,ProteinID2 or "
                                         "QueryID,TargetID or query_id,target_id)")
    ex.add_argument("--rename", nargs="*", metavar="OLD=NEW",
                    help="rename input columns to model feature names, e.g. Full_Bitscore=BitScore")
    ex.add_argument("--reference-probability", metavar="COLUMN",
                    help="column with a probability computed elsewhere; its difference is recorded")
    ex.add_argument("--debug", action="store_true", help="show Python tracebacks")
    ex.set_defaults(func=cmd_explain)

    ci = sub.add_parser("cache-info", help="show the AFDB cache size")
    ci.add_argument("--cache-dir", type=Path, default=None)
    ci.set_defaults(func=cmd_cache_info)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except FujisanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if getattr(args, "debug", False):
            traceback.print_exc()
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        if getattr(args, "debug", False):
            traceback.print_exc()
        else:
            print("Run again with --debug to see the traceback.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
