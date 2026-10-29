"""Errors with a short, human-readable reason.

``reason`` is one of the fixed categories below; it is what ends up in
``skipped_targets.tsv`` and in CLI error messages. ``detail`` carries the
specifics (file, accession, tool output).
"""


BLAST_NOT_FOUND = "BLAST executable not found"
BLAST_DB_NOT_FOUND = "BLAST database not found"
BLAST_FAILED = "BLAST failed"
NO_BLAST_HIT = "No BLAST hit between the two sequences"
TMALIGN_NOT_FOUND = "TM-align executable not found"
NO_ACCESSION = "No UniProt accession could be parsed"
AFDB_UNAVAILABLE = "No AFDB entry"
AFDB_NETWORK = "AFDB download failed"
STRUCTURE_UNAVAILABLE = "Structure unavailable"
PAE_UNAVAILABLE = "PAE unavailable"
INVALID_PAE = "Invalid PAE format"
INVALID_STRUCTURE = "Invalid structure file"
LENGTH_MISMATCH = "Sequence/structure length mismatch"
NO_CONFIDENT_RESIDUES = "No residues with pLDDT >= 70"
SEGMENTATION_FAILED = "Domain segmentation failed"
NO_DOMAIN = "No domain >= 30 residues"
TMALIGN_FAILED = "TM-align failed"
NO_DOMAIN_PAIR = "No valid domain pair"
MODEL_MISMATCH = "Model feature mismatch"
INVALID_INPUT = "Invalid input"


class FujisanError(Exception):
    def __init__(self, reason, detail=""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)
