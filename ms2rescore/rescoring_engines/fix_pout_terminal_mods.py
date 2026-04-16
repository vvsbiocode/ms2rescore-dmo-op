import argparse
from pathlib import Path


def convert_terminal_mod_notation(peptide: str) -> str:
    """Convert Percolator-style terminal UNIMOD tags to ProForma-style terminal tags."""
    original = peptide

    while True:
        updated = original

        # N-term: n[UNIMOD:x]PEPTIDE -> [UNIMOD:x]-PEPTIDE
        updated = _replace_nterm_unimod(updated)

        # C-term: PEPTIDEc[UNIMOD:x] -> PEPTIDE-[UNIMOD:x]
        updated = _replace_cterm_unimod(updated)

        if updated == original:
            return updated
        original = updated


def _replace_nterm_unimod(peptide: str) -> str:
    marker = "n[UNIMOD:"
    start = peptide.find(marker)
    if start == -1:
        return peptide

    end = peptide.find("]", start)
    if end == -1:
        return peptide

    replacement = peptide[start + 1 : end + 1] + "-"
    return peptide[:start] + replacement + peptide[end + 1 :]


def _replace_cterm_unimod(peptide: str) -> str:
    marker = "c[UNIMOD:"
    start = peptide.find(marker)
    if start == -1:
        return peptide

    end = peptide.find("]", start)
    if end == -1:
        return peptide

    replacement = "-" + peptide[start + 1 : end + 1]
    return peptide[:start] + replacement + peptide[end + 1 :]


def rewrite_pout_file(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as src:
        lines = src.readlines()

    if not lines:
        raise ValueError(f"Input file is empty: {input_path}")

    header = lines[0].rstrip("\n\r").split("\t")
    try:
        peptide_idx = header.index("peptide")
    except ValueError as exc:
        try:
            peptide_idx = header.index("Peptide")
        except ValueError:
            raise ValueError("Could not find a `peptide` column in the POUT file.") from exc

    output_lines = [lines[0]]

    for line in lines[1:]:
        stripped = line.rstrip("\n\r")
        newline = line[len(stripped) :]
        if not stripped:
            output_lines.append(line)
            continue

        fields = stripped.split("\t")
        if len(fields) > peptide_idx:
            fields[peptide_idx] = convert_terminal_mod_notation(fields[peptide_idx])

        output_lines.append("\t".join(fields) + newline)

    with output_path.open("w", encoding="utf-8", newline="") as dst:
        dst.writelines(output_lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite the peptide column of a Percolator .pout file so terminal "
            "UNIMOD tags use ProForma-style terminal notation."
        )
    )
    parser.add_argument("input_pout", type=Path, help="Path to input .pout file")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_pout",
        type=Path,
        help="Path to output .pout file. Defaults to <input>.terminal-fixed.pout",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_pout = args.input_pout
    output_pout = args.output_pout or input_pout.with_name(
        input_pout.stem + ".terminal-fixed" + input_pout.suffix
    )

    rewrite_pout_file(input_pout, output_pout)


if __name__ == "__main__":
    main()
