import argparse
from pathlib import Path


def convert_terminal_mod_notation(peptide: str) -> str:
    """Convert ProForma-style terminal UNIMOD tags to Percolator-style terminal tags."""
    original = peptide

    while True:
        updated = original

        # N-term: [UNIMOD:x]-PEPTIDE -> n[UNIMOD:x]PEPTIDE
        updated = _replace_nterm_unimod(updated)

        # C-term: PEPTIDE-[UNIMOD:x] -> PEPTIDEc[UNIMOD:x]
        updated = _replace_cterm_unimod(updated)

        if updated == original:
            return updated
        original = updated


def _replace_nterm_unimod(peptide: str) -> str:
    marker = "[UNIMOD:"
    start = peptide.find(marker)
    if start == -1:
        return peptide

    end = peptide.find("]", start)
    if end == -1 or end + 1 >= len(peptide) or peptide[end + 1] != "-":
        return peptide

    replacement = "n" + peptide[start : end + 1]
    return peptide[:start] + replacement + peptide[end + 2 :]


def _replace_cterm_unimod(peptide: str) -> str:
    marker = "-[UNIMOD:"
    start = peptide.find(marker)
    if start == -1:
        return peptide

    end = peptide.find("]", start + 1)
    if end == -1:
        return peptide

    replacement = "c" + peptide[start + 1 : end + 1]
    return peptide[:start] + replacement + peptide[end + 1 :]


def rewrite_pin_file(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as src:
        lines = src.readlines()

    if not lines:
        raise ValueError(f"Input file is empty: {input_path}")

    header = lines[0].rstrip("\n\r").split("\t")
    try:
        peptide_idx = header.index("Peptide")
    except ValueError as exc:
        try:
            peptide_idx = header.index("peptide")
        except ValueError:
            raise ValueError("Could not find a `Peptide` column in the PIN file.") from exc

    output_lines = [lines[0]]

    for line in lines[1:]:
        stripped = line.rstrip("\n\r")
        newline = line[len(stripped) :]
        if not stripped:
            output_lines.append(line)
            continue

        fields = stripped.split("\t")
        if len(fields) > peptide_idx and fields[0] != "DefaultDirection":
            fields[peptide_idx] = "-"+convert_terminal_mod_notation(fields[peptide_idx])+"-" #placeholders for flanking aa

        output_lines.append("\t".join(fields) + newline)

    with output_path.open("w", encoding="utf-8", newline="") as dst:
        dst.writelines(output_lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite the Peptide column of a Percolator .pin file so terminal "
            "UNIMOD ProForma tags use n[...] / c[...] notation."
        )
    )
    parser.add_argument("input_pin", type=Path, help="Path to input .pin file")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_pin",
        type=Path,
        help="Path to output .pin file. Defaults to <input>.terminal-fixed.pin",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_pin = args.input_pin
    output_pin = args.output_pin or input_pin.with_name(
        input_pin.stem + ".terminal-fixed" + input_pin.suffix
    )

    rewrite_pin_file(input_pin, output_pin)


if __name__ == "__main__":
    main()
