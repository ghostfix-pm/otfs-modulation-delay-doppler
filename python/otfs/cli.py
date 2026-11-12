"""Command line interface for the OTFS simulation study.

    python -m otfs                       # the whole study: toy example, figures, BER curve
    python -m otfs ber --frames 40       # only the Monte-Carlo BER curve
    python -m otfs toy                   # only the 2x4 illustrative frame

Legacy invocations keep working: an argument list without a sub-command is
routed to ``all``, so ``python python/otfs_sim.py --frames 3000`` still runs
the toy example and the BER curve as it always did.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import figures
from .sim import ber_curve, default_snr_db, rayleigh_qpsk_ber, toy_example

__all__ = ["build_parser", "grid", "main", "positive_int", "snr_list"]

COMMANDS = ("all", "ber", "toy")


def positive_int(text: str) -> int:
    """argparse type: an integer >= 1 (``--frames 0`` used to divide by zero)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def grid(text: str) -> tuple[int, int]:
    """argparse type: a frame size such as ``4x8`` or ``16x16``."""
    parts = text.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected NxM, got {text!r}")
    try:
        n, m = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected NxM with integers, got {text!r}") from None
    if n < 1 or m < 1:
        raise argparse.ArgumentTypeError(f"grid sides must be >= 1, got {text!r}")
    return n, m


def snr_list(text: str) -> list[float]:
    """argparse type: comma separated SNR values in dB, e.g. ``0,2,4``."""
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a number, got {chunk!r}") from None
    if not values:
        raise argparse.ArgumentTypeError("expected at least one SNR value")
    return values


def _common_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=3, help="random seed (default: 3)")
    parser.add_argument("--out", type=Path, default=None,
                        help="figure directory (default: <checkout>/docs/figures or ./figures)")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    return parser


def _ber_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--frames", type=positive_int, default=400,
                        help="frames per SNR point (default: 400)")
    parser.add_argument("--paths", type=positive_int, default=3,
                        help="propagation paths per frame (default: 3)")
    parser.add_argument("--grid", type=grid, default=(4, 8), metavar="NxM",
                        help="frame size, Doppler x delay bins (default: 4x8)")
    parser.add_argument("--snr", type=snr_list, default=None, metavar="DB[,DB...]",
                        help="SNR grid in dB (default: 0,2,...,20)")
    parser.add_argument("--method", choices=("sparse", "dense"), default="sparse",
                        help="message-passing implementation (default: sparse)")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--csv", type=Path, default=None, help="write the result as CSV")
    parser.add_argument("--quiet", action="store_true", help="do not print the table")
    return parser


def build_parser() -> argparse.ArgumentParser:
    common, ber = _common_options(), _ber_options()
    parser = argparse.ArgumentParser(
        prog="otfs",
        description="OTFS delay-Doppler channel modelling and message-passing detection",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("all", parents=[common, ber],
                   help="the whole study (default when no sub-command is given)")
    sub.add_parser("toy", parents=[common],
                   help="the 2x4 illustrative frame (fig01, fig02)")
    sub.add_parser("ber", parents=[common, ber],
                   help="Monte-Carlo BER curve (fig03)")
    return parser


def _with_default_command(argv) -> list[str]:
    """Route a bare option list to the ``all`` sub-command (legacy invocations)."""
    argv = [str(token) for token in argv]
    if any(token in COMMANDS for token in argv):
        return argv
    if any(token in ("-h", "--help") for token in argv):
        return argv
    return ["all", *argv]


def _write_csv(points, csv_path) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["snr_db", "errors", "bits", "ber", "upper_bound_95"])
        for point in points:
            writer.writerow([f"{point.snr_db:.6g}", point.errors, point.bits,
                             f"{point.ber:.6e}" if point.errors else "",
                             f"{point.upper_bound:.6e}" if not point.errors else ""])
    return csv_path


def _run_toy(args, out_dir) -> dict:
    rng = np.random.default_rng(args.seed)
    result = toy_example(rng, out_dir=None if args.no_figures else out_dir)
    print("toy example - noiseless MP detection, symbol errors:", result["symbol_errors"])
    return {"seed": args.seed, "symbol_errors": result["symbol_errors"]}


def _run_ber(args, out_dir) -> dict:
    N, M = args.grid
    snr = args.snr if args.snr is not None else default_snr_db()
    result = ber_curve(frames=args.frames, snr_db=snr, N=N, M=M, n_paths=args.paths,
                       seed=args.seed, method=args.method,
                       progress=None if args.quiet else (lambda p: print(p.format())))
    reference = rayleigh_qpsk_ber(result.snr_db)
    if not args.quiet:
        print("SNR  OTFS/MP     QPSK Rayleigh reference")
        for point, ref in zip(result.points, reference):
            print(f"{point.snr_db:4.0f} dB  {point.ber:.3e}   {ref:.3e}")
    if not args.no_figures:
        figures.save_ber_figure(out_dir / "fig03_ber_mp.png", result, reference=reference)
    if args.csv is not None:
        _write_csv(result.points, args.csv)
    payload = result.as_dict()
    if args.json:
        print(json.dumps(payload, indent=1))
    return payload


def main(argv=None) -> int:
    """Entry point; returns a process exit code."""
    parser = build_parser()
    raw = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(_with_default_command(raw))
    out_dir = args.out if args.out is not None else figures.default_output_dir()
    try:
        if args.command == "toy":
            _run_toy(args, out_dir)
        else:
            if args.command == "all":
                _run_toy(args, out_dir)
            _run_ber(args, out_dir)
    except (ValueError, FileNotFoundError) as exc:
        print(f"otfs: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
