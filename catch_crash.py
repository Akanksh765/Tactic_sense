"""
catch_crash.py  —  TacticSense Streamlit Crash Catcher
=======================================================
Launches `streamlit run app.py` as a subprocess, streams stdout + stderr
to the console in real-time, and simultaneously writes every line to
crash_report.txt.  If the process exits with a non-zero code a large,
hard-to-miss warning block is printed so you know exactly where to look.

Usage
-----
    python catch_crash.py [extra streamlit args...]

Example (custom port):
    python catch_crash.py --server.port 8502
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Reconfigure stdout/stderr to UTF-8 to prevent cp1252/UnicodeEncodeError on Windows
try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ── Configuration ────────────────────────────────────────────────────────────
APP_TARGET    = "app.py"          # Streamlit entry-point (do not change)
REPORT_FILE   = Path("crash_report.txt")
SEPARATOR     = "-" * 72

# ANSI colour helpers (no external deps; Windows 10+ / WT support these)
RED     = "\033[91m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
BOLD    = "\033[1m"
RESET   = "\033[0m"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _print_banner(report_path: Path) -> None:
    """Pretty header so you know the wrapper is active."""
    print(f"\n{CYAN}{BOLD}{SEPARATOR}")
    print("  TacticSense  ·  Streamlit Crash Catcher  ·  catch_crash.py")
    print(f"  Started : {_now()}")
    print(f"  Target  : streamlit run {APP_TARGET}")
    print(f"  Report  : {report_path.resolve()}")
    print(f"{SEPARATOR}{RESET}\n")


def _print_crash_warning(report_path: Path, exit_code: int) -> None:
    """Loud, impossible-to-miss terminal block on a non-zero exit."""
    border = "#" * 72
    msg = [
        "",
        f"{RED}{BOLD}{border}",
        f"# {'':<68} #",
        f"# {'!!! STREAMLIT PROCESS CRASHED !!!':^68} #",
        f"# {'':<68} #",
        f"#   Exit code  : {exit_code:<53} #",
        f"#   Timestamp  : {_now():<53} #",
        f"# {'':<68} #",
        f"#   Open the crash report for the full traceback:{'':<21} #",
        f"#   {str(report_path.resolve()):<66} #",
        f"# {'':<68} #",
        f"#   Tip: look for lines starting with 'Traceback' or 'Error'{'':<10} #",
        f"# {'':<68} #",
        f"{border}{RESET}",
        "",
    ]
    print("\n".join(msg))


def _print_clean_exit(exit_code: int) -> None:
    print(
        f"\n{GREEN}{BOLD}✔  Streamlit exited cleanly "
        f"(code {exit_code}) at {_now()}.{RESET}\n"
    )


# ── Core logic ───────────────────────────────────────────────────────────────

def _stream_pipe(pipe, report_fh, tag: str, lock: threading.Lock) -> None:
    """
    Drain *pipe* line-by-line.
    • Print each line to stdout (with an optional stderr prefix).
    • Write every line to the open report file handle.
    Thread-safe via *lock*.
    """
    prefix = f"{YELLOW}[STDERR]{RESET} " if tag == "stderr" else ""
    try:
        for raw_line in iter(pipe.readline, b""):
            line = raw_line.decode(errors="replace").rstrip("\r\n")
            with lock:
                # Console — colour stderr lines for instant visibility
                print(f"{prefix}{line}")
                # Report file — plain text, no ANSI
                report_fh.write(f"[{tag.upper()}] {line}\n")
                report_fh.flush()
    except Exception as exc:  # pragma: no cover
        with lock:
            print(f"{RED}[catch_crash] pipe read error: {exc}{RESET}")


def run() -> int:
    """Entry-point.  Returns the child's exit code."""
    # Enable ANSI on Windows (≥ Win10 build 14393 supports VT100)
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

    # Build the command: honour any extra args passed to *this* script
    extra_args = sys.argv[1:]
    cmd = [sys.executable, "-m", "streamlit", "run", APP_TARGET] + extra_args

    # Open the report file (overwrite each run so it stays fresh)
    REPORT_FILE.write_text(
        f"TacticSense Crash Report\n"
        f"Generated  : {_now()}\n"
        f"Command    : {' '.join(cmd)}\n"
        f"{SEPARATOR}\n\n",
        encoding="utf-8",
    )

    _print_banner(REPORT_FILE)
    print(f"{CYAN}▶  Executing: {' '.join(cmd)}{RESET}\n")

    t_start = time.monotonic()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Keep the child in the same process group so Ctrl-C propagates
            close_fds=(sys.platform != "win32"),
        )
    except FileNotFoundError:
        msg = (
            f"\n{RED}{BOLD}[catch_crash] ERROR: 'streamlit' is not installed "
            f"or not on PATH.\n"
            f"Try:  pip install streamlit{RESET}\n"
        )
        print(msg)
        REPORT_FILE.open("a", encoding="utf-8").write(msg)
        return 1

    lock = threading.Lock()

    with REPORT_FILE.open("a", encoding="utf-8") as report_fh:
        # Drain stdout and stderr concurrently so neither blocks the other
        t_out = threading.Thread(
            target=_stream_pipe,
            args=(proc.stdout, report_fh, "stdout", lock),
            daemon=True,
        )
        t_err = threading.Thread(
            target=_stream_pipe,
            args=(proc.stderr, report_fh, "stderr", lock),
            daemon=True,
        )
        t_out.start()
        t_err.start()

        try:
            proc.wait()
        except KeyboardInterrupt:
            print(f"\n{YELLOW}[catch_crash] Keyboard interrupt — stopping Streamlit…{RESET}")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        t_out.join()
        t_err.join()

        elapsed = time.monotonic() - t_start
        summary = (
            f"\n{SEPARATOR}\n"
            f"Exit code  : {proc.returncode}\n"
            f"Elapsed    : {elapsed:.1f}s\n"
            f"Finished   : {_now()}\n"
        )
        report_fh.write(summary)

    # ── Final verdict ────────────────────────────────────────────────────────
    if proc.returncode != 0:
        _print_crash_warning(REPORT_FILE, proc.returncode)
    else:
        _print_clean_exit(proc.returncode)

    return proc.returncode


if __name__ == "__main__":
    sys.exit(run())
