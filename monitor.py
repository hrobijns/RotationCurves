"""
Convergence monitor: watches a PySR hall_of_fame.csv and logs
(wall_time, best_loss, best_complexity, best_equation) to convergence.csv
every time the HOF file changes.

Usage:
    python monitor.py <output_directory>

Example:
    python monitor.py outputs/SPARC/production/velocity1param
"""
import sys
import os
import time
import csv
import glob
import datetime


def find_hof(output_dir: str) -> str | None:
    """Find the hall_of_fame.csv, handling PySR's timestamped subdirectory."""
    direct = os.path.join(output_dir, "hall_of_fame.csv")
    if os.path.exists(direct):
        return direct
    # PySR creates a timestamped subdirectory
    matches = glob.glob(os.path.join(output_dir, "*", "hall_of_fame.csv"))
    if matches:
        return max(matches, key=os.path.getmtime)
    return None


def read_best(hof_path: str) -> tuple[float, int, str] | None:
    """Return (best_loss, best_complexity, best_equation) — lowest loss row."""
    try:
        with open(hof_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        best = min(rows, key=lambda r: float(r["Loss"]))
        return float(best["Loss"]), int(best["Complexity"]), best["Equation"].strip()
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python monitor.py <output_directory>")
        sys.exit(1)

    output_dir = sys.argv[1]
    convergence_path = os.path.join(output_dir, "convergence.csv")

    print(f"Monitoring: {output_dir}")
    print(f"Logging to: {convergence_path}")
    print("Waiting for hall_of_fame.csv …")

    os.makedirs(output_dir, exist_ok=True)

    last_mtime = None
    iteration = 0
    hof_path = None

    with open(convergence_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wall_time", "iteration", "best_loss", "best_complexity", "best_equation"])

    while True:
        # (re)find HOF in case the timestamped subdir appears after startup
        if hof_path is None or not os.path.exists(hof_path):
            hof_path = find_hof(output_dir)

        if hof_path is not None:
            try:
                mtime = os.path.getmtime(hof_path)
            except OSError:
                mtime = None

            if mtime != last_mtime and mtime is not None:
                result = read_best(hof_path)
                if result is not None:
                    best_loss, best_complexity, best_eq = result
                    iteration += 1
                    now = datetime.datetime.now().isoformat(timespec="seconds")
                    print(f"[{now}] iter ~{iteration:4d} | loss={best_loss:.6g} | c{best_complexity} | {best_eq}")
                    with open(convergence_path, "a", newline="") as f:
                        csv.writer(f).writerow([now, iteration, best_loss, best_complexity, best_eq])
                last_mtime = mtime

        time.sleep(5)


if __name__ == "__main__":
    main()
