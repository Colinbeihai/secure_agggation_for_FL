import os
import sys
import time
import subprocess
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any

import yaml
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = BASE_DIR / "config.yaml"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(cfg: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def bench_csv_path(protocol: str, num_clients: int, dropout: float, run_idx: int) -> Path:
    name = f"bench_{protocol}_C{num_clients}_D{dropout}_run{run_idx}.csv"
    return LOGS_DIR / name


def run_one(protocol: str, num_clients: int, dropout: float, run_idx: int) -> Path:
    cfg = load_yaml(CONFIG_PATH)
    cfg["secure_aggregation"] = protocol
    cfg["num_clients"] = int(num_clients)
    cfg["dropout_prob"] = float(dropout)
    cfg.setdefault("benchmark", {})
    cfg["benchmark"]["enable"] = True
    csv_path = bench_csv_path(protocol, num_clients, dropout, run_idx)
    cfg["benchmark"]["csv_path"] = str(csv_path.as_posix())

    # Optional: keep scsecagg params as in config; or set defaults if absent
    scsec = cfg.setdefault("scsecagg", {})
    scsec.setdefault("num_servers", 6)
    scsec.setdefault("read_threshold", 4)
    scsec.setdefault("storage_factor", 2)

    save_yaml(cfg, CONFIG_PATH)

    print(f"[RUN] protocol={protocol} num_clients={num_clients} dropout={dropout} run={run_idx}")
    start = time.time()
    try:
        subprocess.run([sys.executable, "main.py"], cwd=str(BASE_DIR), check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] main.py failed: {e}")
    dur = time.time() - start
    print(f"[DONE] elapsed {dur:.2f}s -> {csv_path}")
    return csv_path


def summarize(csv_files: List[Path]) -> pd.DataFrame:
    rows = []
    for p in csv_files:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}")
            continue
        if df.empty:
            continue
        df = df[df["aggregated"] == True].copy()
        if df.empty:
            continue
        total_time = (pd.to_numeric(df["train_time_s"], errors="coerce").fillna(0.0)
                      + pd.to_numeric(df["agg_time_s"], errors="coerce").fillna(0.0)).sum()
        # parse meta from filename
        name = p.name.replace(".csv", "")
        m = re.match(r"bench_(\w+)_C(\d+)_D([0-9.]+)_run(\d+)$", name)
        if m:
            protocol = m.group(1)
            num_clients = int(m.group(2))
            dropout = float(m.group(3))
            run = int(m.group(4))
        else:
            # fallback minimal parse
            parts = name.split("_")
            protocol = parts[1] if len(parts) > 1 else "unknown"
            num_clients = -1
            dropout = -1.0
            run = -1
        rows.append({
            "protocol": protocol,
            "num_clients": num_clients,
            "dropout_prob": dropout,
            "run": run,
            "total_time_s": float(total_time)
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    summary = (res.groupby(["protocol", "num_clients", "dropout_prob"], as_index=False)
                 .agg(total_time_mean=("total_time_s", "mean"),
                      total_time_std=("total_time_s", "std"),
                      total_time_count=("total_time_s", "count")))
    # 95% confidence interval (mean ± 1.96*std/sqrt(n))
    summary["total_time_ci95"] = 1.96 * summary["total_time_std"] / summary["total_time_count"].clip(lower=1).pow(0.5)
    return summary


def plot_curves(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("[WARN] empty summary, skip plotting")
        return
    # Consistent colors/markers per protocol
    proto_colors = {
        "fastsecagg": "tab:blue",
        "secagg_plus": "tab:orange",
        "scsecagg": "tab:green",
    }
    proto_markers = {
        "fastsecagg": "o",
        "secagg_plus": "s",
        "scsecagg": "^",
    }
    # Plot total_time vs num_clients (lines by protocol), faceted by dropout_prob
    for drop in sorted(summary["dropout_prob"].unique()):
        sub = summary[summary["dropout_prob"] == drop]
        plt.figure(figsize=(7, 5))
        for proto in sorted(sub["protocol"].unique()):
            ss = sub[sub["protocol"] == proto].sort_values("num_clients")
            x = ss["num_clients"].to_numpy()
            y = ss["total_time_mean"].to_numpy()
            ci = ss["total_time_ci95"].fillna(0.0).to_numpy()
            plt.plot(x, y, marker=proto_markers.get(proto, "o"), color=proto_colors.get(proto), label=proto)
            # shaded 95% CI band instead of std errorbar
            plt.fill_between(x, y - ci, y + ci, alpha=0.15, color=proto_colors.get(proto))
        plt.title(f"Total time vs Clients (dropout={drop})")
        plt.xlabel("num_clients")
        plt.ylabel("total_time_mean [s]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = PLOTS_DIR / f"total_time_vs_clients_drop{drop}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()

    # Plot total_time vs dropout_prob (lines by protocol), faceted by num_clients
    for nc in sorted(summary["num_clients"].unique()):
        sub = summary[summary["num_clients"] == nc]
        plt.figure(figsize=(7, 5))
        for proto in sorted(sub["protocol"].unique()):
            ss = sub[sub["protocol"] == proto].sort_values("dropout_prob")
            x = ss["dropout_prob"].to_numpy()
            y = ss["total_time_mean"].to_numpy()
            ci = ss["total_time_ci95"].fillna(0.0).to_numpy()
            plt.plot(x, y, marker=proto_markers.get(proto, "o"), color=proto_colors.get(proto), label=proto)
            plt.fill_between(x, y - ci, y + ci, alpha=0.15, color=proto_colors.get(proto))
        plt.title(f"Total time vs Dropout (clients={nc})")
        plt.xlabel("dropout_prob")
        plt.ylabel("total_time_mean [s]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = PLOTS_DIR / f"total_time_vs_dropout_clients{nc}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()

    # Additional: grouped bar charts to directly compare protocols
    # For each dropout, group by num_clients with bars for protocols
    for drop in sorted(summary["dropout_prob"].unique()):
        sub = summary[summary["dropout_prob"] == drop]
        clients = sorted(sub["num_clients"].unique())
        protos = sorted(sub["protocol"].unique())
        width = 0.8 / max(1, len(protos))
        x_positions = range(len(clients))
        plt.figure(figsize=(8, 5))
        for i, proto in enumerate(protos):
            means = []
            for nc in clients:
                row = sub[(sub["protocol"] == proto) & (sub["num_clients"] == nc)]
                means.append(row["total_time_mean"].iloc[0] if not row.empty else float("nan"))
            xs = [x + i * width for x in x_positions]
            plt.bar(xs, means, width=width, label=proto, color=proto_colors.get(proto))
        plt.xticks([x + (len(protos)-1)*width/2 for x in x_positions], clients)
        plt.xlabel("num_clients")
        plt.ylabel("total_time_mean [s]")
        plt.title(f"Protocol comparison (dropout={drop})")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        out = PLOTS_DIR / f"protocol_compare_by_clients_drop{drop}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()


def collect_existing_csvs(pattern: str) -> List[Path]:
    from glob import glob
    paths = [Path(p) for p in glob(str((LOGS_DIR / pattern).as_posix()))]
    return paths


def main():
    parser = argparse.ArgumentParser(description="Run FL protocol experiments and/or plot existing results.")
    parser.add_argument("--plot-only", action="store_true", help="Only summarize and plot existing CSVs in results/logs.")
    parser.add_argument("--glob", default="bench_*.csv", help="Glob pattern under results/logs to select CSVs for plotting.")
    parser.add_argument("--protocols", nargs="*", default=["fastsecagg", "secagg_plus", "scsecagg"], help="Protocols to run when not in plot-only mode.")
    parser.add_argument("--clients", nargs="*", type=int, default=[10, 20, 40], help="Client counts to run when not in plot-only mode.")
    parser.add_argument("--dropouts", nargs="*", type=float, default=[0.0, 0.2, 0.4], help="Dropout probabilities to run when not in plot-only mode.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per setting when not in plot-only mode.")
    args = parser.parse_args()

    ensure_dirs()

    if args.plot_only:
        csv_files = collect_existing_csvs(args.glob)
        print(f"[INFO] plotting from {len(csv_files)} CSV files matching pattern: {args.glob}")
        summary = summarize(csv_files)
        out_summary = LOGS_DIR / "summary.csv"
        if not summary.empty:
            summary.to_csv(out_summary, index=False)
            print(f"[OK] summary -> {out_summary}")
            plot_curves(summary)
            print(f"[OK] plots -> {PLOTS_DIR}")
        else:
            print("[WARN] no data summarized; check your pattern or logs directory.")
        return

    # Not plot-only: run experiments then plot
    protocols = args.protocols
    client_grid = args.clients
    dropout_grid = args.dropouts
    repeats = args.repeats

    orig = load_yaml(CONFIG_PATH)
    csv_files: List[Path] = []
    try:
        for proto in protocols:
            for nc in client_grid:
                for dp in dropout_grid:
                    for r in range(1, repeats + 1):
                        csv_files.append(run_one(proto, nc, dp, r))
    finally:
        save_yaml(orig, CONFIG_PATH)

    summary = summarize(csv_files)
    out_summary = LOGS_DIR / "summary.csv"
    if not summary.empty:
        summary.to_csv(out_summary, index=False)
        print(f"[OK] summary -> {out_summary}")
    else:
        print("[WARN] no data summarized")

    plot_curves(summary)
    print(f"[OK] plots -> {PLOTS_DIR}")


if __name__ == "__main__":
    main()


