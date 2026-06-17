import csv
import os
import re
from collections import defaultdict


LOGS = [
    "logs/train_formal.log",
    "logs/train_formal_resume.log",
    "logs/train_resume_qframefix_manual.log",
    "logs/train_resume_qframefix_fix2.log",
]

PAT_EPOCH = re.compile(r"Epoch\s+(\d+)\s+Evaluation:")
PAT_METRIC = re.compile(
    r"\s+(AJ|OA|<4px|<avg|average_jaccard|occlusion_accuracy|average_pts_within_thresh):\s*([0-9.]+)"
)
PAT_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def main() -> None:
    entries = defaultdict(list)
    seq = 0

    for path in LOGS:
        if not os.path.exists(path):
            continue
        raw = open(path, "rb").read().replace(b"\r", b"\n").decode("utf-8", "ignore")
        lines = raw.split("\n")
        for i, line in enumerate(lines):
            m = PAT_EPOCH.search(line)
            if not m:
                continue
            seq += 1
            row = {
                "epoch": int(m.group(1)),
                "log": os.path.basename(path),
                "seq": seq,
            }
            ts = PAT_TS.search(line)
            row["ts"] = ts.group(1) if ts else ""

            for j in range(i + 1, min(i + 200, len(lines))):
                mm = PAT_METRIC.search(lines[j])
                if mm:
                    row[mm.group(1)] = float(mm.group(2))

            if "AJ" not in row and "average_jaccard" in row:
                row["AJ"] = row["average_jaccard"]
            if "OA" not in row and "occlusion_accuracy" in row:
                row["OA"] = row["occlusion_accuracy"]
            if "<avg" not in row and "average_pts_within_thresh" in row:
                row["<avg"] = row["average_pts_within_thresh"]

            entries[row["epoch"]].append(row)

    rows = []
    for epoch in sorted(entries):
        cand = sorted(
            entries[epoch],
            key=lambda item: (
                sum(k in item for k in ("AJ", "OA", "<4px", "<avg")),
                item.get("ts", ""),
                item.get("seq", 0),
            ),
        )
        rows.append(cand[-1])

    if not rows:
        raise SystemExit("No evaluation rows found")

    metrics = ["AJ", "OA", "<4px", "<avg"]
    ranks = {metric: {} for metric in metrics}
    for metric in metrics:
        ordered = sorted(
            [row for row in rows if metric in row],
            key=lambda item: item[metric],
            reverse=True,
        )
        for idx, row in enumerate(ordered, 1):
            ranks[metric][row["epoch"]] = idx

    best = {
        metric: max([row for row in rows if metric in row], key=lambda item: item[metric])
        for metric in metrics
    }

    csv_path = "logs/metrics_epoch_table.csv"
    md_path = "logs/metrics_epoch_table.md"
    summary_path = "logs/metrics_latest_summary.txt"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "AJ",
                "OA",
                "<4px",
                "<avg",
                "AJ_rank",
                "OA_rank",
                "<4px_rank",
                "<avg_rank",
                "best_AJ",
                "best_OA",
                "best_<4px",
                "best_<avg",
                "source_log",
                "timestamp",
            ]
        )

        for row in rows:
            epoch = row["epoch"]
            writer.writerow(
                [
                    epoch,
                    f"{row.get('AJ', float('nan')):.4f}",
                    f"{row.get('OA', float('nan')):.4f}",
                    f"{row.get('<4px', float('nan')):.4f}",
                    f"{row.get('<avg', float('nan')):.4f}",
                    ranks["AJ"].get(epoch, ""),
                    ranks["OA"].get(epoch, ""),
                    ranks["<4px"].get(epoch, ""),
                    ranks["<avg"].get(epoch, ""),
                    1 if epoch == best["AJ"]["epoch"] else 0,
                    1 if epoch == best["OA"]["epoch"] else 0,
                    1 if epoch == best["<4px"]["epoch"] else 0,
                    1 if epoch == best["<avg"]["epoch"] else 0,
                    row.get("log", ""),
                    row.get("ts", ""),
                ]
            )

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Epoch Metrics Table\n\n")
        handle.write("- 标记: ⭐ 表示该指标全历史最佳\n\n")
        handle.write(
            "| Epoch | AJ | OA | <4px | <avg | AJ排位 | OA排位 | <4px排位 | <avg排位 | 来源日志 | 时间 |\n"
        )
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n")

        for row in rows:
            epoch = row["epoch"]

            def cell(metric: str) -> str:
                value = f"{row.get(metric, float('nan')):.4f}"
                if best[metric]["epoch"] == epoch:
                    value += " ⭐"
                return value

            handle.write(
                f"| {epoch} | {cell('AJ')} | {cell('OA')} | {cell('<4px')} | {cell('<avg')} | "
                f"{ranks['AJ'].get(epoch, '-')} | {ranks['OA'].get(epoch, '-')} | "
                f"{ranks['<4px'].get(epoch, '-')} | {ranks['<avg'].get(epoch, '-')} | "
                f"{row.get('log', '-')} | {row.get('ts', '-')} |\n"
            )

    latest = rows[-1]
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"latest_epoch={latest['epoch']}\n")
        for metric in metrics:
            handle.write(f"latest_{metric}={latest.get(metric)}\n")
            handle.write(f"best_{metric}_epoch={best[metric]['epoch']}\n")
            handle.write(f"best_{metric}_value={best[metric][metric]}\n")

    print("UPDATED", latest["epoch"])


if __name__ == "__main__":
    main()

