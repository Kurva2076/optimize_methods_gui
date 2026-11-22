#!/usr/bin/env python3
"""Сводный report.md по всем прогонам hpo_pointnet.py (задача №7).

Читает <runs>/*/best.json и пишет по таблице на класс, сравнение xyz и xyz_hag
(грунт, растительность и любые другие цели, где есть оба прогона) и данные о весах классов.
Пример: python summary_report.py runs --out runs/report.md
"""
import argparse
import json
from pathlib import Path

from hpo_pointnet import BASELINE_MARGIN, TARGET_TITLES, TARGETS


def fmt(value):
    return f"{value:.4f}" if isinstance(value, float) else "—"


def build(runs_dir):
    runs = [json.loads(p.read_text(encoding="utf-8")) | {"dir": p.parent.name}
            for p in sorted(Path(runs_dir).glob("*/best.json"))]
    if not runs:
        raise SystemExit(f"Нет прогонов с best.json в {runs_dir}")
    lines = ["# Сводный отчёт: подбор PointNet++ по классам", "",
             "Метрики — F1, Precision, Recall целевого класса на Val (лучшая эпоха лучшего trial). "
             "Baseline — F1 константного предсказания «всё = 1» на тех же Val-патчах.", ""]
    for target in TARGETS:
        rows = [r for r in runs if r["target"] == target]
        lines += [f"## {TARGET_TITLES[target]} (`{target}`, классы {list(TARGETS[target])})", ""]
        if not rows:
            lines += ["Прогонов нет.", ""]
            continue
        lines += ["| Прогон | Признаки | Loss | Trials (успешных) | Trial | F1 | Precision | Recall | Baseline F1 | "
                  "Веса классов (фон, цель) | Seed | Эпох | Гиперпараметры |",
                  "|:---|:---|:---|---:|---:|---:|---:|---:|---:|:---|---:|---:|:---|"]
        for r in rows:
            params = json.dumps(r.get("params", {}), ensure_ascii=False, sort_keys=True)
            lines.append(f"| {r['dir']} | `{r['features']}` | {r.get('loss', 'ce')} | {r['trials']} ({r['complete']}) | "
                         f"{r.get('best_trial', '—')} | {fmt(r.get('f1'))} | {fmt(r.get('precision'))} | "
                         f"{fmt(r.get('recall'))} | {fmt(r.get('baseline_f1'))} | {r['class_weights']} | "
                         f"{r['seed']} | {r['epochs']} | `{params}` |")
        warnings = [f"- {r['dir']}: F1 выше baseline меньше чем на {BASELINE_MARGIN} — проверить, учится ли сеть."
                    for r in rows if isinstance(r.get("f1"), float) and
                    r["f1"] - (r.get("baseline_f1") or 0) < BASELINE_MARGIN]
        warnings += [f"- {r['dir']}: в Val-патчах нет точек целевого класса."
                     for r in rows if r.get("val_positive_points") == 0]
        lines += [""] + (["**Предупреждения:**", ""] + warnings + [""] if warnings else [])
        # Сравниваются только подборы (не final_*) с одной функцией потерь.
        for loss in sorted({r.get("loss", "ce") for r in rows}):
            by_feat = {r["features"]: r for r in rows if isinstance(r.get("f1"), float)
                       and r.get("loss", "ce") == loss and not r["dir"].startswith("final_")}
            if not {"xyz", "xyz_hag"} <= by_feat.keys():
                continue
            a, b = by_feat["xyz"], by_feat["xyz_hag"]
            same = all(a[k] == b[k] for k in ("seed", "epochs"))
            lines += [f"**Сравнение признаков `xyz` → `xyz_hag` ({loss}):** " +
                          ", ".join(f"{name} {b[k] - a[k]:+.4f}" for name, k in
                                    (("F1", "f1"), ("Precision", "precision"), ("Recall", "recall"))) +
                          ("." if same else " (внимание: разные seed или число эпох — сравнение некорректно)."),
                          ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(build(args.runs_dir), encoding="utf-8")
    print(f"Отчёт: {args.out}")


if __name__ == "__main__":
    main()
