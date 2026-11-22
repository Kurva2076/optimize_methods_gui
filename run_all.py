"""Задачи из tasks.md: №2–6 подбор, №7 сводный отчёт, final — финальные прогоны.

Запуск: python run_all.py [--tasks 2 3 7 final] [--n-trials 1 --epochs 1]
(Windows: run_all.cmd сам готовит .venv311). Без --tasks выполняются все задачи по порядку.
Вывод идёт на экран сразу и дублируется в <runs>/run_all.log. Прерванный запуск продолжается
повторным запуском: каждый прогон подхватывает свой study.db.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS_PLAN = [  # (задача, имя, target, features)
    ("2", "ground_xyz", "ground", "xyz"),
    ("2", "ground_xyz_hag", "ground", "xyz_hag"),
    ("3", "vegetation_xyz", "vegetation", "xyz"),
    ("3", "vegetation_xyz_hag", "vegetation", "xyz_hag"),
    ("4", "noise_xyz", "noise", "xyz"),
    ("5", "building_xyz", "building", "xyz"),
    ("6", "water_xyz", "water", "xyz"),
]
TASKS = ["2", "3", "4", "5", "6", "7", "final"]


def run_all_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS,
                    help="какие задачи выполнить: 2–6 подбор по классам, 7 отчёт, final финальные "
                         "прогоны (для выбранных 2–6, а если их нет — для всех, где есть best.json)")
    ap.add_argument("--data", type=Path, default=ROOT / "exports")
    ap.add_argument("--final-data", type=Path, default=ROOT / "exports_final",
                    help="каталог данных для финальных прогонов (задача final); должен существовать — "
                         "иначе сценарий останавливается с ошибкой")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=25)   # ТЗ: 20–30
    ap.add_argument("--epochs", type=int, default=10)     # ТЗ: 8–10
    ap.add_argument("--final-epochs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patch-points", type=int, default=16384)
    ap.add_argument("--patches-per-scene", type=int, default=2)
    ap.add_argument("--loss", choices=["ce", "focal", "focal_lovasz"], default="ce")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    # Те же настройки обучения, что у hpo_pointnet.py; по умолчанию — прежнее обучение.
    ap.add_argument("--lovasz-weight", type=float, default=0.5)
    ap.add_argument("--class-weight-power", type=float, default=1.0)
    ap.add_argument("--equibatch", action="store_true")
    ap.add_argument("--equibatch-positive-fraction", type=float, default=0.5)
    ap.add_argument("--boundary-patch-fraction", type=float, default=0.0)
    ap.add_argument("--boundary-weight", type=float, default=1.0)
    ap.add_argument("--boundary-radius", type=float, default=0.5)
    ap.add_argument("--amp", action="store_true", help="fp16 на CUDA для всех задач")
    return ap


def child_hpo_args(args, output_dir, target, features, data, n_trials, epochs, *extra):
    return [ROOT / "hpo_pointnet.py", "--data-dir", data, "--output-dir", output_dir,
            "--target", target, "--features", features, "--seed", args.seed,
            "--n-trials", n_trials, "--epochs", epochs, "--workers", args.workers,
            "--patch-points", args.patch_points, "--patches-per-scene", args.patches_per_scene,
            "--loss", args.loss, "--focal-gamma", args.focal_gamma,
            "--lovasz-weight", args.lovasz_weight, "--class-weight-power", args.class_weight_power,
            "--equibatch-positive-fraction", args.equibatch_positive_fraction,
            "--boundary-patch-fraction", args.boundary_patch_fraction,
            "--boundary-weight", args.boundary_weight, "--boundary-radius", args.boundary_radius,
            *(["--equibatch"] if args.equibatch else []),
            *(["--amp"] if args.amp else []), *extra]


def train_tag(args):
    """Суффикс каталогов: study с другими настройками обучения не продолжится (как TrainOpts::suffix в mlstudio)."""
    tag = {"ce": "", "focal": "_focal", "focal_lovasz": "_focal_lovasz"}[args.loss]
    if args.loss != "ce" and args.focal_gamma != 2:
        tag += f"_g{args.focal_gamma:g}"
    if args.loss == "focal_lovasz" and args.lovasz_weight != 0.5:
        tag += f"_lw{args.lovasz_weight:g}"
    if args.class_weight_power != 1:
        tag += f"_cw{args.class_weight_power:g}"
    if args.equibatch:
        tag += "_eq" + ("" if args.equibatch_positive_fraction == 0.5 else f"{args.equibatch_positive_fraction:g}")
    if args.boundary_patch_fraction:
        tag += f"_bf{args.boundary_patch_fraction:g}"
    if args.boundary_weight != 1:
        tag += f"_bw{args.boundary_weight:g}"
    if (args.boundary_patch_fraction or args.boundary_weight != 1) and args.boundary_radius != 0.5:
        tag += f"_br{args.boundary_radius:g}"
    return tag


def write_final_params(params_path, params):
    if params_path.exists():
        raise SystemExit(f"Файл параметров финала уже существует: {params_path}; перезапись запрещена")
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps(params, ensure_ascii=False), encoding="utf-8")


def main():
    args = run_all_parser().parse_args()
    if min(args.n_trials, args.epochs, args.final_epochs, args.patch_points, args.patches_per_scene) < 1:
        raise SystemExit("n-trials, epochs, final-epochs, patch-points и patches-per-scene должны быть положительными")
    args.runs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    # Study с другим loss hpo_pointnet.py не продолжит: focal-прогоны идут в свои каталоги.
    # Gamma в имени: study с другой gamma тоже не продолжится (2 — по умолчанию, без суффикса).
    tag = train_tag(args)
    selected = [(t, name + tag, *rest) for t, name, *rest in RUNS_PLAN if t in args.tasks]

    with (args.runs / "run_all.log").open("a", encoding="utf-8") as log:
        def say(text):
            print(text, flush=True)
            log.write(text + "\n")
            log.flush()

        # До запуска детей: битый --final-data не должен портить прогоны; ошибка — и в лог, и на экран.
        if "final" in args.tasks and not args.final_data.is_dir():
            say(f"=== ОШИБКА: выберите существующий каталог данных для финального прогона: {args.final_data}")
            sys.exit(1)

        def sh(*cmd):
            # Строки процесса сразу на экран и в лог; падение останавливает весь сценарий.
            proc = subprocess.Popen([sys.executable, "-u", *map(str, cmd)], cwd=ROOT, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                say(line.rstrip("\n"))
            if proc.wait():
                say(f"=== ОШИБКА: код {proc.returncode}")
                sys.exit(proc.returncode)

        def since(t0):
            return timedelta(seconds=round(time.time() - t0))

        def run(name, target, features, data, n_trials, epochs, *extra):
            started = time.time()
            say(f"\n=== {time.strftime('%T')} {name} ({target}, {features})")
            sh(*child_hpo_args(args, args.runs / name, target, features, data, n_trials, epochs, *extra))
            say(f"=== {time.strftime('%T')} {name} за {since(started)}")

        total = time.time()
        say(f"=== {time.strftime('%F %T')} старт: задачи {' '.join(args.tasks)}; data={args.data} "
            f"seed={args.seed} trials={args.n_trials} epochs={args.epochs} python={sys.executable}")
        sh("-c", "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())")
        for _, name, target, features in selected:
            run(name, target, features, args.data, args.n_trials, args.epochs)

        if "final" in args.tasks:
            for _, name, _, _ in selected or [(t, n + tag, *r) for t, n, *r in RUNS_PLAN]:
                best_path = args.runs / name / "best.json"
                if not best_path.exists():
                    say(f"=== {name}: нет {best_path} — сначала подбор, финальный прогон пропущен")
                    continue
                best = json.loads(best_path.read_text(encoding="utf-8"))
                params = args.runs / f"{name}.params.json"
                output = args.runs / f"final_{name}"
                # Проверки — до записи params-файла: неудачная попытка не оставляет артефакт,
                # который заблокирует повтор (файл намеренно не перезаписывается).
                if output.exists():
                    say(f"=== ОШИБКА: каталог финального прогона уже существует: {output}; перезапись запрещена")
                    sys.exit(1)
                if params.exists():
                    say(f"=== ОШИБКА: файл параметров финала уже существует: {params}; перезапись запрещена")
                    sys.exit(1)
                write_final_params(params, best["params"])
                run(f"final_{name}", best["target"], best["features"], args.final_data,
                    1, args.final_epochs, "--fixed-params", params)

        if "7" in args.tasks:
            say(f"\n=== {time.strftime('%T')} сводный отчёт (№7)")
            sh(ROOT / "summary_report.py", args.runs, "--out", args.runs / "report.md")
        say(f"=== {time.strftime('%F %T')} готово за {since(total)}: {args.runs}")


if __name__ == "__main__":
    main()
