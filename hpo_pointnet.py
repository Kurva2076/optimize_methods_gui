#!/usr/bin/env python3
"""Подбор PointNet++ для бинарной сегментации сцен LiDAR на локальных патчах.

Сцена — .npy [N, >=4]: колонка 0 = класс, колонки 1–3 = x, y, z; остальные колонки не используются
(они могут быть выведены из метки класса и дать утечку). Целевой класс = 1, всё остальное = 0.
Проверено по API torch-pointcloud v0.0.16. Нужны Python >=3.10,
PyTorch >=2.8, torch-geometric >=2.8, torch-pointcloud==0.0.16, Optuna, NumPy.
"""
import argparse
import csv
import json
import logging
import math
import platform
import shutil
import time
from datetime import timedelta
from pathlib import Path

# Реэкспорт: GUI, predict_scene.py, summary_report.py и тесты импортируют имена отсюда.
from hpo_core.data import (ABOVE_GROUND_THRESHOLD, AUG_SCALE, CLASS_HEADERS, CLASS_NAMES,  # noqa: F401
                           FEATURE_CHANNELS, GROUND_CELL, LABEL_COL, TARGET_TITLES, TARGETS, XY, XYZ,
                           Keyed, PatchDataset, class_counts, class_weights, collate_flat, compute_hag,
                           generate_data_summary, ground_grid, grow_circle, hag_at, load_scenes,
                           make_loader, patch_radius, split_scenes)
from hpo_core.metrics import (BASELINE_MARGIN, THRESHOLD_BINS, binary_metrics,  # noqa: F401
                              confusion_counts, counts_metrics, evaluate, threshold_metrics)
from hpo_core.model import (MODEL_NAME, N_POINTS_L0_MAX, FpsCache, build_model,  # noqa: F401
                            forward, make_criterion, seed_everything)
from hpo_core.study import (COMPATIBILITY_FIELDS, best_trial, compat_value_matches,  # noqa: F401
                            compatibility_info, generate_report, study_compatibility_error)

LOG = logging.getLogger("hpo_pointnet")
# Повышать при любом изменении диапазонов в run_trial или обучения, делающем trials несравнимыми
# (5: кеш FPS со своим случайным стартом — другой поток RNG, чем у прежних trials;
#  6: аугментации Train и F1 при пороге, подобранном по Val;
#  7: AdamW вместо Adam, weight_decay 1e-5..1e-1).
SEARCH_SPACE_VERSION = 7
FIELDS = ["number", "state", "f1", "precision", "recall", "baseline_f1", "seed", "target",
          "features", "checkpoint", "error", "params", "seconds"]
EPOCH_FIELDS = ["epoch", "f1", "precision", "recall", "iou", "train_f1", "train_precision",
                "train_recall", "train_iou", "lr", "seconds"]


def write_best(study, output_dir, run_info):
    """best.json — лучшие параметры и метрики прогона; его читает summary_report.py."""
    complete = best_trial(study)
    best = complete[0] if complete else None
    out = Path(output_dir)
    model_path = out / "best_model.pt"
    try:
        prev = json.loads((out / "best.json").read_text(encoding="utf-8")).get("best_trial")
    except (OSError, ValueError):
        prev = None
    result = dict(run_info, search_space=SEARCH_SPACE_VERSION, trials=len(study.trials),
                  complete=len(complete))
    if best:
        result.update(best_trial=best.number, f1=best.value,
                      **{k: best.user_attrs.get(k) for k in
                         ("precision", "recall", "iou", "baseline_f1", "val_positive_points",
                          "epoch", "train", "val", "threshold", "f1_default")},
                      params=best.params)
        # Итоговая модель прогона — копия best.pt лучшего trial (веса, params, run_info).
        ckpt = out / f"trial_{best.number:04d}" / "best.pt"
        if ckpt.exists():
            if prev != best.number or not model_path.exists():  # копируем только при смене лучшего
                shutil.copy2(ckpt, model_path)
            result["model"] = model_path.name
    if "model" not in result:
        model_path.unlink(missing_ok=True)  # не оставлять веса другого trial рядом с best.json
    (out / "best.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Отдельный каталог на каждый прогон, например runs/ground_xyz")
    parser.add_argument("--target", choices=list(TARGETS), required=True)
    parser.add_argument("--features", choices=list(FEATURE_CHANNELS), default="xyz",
                        help="xyz или xyz_hag (+ HAG и is_above_ground, считаются по геометрии)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=8, help="Эпохи на trial")
    parser.add_argument("--patch-points", type=int, default=16384, help="Точек в патче")
    parser.add_argument("--patches-per-scene", type=int, default=2,
                        help="Патчей на сцену за эпоху (train и val)")
    parser.add_argument("--workers", type=int, default=0, help="Процессов DataLoader")
    parser.add_argument("--fixed-params", type=Path, default=None,
                        help="JSON с гиперпараметрами (например params из best.json): первым "
                             "trial обучается ровно эта конфигурация — для финальных прогонов")
    parser.add_argument("--loss", choices=["ce", "focal", "focal_lovasz"], default="ce",
                        help="ce — взвешенная CrossEntropyLoss, focal — Focal Loss с теми же весами, "
                             "focal_lovasz — (1-w)*Focal + w*Lovász-Softmax")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="gamma для focal и focal_lovasz")
    parser.add_argument("--lovasz-weight", type=float, default=0.5,
                        help="Доля Lovász в focal_lovasz, от 0 до 1")
    parser.add_argument("--class-weight-power", type=float, default=1.0,
                        help="Степень обратных частот классов: 1 — как раньше, 0.47 — мягче (Авин), 0 — без весов")
    parser.add_argument("--equibatch", action="store_true",
                        help="Train: часть патчей с центром на точке цели, остальные — на фоне")
    parser.add_argument("--equibatch-positive-fraction", type=float, default=0.5,
                        help="Доля патчей с центром на цели среди не граничных, 0 < доля < 1")
    parser.add_argument("--boundary-patch-fraction", type=float, default=0.0,
                        help="Train: доля патчей с центром на границе классов, [0, 1); нужна --equibatch")
    parser.add_argument("--boundary-weight", type=float, default=1.0,
                        help="Множитель веса пограничных точек в CE/Focal, >= 1; 1 — выключено")
    parser.add_argument("--boundary-radius", type=float, default=0.5,
                        help="Радиус поиска точки другого класса, в исходных единицах координат")
    parser.add_argument("--boundary-space", choices=["xyz", "xy"], default="xyz",
                        help="Расстояние до другого класса: 3D или по XY")
    parser.add_argument("--amp", action="store_true",
                        help="Смешанная точность fp16 на CUDA (быстрее, меньше памяти); на CPU не действует")
    parser.add_argument("--wandb-api-key", default=None, help="Не записывается в артефакты")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if min(args.n_trials, args.epochs, args.patch_points, args.patches_per_scene) < 1:
        parser.error("n-trials, epochs, patch-points и patches-per-scene должны быть положительными")
    if not 0 <= args.lovasz_weight <= 1:
        parser.error("lovasz-weight должна быть от 0 до 1")
    if not 0 <= args.class_weight_power <= 1:
        parser.error("class-weight-power должна быть от 0 до 1")
    if not 0 < args.equibatch_positive_fraction < 1:
        parser.error("equibatch-positive-fraction должна быть строго между 0 и 1")
    if not 0 <= args.boundary_patch_fraction < 1:
        parser.error("boundary-patch-fraction должна быть в [0, 1)")
    if args.boundary_patch_fraction and not args.equibatch:
        parser.error("boundary-patch-fraction требует --equibatch")
    if not math.isfinite(args.boundary_weight) or args.boundary_weight < 1:
        parser.error("boundary-weight должна быть >= 1")
    if not math.isfinite(args.boundary_radius) or args.boundary_radius <= 0:
        parser.error("boundary-radius должна быть > 0")
    if args.boundary_weight > 1 and args.loss == "focal_lovasz" and args.lovasz_weight == 1:
        parser.error("boundary-weight действует только на долю CE/Focal; уменьшите lovasz-weight")
    loss = "ce" if args.loss == "ce" else f"focal(gamma={args.focal_gamma:g})"
    if args.loss == "focal_lovasz":
        loss = f"(1-{args.lovasz_weight:g})*{loss}+{args.lovasz_weight:g}*lovasz"
    boundary = args.boundary_patch_fraction > 0 or args.boundary_weight > 1
    # Параметры обучения, при которых trials сопоставимы (и попадают в run_info/best.json).
    training = dict(
        loss=loss, class_weight_power=args.class_weight_power,
        positive_fraction=args.equibatch_positive_fraction if args.equibatch else None,
        boundary_fraction=args.boundary_patch_fraction, boundary_weight=args.boundary_weight,
        boundary_radius=args.boundary_radius if boundary else None,
        boundary_space=args.boundary_space if boundary else None)
    # Зависимости импортируем после разбора аргументов для понятного --help.
    import optuna
    import torch
    try:
        import torch_pointcloud  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Установите torch-pointcloud==0.0.16 и его PyG-зависимости") from exc
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "study.db").exists():
        # Битый или занятый study.db — понятная ошибка до записи каких-либо артефактов.
        try:
            existing = optuna.load_study(study_name="pointnet_binary",
                                         storage=f"sqlite:///{(args.output_dir / 'study.db').as_posix()}")
        except SystemExit:
            raise
        except Exception as exc:
            raise SystemExit(f"{args.output_dir / 'study.db'} не удалось прочитать "
                             f"({type(exc).__name__}: {exc}); укажите новую --output-dir") from exc
        if existing.trials:
            requested = dict(data_dir=str(args.data_dir), seed=args.seed, epochs=args.epochs,
                             target=args.target, features=args.features, patch_points=args.patch_points,
                             patches_per_scene=args.patches_per_scene, **training)
            error = study_compatibility_error(existing.user_attrs, requested)
            if error:
                raise SystemExit(f"{args.output_dir}: {error}")
            if existing.user_attrs.get("search_space") != SEARCH_SPACE_VERSION:
                raise SystemExit(f"{args.output_dir}: несовместимая версия study; укажите новую --output-dir")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"
    if args.amp and not amp:
        LOG.warning("--amp: CUDA недоступна, обучение в fp32")
    target_ids = TARGETS[args.target]
    data_summary = generate_data_summary(args.data_dir)
    (args.output_dir / "data_summary.md").write_text(
        "# Сводка исходного датасета\n\n" + data_summary, encoding="utf-8")
    train_paths, val_paths = split_scenes(load_scenes(args.data_dir), args.seed)
    train = PatchDataset(train_paths, target_ids, args.patch_points,
                         args.patches_per_scene, args.seed, args.features, augment=True,
                         **{k: v for k, v in training.items()
                            if k not in ("loss", "class_weight_power") and v is not None})
    val = PatchDataset(val_paths, target_ids, args.patch_points,
                       args.patches_per_scene, args.seed, args.features)
    # Длина порядка = верх n_points_l0, чтобы префикс покрывал любой trial. Хранить стоит только
    # между trials; при одном trial (финальный прогон) — та же схема FPS без хранения.
    keep = args.n_trials > 1
    train_cache = FpsCache(N_POINTS_L0_MAX, True, args.seed, keep)
    val_cache = FpsCache(N_POINTS_L0_MAX, False, args.seed, keep)
    counts = class_counts(train_paths, target_ids)
    weights = class_weights(counts, args.class_weight_power)
    run_info = {
        "target": args.target, "target_ids": list(target_ids), "features": args.features,
        "seed": args.seed, "epochs": args.epochs, "n_trials": args.n_trials,
        "patch_points": args.patch_points, "patches_per_scene": args.patches_per_scene,
        "data_dir": str(args.data_dir), "fixed_params": str(args.fixed_params or ""),
        "train_points": {"background": int(counts[0]), "target": int(counts[1])},
        "class_weights": [round(float(w), 4) for w in weights],
        **training,
        "optimizer": "AdamW", "amp": amp,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "optuna": optuna.__version__},
    }
    (args.output_dir / "split.json").write_text(json.dumps(dict(
        run_info, train=[p.name for p in train_paths], val=[p.name for p in val_paths]),
        indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Патчей за эпоху: train=%s val=%s; устройство=%s; веса классов=%s",
             len(train), len(val), device, run_info["class_weights"])
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    wandb = None
    if args.wandb_api_key:
        import wandb as wandb_module
        wandb_module.login(key=args.wandb_api_key)
        wandb = wandb_module
    csv_path = args.output_dir / "trials.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def objective(trial):
        # Причина падения идёт в колонку error trials.csv.
        try:
            return run_trial(trial)
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            raise

    def run_trial(trial):
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            # AdamW: decoupled weight decay (шаг lr * wd), поэтому диапазон выше, чем у L2 в Adam.
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "batch_size": trial.suggest_categorical("batch_size", [2, 3, 4]),  # патчей в батче
            "n_sa_layers": trial.suggest_int("n_sa_layers", 2, 4),
            "n_points_l0": trial.suggest_int("n_points_l0", 1024, N_POINTS_L0_MAX),
            "radius_l0": trial.suggest_float("radius_l0", 0.05, 0.5),  # в долях радиуса патча
            "nsample": trial.suggest_int("nsample", 16, 64),
            "use_MSG": trial.suggest_categorical("use_MSG", [False, True]),
        }
        trial_seed = args.seed  # Один и тот же seed для сравнимости trials.
        seed_everything(trial_seed)
        run_dir = args.output_dir / f"trial_{trial.number:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(dict(
            run_info, model=MODEL_NAME, params=params,
            train=[p.name for p in train_paths], val=[p.name for p in val_paths]),
            indent=2, ensure_ascii=False), encoding="utf-8")
        model = build_model(params, FEATURE_CHANNELS[args.features]).to(device)
        criterion = make_criterion(args.loss, weights, args.focal_gamma, args.lovasz_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                                      weight_decay=params["weight_decay"])
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
        # ТЗ: переобучение замечено после 4-й эпохи — снижаем lr, когда F1 на Val не растёт.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                               factor=0.5, patience=1)
        val_loader = make_loader(Keyed(val), params["batch_size"], False, trial_seed, args.workers)
        best, best_m, history = -1.0, None, []
        ckpt_path = run_dir / "best.pt"
        wb_run = wandb.init(project="pointnet-hpo", config=dict(params, target=args.target, loss=run_info["loss"],
                                                                features=args.features),
                            name=f"{args.target}_{args.features}_trial_{trial.number:04d}",
                            reinit=True) if wandb else None
        try:
            for epoch in range(args.epochs):
                train.set_epoch(epoch)
                train_loader = make_loader(Keyed(train), params["batch_size"], True,
                                           trial_seed + epoch, args.workers)
                model.train()
                # Train-метрики — по предсказаниям самих шагов обучения (режим train, веса
                # меняются по ходу эпохи): без второго прохода по Train.
                train_counts = torch.zeros(4, dtype=torch.long, device=device)
                start, n_batches = time.time(), len(train_loader)
                for step, (feat, batch, y, keys, *point_weight) in enumerate(train_loader, 1):
                    feat, batch, y = feat.to(device), batch.to(device), y.to(device)
                    point_weight = point_weight[0].to(device) if point_weight else None
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                        out = forward(model, feat, batch, keys, train_cache)
                    out = out.float()  # loss в fp32: exp в focal и веса классов без потери точности
                    loss = criterion(out, y, point_weight)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_counts += confusion_counts(out.detach().argmax(dim=1), y)
                    if step % 20 == 0 or step == n_batches:  # прогресс внутри долгой эпохи
                        LOG.info("trial=%s epoch=%s batch %s/%s loss=%.4f %.0fs", trial.number,
                                 epoch + 1, step, n_batches, loss.item(), time.time() - start)
                train_m = counts_metrics(train_counts)
                m = evaluate(model, val_loader, device, val_cache, amp)
                seconds = round(time.time() - start, 1)  # обучение + валидация
                f1 = m["f1"]
                lr = optimizer.param_groups[0]["lr"]
                history.append({"epoch": epoch + 1, "f1": f1, "precision": m["precision"],
                                "recall": m["recall"], "iou": m["iou"],
                                **{f"train_{k}": train_m[k] for k in ("f1", "precision", "recall", "iou")},
                                "lr": lr, "seconds": seconds})
                scheduler.step(f1)
                if wb_run:
                    wb_run.log({f"{name}/{k}": v for name, d in (("val", m), ("train", train_m))
                                for k, v in d.items() if k != "confusion"} |
                               {"epoch": epoch + 1, "lr": lr})
                if f1 > best:
                    best, best_m = f1, dict(m, epoch=epoch + 1, train=train_m, val=m)
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "params": params, "run_info": run_info, "epoch": epoch + 1,
                                "f1": best, "threshold": m["threshold"]}, ckpt_path)
                trial.report(f1, epoch)
                LOG.info("trial=%s epoch=%s F1=%.4f best=%.4f baseline=%.4f lr=%.2g время=%s",
                         trial.number, epoch + 1, f1, best, m["baseline_f1"], lr,
                         timedelta(seconds=round(seconds)))
                # Без «trial=»: mlstudio разбирает строки с «trial=… epoch=… F1=» как точки графика.
                LOG.info("trial %s epoch %s | Train: IoU=%.4f F1=%.4f P=%.4f R=%.4f "
                         "| Val: IoU=%.4f F1=%.4f P=%.4f R=%.4f порог=%.2f (F1 при 0.5: %.4f)",
                         trial.number, epoch + 1,
                         *(d[k] for d in (train_m, m) for k in ("iou", "f1", "precision", "recall")),
                         m["threshold"], m["f1_default"])
                LOG.info("trial %s epoch %s | матрица ошибок [[TN, FP], [FN, TP]]: Train=%s Val=%s",
                         trial.number, epoch + 1, train_m["confusion"], m["confusion"])
                if trial.should_prune():
                    raise optuna.TrialPruned()
            trial.set_user_attr("checkpoint", str(ckpt_path))
            return best
        finally:
            if best_m:
                for key in ("precision", "recall", "iou", "baseline_f1", "val_positive_points",
                            "epoch", "train", "val", "threshold", "f1_default"):
                    trial.set_user_attr(key, best_m[key])
                with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=EPOCH_FIELDS)
                    writer.writeheader()
                    writer.writerows(history)
                (run_dir / "metrics.json").write_text(json.dumps({
                    "seed": trial_seed, "target": args.target, "features": args.features,
                    "class_weights": run_info["class_weights"], "best": best_m,
                }, indent=2), encoding="utf-8")
            if wb_run:
                wb_run.finish()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def log_trial(study, trial):
        seconds = round(trial.duration.total_seconds(), 1) if trial.duration else ""
        if seconds != "":
            LOG.info("trial=%s %s за %s", trial.number, trial.state.name,
                     timedelta(seconds=round(seconds)))
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow({
                "number": trial.number, "state": trial.state.name, "f1": trial.value,
                "precision": trial.user_attrs.get("precision", ""),
                "recall": trial.user_attrs.get("recall", ""),
                "baseline_f1": trial.user_attrs.get("baseline_f1", ""), "seed": args.seed,
                "target": args.target, "features": args.features,
                "checkpoint": trial.user_attrs.get("checkpoint", ""),  # только у COMPLETE
                "error": trial.user_attrs.get("error", ""),
                "params": json.dumps(trial.params, ensure_ascii=False), "seconds": seconds})
        generate_report(study, args.output_dir / "report.md", run_info, data_summary,
                        importances=False)
        write_best(study, args.output_dir, run_info)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed),
        # Медиана по 5 полным trials; обрезаем не раньше 4-й эпохи: к ней ReduceLROnPlateau
        # (patience 1) успевает снизить lr, и медленные конфиги не режутся зря.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                           n_warmup_steps=max(3, args.epochs // 3)),
        study_name="pointnet_binary", storage=f"sqlite:///{(args.output_dir / 'study.db').as_posix()}",
        load_if_exists=True)
    # Trials старого пространства поиска несовместимы с новым: Optuna упадёт на первом же trial.
    if study.trials and study.user_attrs.get("search_space") != SEARCH_SPACE_VERSION:
        raise SystemExit(f"{args.output_dir}: study версии {study.user_attrs.get('search_space')}, "
                         f"код — версии {SEARCH_SPACE_VERSION} (изменились диапазоны поиска или "
                         "обучение, trials несравнимы); укажите новую --output-dir "
                         "(для run_all.py — новую --runs) или перенесите старый каталог")
    # Trials с разными функциями потерь несравнимы; studies без атрибута обучались на CE.
    if study.trials and study.user_attrs.get("loss", "ce") != run_info["loss"]:
        raise SystemExit(f"{args.output_dir}: study обучалась с loss={study.user_attrs.get('loss', 'ce')}, "
                         f"запрошено {run_info['loss']}; укажите другую --output-dir")
    # fp16 и fp32 дают несравнимые trials; у study без атрибута amp был выключен.
    if study.trials and study.user_attrs.get("amp", False) != run_info["amp"]:
        raise SystemExit(f"{args.output_dir}: study обучалась с amp={study.user_attrs.get('amp', False)}, "
                         f"запрошено {run_info['amp']}; укажите другую --output-dir")
    study.set_user_attr("search_space", SEARCH_SPACE_VERSION)
    study.set_user_attr("loss", run_info["loss"])
    study.set_user_attr("amp", run_info["amp"])
    study.set_user_attr("compatibility", compatibility_info(run_info))
    if args.fixed_params and not study.trials:
        study.enqueue_trial(json.loads(args.fixed_params.read_text(encoding="utf-8")))
    study.optimize(objective, n_trials=args.n_trials, callbacks=[log_trial],
                   catch=(Exception,))  # падение одного trial не обрывает поиск
    generate_report(study, args.output_dir / "report.md", run_info, data_summary)
    write_best(study, args.output_dir, run_info)
    LOG.info("Отчёт: %s", args.output_dir / "report.md")


if __name__ == "__main__":
    main()
