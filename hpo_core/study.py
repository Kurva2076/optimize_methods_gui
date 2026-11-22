"""Отчёт по study Optuna и совместимость продолжаемых прогонов."""
import json
import math
from pathlib import Path

from .data import TARGET_TITLES
from .metrics import BASELINE_MARGIN


def best_trial(study):
    import optuna
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and
                t.value is not None and math.isfinite(t.value)]
    return sorted(complete, key=lambda t: t.value, reverse=True)


def generate_report(study, output_path, run_info, data_summary=None, importances=True):
    import optuna
    complete = best_trial(study)
    title = TARGET_TITLES[run_info["target"]]
    lines = [f"# Подбор PointNet++: {title} ({run_info['features']})", "", "## Сводка", "",
             f"- Целевые классы: {run_info['target_ids']} → 1, остальное → 0.",
             f"- Признаки: `{run_info['features']}`; seed {run_info['seed']}; эпох на trial {run_info['epochs']}.",
             f"- Функция потерь: {run_info['loss']}; веса классов (фон, цель): {run_info['class_weights']}.",
             f"- Испытаний: {len(study.trials)} (успешных: {len(complete)})."]
    if not complete:
        lines += ["- Успешных испытаний нет; рекомендации невозможны.", ""]
    else:
        best = complete[0]
        baseline = best.user_attrs.get("baseline_f1", float("nan"))
        lines += [f"- Лучший F1 целевого класса: {best.value:.4f} (trial {best.number}; порог подобран "
                  f"по Val, поэтому оценка оптимистична); baseline «всё = 1»: {baseline:.4f}.",
                  f"- Порог вероятности цели (подобран по Val): {best.user_attrs.get('threshold', 0.5):.2f}; "
                  f"F1 при пороге 0.5 без подбора: {best.user_attrs.get('f1_default', float('nan')):.4f}.",
                  f"- Лучшие гиперпараметры: `{json.dumps(best.params, ensure_ascii=False, sort_keys=True)}`.",
                  "", "## Топ-5 конфигураций", "",
                  "| Ранг | Trial | F1 | Precision | Recall | lr | weight_decay | dropout | batch_size (патчи) | n_sa_layers | n_points_l0 | radius_l0 | nsample | use_MSG |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"]
        keys = ["lr", "weight_decay", "dropout", "batch_size", "n_sa_layers",
                "n_points_l0", "radius_l0", "nsample", "use_MSG"]
        for rank, t in enumerate(complete[:5], 1):
            vals = [f"{t.params.get(k, '—'):.3g}" if isinstance(t.params.get(k), float)
                    else str(t.params.get(k, "—")) for k in keys]
            lines.append(f"| {rank} | {t.number} | {t.value:.4f} | "
                         f"{t.user_attrs.get('precision', float('nan')):.4f} | "
                         f"{t.user_attrs.get('recall', float('nan')):.4f} | " + " | ".join(vals) + " |")
        train_m, val_m = best.user_attrs.get("train"), best.user_attrs.get("val")
        if train_m and val_m:
            lines += ["", f"## Метрики лучшего trial (эпоха {best.user_attrs.get('epoch')})", "",
                      "| Выборка | IoU | F1 | Precision | Recall |", "|:---|---:|---:|---:|---:|"]
            lines += [f"| {name} | {m['iou']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | "
                      f"{m['recall']:.4f} |" for name, m in (("Train", train_m), ("Val", val_m))]
            for name, m in (("Train", train_m), ("Val", val_m)):
                (tn, fp), (fn, tp) = m["confusion"]
                lines += ["", f"Матрица ошибок {name} (точки; строки — истина, столбцы — предсказание):", "",
                          "| | Пред. фон | Пред. цель |", "|:---|---:|---:|",
                          f"| Фон | {tn} | {fp} |", f"| Цель | {fn} | {tp} |"]
        lines += ["", "## Важность параметров", ""]
        try:
            if not importances:  # fANOVA дорогая: считаем только в итоговом отчёте
                raise ValueError("считается в конце прогона")
            if len(complete) < 2:
                raise ValueError("Нужно не менее двух успешных испытаний")
            values = optuna.importance.get_param_importances(study)
            lines.extend(f"- `{k}`: {v:.3f}" for k, v in values.items())
        except (ValueError, RuntimeError) as exc:
            lines.append(f"Недоступна: {exc}.")
        lines += ["", "## Предупреждения", ""]
        # Сравнение с baseline — по F1 без подбора порога (подбор по тому же Val завышает F1).
        if best.user_attrs.get("f1_default", best.value) - baseline < BASELINE_MARGIN:
            lines.append(f"- F1 при пороге 0.5 выше baseline «всё = 1» меньше чем на {BASELINE_MARGIN}: "
                         "проверьте, учится ли сеть.")
        if best.user_attrs.get("val_positive_points", 1) == 0:
            lines.append("- В Val-патчах нет точек целевого класса: метрики не информативны.")
        if best.value < 0.6:
            lines.append("- Лучший F1 < 0.6: модель не достигла приемлемого качества.")
        if best.params["lr"] <= 1.1e-4 or best.params["lr"] >= 9e-3:
            lines.append("- lr близко к границе диапазона; расширьте поиск.")
        if lines[-1] == "":
            lines.append("- Пороговых предупреждений нет.")
        lines += ["", "## Кандидаты для финального прогона", ""]
        chosen = []
        for t in complete:
            signature = (t.params["n_sa_layers"], t.params["n_points_l0"],
                         t.params["radius_l0"], t.params["use_MSG"])
            if not any(s == signature for s in chosen):
                chosen.append(signature)
                lines.append(f"- Trial {t.number} (F1={t.value:.4f}): " +
                             ("лучший F1; проверить на длинном обучении." if len(chosen) == 1 else
                              "другая геометрия SA/MSG; сравнить устойчивость при длинном обучении."))
            if len(chosen) == 3:
                break
        if len(chosen) < 2:
            lines.append("- Различных конфигураций меньше двух; продолжите поиск перед выбором.")
    if data_summary:
        lines += ["", "# Сводка исходного датасета", "", data_summary.rstrip()]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


COMPATIBILITY_FIELDS = ("data_dir", "seed", "epochs", "target", "features",
                       "patch_points", "patches_per_scene", "loss", "class_weight_power",
                       "positive_fraction", "boundary_fraction", "boundary_weight",
                       "boundary_radius", "boundary_space")
# Значения полей, которых нет у studies, записанных до их появления (= прежнее обучение).
COMPATIBILITY_DEFAULTS = {"class_weight_power": 1.0, "positive_fraction": None,
                          "boundary_fraction": 0.0, "boundary_weight": 1.0,
                          "boundary_radius": None, "boundary_space": None}


def compatibility_info(run_info):
    """Параметры, при которых trials остаются сопоставимыми."""
    return {key: run_info.get(key) for key in COMPATIBILITY_FIELDS}


def study_compatibility_error(study_attrs, run_info):
    """Возвращает текст ошибки для несовместимого существующего study."""
    expected = compatibility_info(run_info)
    actual = study_attrs.get("compatibility")
    if isinstance(actual, str):
        try:
            actual = json.loads(actual)
        except json.JSONDecodeError:
            actual = None
    if not isinstance(actual, dict):
        return "у study отсутствуют параметры совместимости; укажите новую --output-dir"
    differences = [key for key in COMPATIBILITY_FIELDS
                   if not compat_value_matches(key, actual.get(key, COMPATIBILITY_DEFAULTS.get(key)),
                                                expected.get(key))]
    if differences:
        changed = ", ".join(differences)
        return f"у study отличаются параметры совместимости: {changed}; укажите новую --output-dir"
    return None


def compat_value_matches(key, stored, requested):
    """Совпадение значения параметра совместимости.

    data_dir допускает разные написания одного каталога (относительное, с «./», абсолютное):
    сравниваем разрешённые абсолютные пути, чтобы продолжение не блокировалось ложно.
    """
    if key == "data_dir" and isinstance(stored, str) and isinstance(requested, str):
        return Path(stored).resolve() == Path(requested).resolve()
    return stored == requested
