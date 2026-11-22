#!/usr/bin/env python3
"""Предсказание модели из hpo_pointnet.py на целой сцене (для 3D-превью в ML Studio).

Сцена покрывается сеткой кругов того же радиуса, что патчи обучения (≈patch_points точек при средней
плотности); круг, где точек больше, делится на случайные группы по patch_points. Признаки — как в
PatchDataset. Вероятности перекрывающихся патчей усредняются.

Выход --out: .npy uint8 [N] — 1 целевой класс, 0 нет, 255 точка не покрыта.

python predict_scene.py --checkpoint runs/ground_xyz/trial_0003/best.pt --scene exports/scene_001_cleaned.npy --out pred.npy
python predict_scene.py --checkpoint … --scene exports/scene_00*.npy --out-dir runs/ground_xyz/pred --tag trial3  (метрики по сценам)
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from hpo_pointnet import (ABOVE_GROUND_THRESHOLD, FEATURE_CHANNELS, LABEL_COL, TARGETS, XY, XYZ,
                          binary_metrics, build_model, ground_grid, grow_circle, hag_at,
                          patch_radius)


def patches(xy, n_points, rng):
    """Индексы точек каждого патча: круги радиуса r по сетке с шагом r√2 (покрывают плоскость).

    Как в обучении (PatchDataset), круг с малым числом точек (край, редкие места) расширяется,
    пока в нём не наберётся n_points; координаты патча делятся на итоговый радиус.
    """
    n = len(xy)
    lo, hi = xy.min(0), xy.max(0)
    r = patch_radius(xy, n_points)
    step = r * math.sqrt(2)
    for cx in np.arange(lo[0], hi[0] + step, step):
        for cy in np.arange(lo[1], hi[1] + step, step):
            d2 = ((xy - (cx, cy)) ** 2).sum(1)
            if not (d2 <= r * r).any():
                continue
            idx, rr = grow_circle(d2, r, min(n_points, n))
            groups = max(1, math.ceil(len(idx) / n_points))
            for g in np.array_split(rng.permutation(idx), groups):
                yield np.sort(g), np.array([cx, cy], dtype=np.float32), rr


def predict(model, arr, features, n_points, seed, device, torch, threshold=0.5):
    """Предсказание по всей сцене: uint8 [N] — 1 целевой класс, 0 нет, 255 точка не покрыта.

    Цель — средняя p(цель) >= threshold (порог из чекпоинта, подобран по Val при обучении).
    """
    xyz = np.asarray(arr[:, XYZ], dtype=np.float32)
    n = len(xyz)
    ground = ground_grid(xyz) if features == "xyz_hag" else None
    prob = np.zeros(n, dtype=np.float32)
    seen = np.zeros(n, dtype=np.float32)
    todo = list(patches(xyz[:, :2], n_points, np.random.default_rng(seed)))
    print(f"=== {n} точек, {len(todo)} патчей, {features}, {device}", flush=True)
    start = time.time()
    with torch.no_grad():
        for i, (idx, center, r) in enumerate(todo, 1):
            p = xyz[idx]
            feat = (p - np.array([center[0], center[1], p[:, 2].min()], dtype=np.float32)) / r
            if ground is not None:
                hag = hag_at(p, *ground)
                feat = np.column_stack((feat, hag / r, hag > ABOVE_GROUND_THRESHOLD))
            f = torch.from_numpy(feat.astype(np.float32)).to(device)
            logits = model(f, f[:, :3].contiguous(), torch.zeros(len(idx), dtype=torch.long, device=device))
            prob[idx] += torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            seen[idx] += 1
            if i % 10 == 0 or i == len(todo):
                print(f"патч {i}/{len(todo)} {time.time() - start:.0f}s", flush=True)
    return np.where(seen > 0, (prob / np.maximum(seen, 1)) >= threshold, 255).astype(np.uint8)


def scene_metrics(pred, arr, target):
    y = np.isin(np.asarray(arr[:, LABEL_COL]), TARGETS[target])
    tp = int(((pred == 1) & y).sum())
    fp = int(((pred == 1) & ~y).sum())
    fn = int(((pred == 0) & y).sum())
    m = binary_metrics(tp, fp, fn, int((~y).sum()))
    m.update(points=len(pred), not_covered=int((pred == 255).sum()))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True, help="best.pt trial-а")
    ap.add_argument("--scene", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, help="одна сцена: файл предсказания .npy")
    ap.add_argument("--out-dir", type=Path, help="много сцен: <out-dir>/<сцена>__<tag>.npy и scenes__<tag>.json с метриками")
    ap.add_argument("--tag", default="pred", help="суффикс файлов в --out-dir, например trial12")
    ap.add_argument("--print-target", action="store_true", help="только напечатать целевой класс модели и выйти")
    args = ap.parse_args()
    if not args.print_target:
        if not args.scene:
            ap.error("нужен --scene")
        if (args.out is None) == (args.out_dir is None):
            ap.error("нужен ровно один из --out (одна сцена) и --out-dir (много сцен)")
        if args.out is not None and len(args.scene) != 1:
            ap.error("--out — для одной сцены; для нескольких используйте --out-dir")

    import torch
    # weights_only — без исполнения кода; run_info хранит torch.__version__ (TorchVersion), его разрешаем явно.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    params, info = ck["params"], ck["run_info"]
    if args.print_target:
        print(info["target"])
        return
    features = info.get("features", "xyz")
    n_points = int(info.get("patch_points", 16384))
    seed = int(info.get("seed", 42))
    threshold = float(ck.get("threshold", 0.5))  # у чекпоинтов до версии 6 порога нет
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(params, in_channels=FEATURE_CHANNELS[features]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    rows = []
    for k, scene in enumerate(args.scene, 1):
        arr = np.load(scene, mmap_mode="r", allow_pickle=False)
        out = args.out or args.out_dir / f"{scene.stem}__{args.tag}.npy"
        print(f"=== сцена {k}/{len(args.scene)}: {scene.name}", flush=True)
        # Повторный запуск не пересчитывает готовые сцены.
        # Кэш годен, только если он новее чекпоинта: иначе маска получена другой моделью/порогом.
        if args.out_dir is not None and out.exists() and out.stat().st_mtime >= args.checkpoint.stat().st_mtime:
            pred = np.load(out)
        else:
            pred = predict(model, arr, features, n_points, seed, device, torch, threshold)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, pred)
        m = scene_metrics(pred, arr, info["target"])
        print(f"=== F1={m['f1']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f} "
              f"baseline={m['baseline_f1']:.4f} не покрыто={m['not_covered']} → {out}", flush=True)
        if args.out_dir is not None:
            rows.append(dict(scene=scene.name, path=str(scene), **m))
            # После каждой сцены: прерванная оценка уже видна в ML Studio.
            (args.out_dir / f"scenes__{args.tag}.json").write_text(
                json.dumps(dict(model=dict(target=info["target"], features=features, checkpoint=str(args.checkpoint), threshold=threshold),
                                done=k, total=len(args.scene), scenes=rows), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
