"""Сцены, HAG, патчи и загрузчики."""
import logging
import math
import random
from pathlib import Path

import numpy as np

LOG = logging.getLogger("hpo_pointnet")  # одно имя логгера на весь движок
LABEL_COL = 0  # раскладка exports: class, x, y, z, ...
XYZ = slice(1, 4)
XY = slice(1, 3)
CLASS_NAMES = {
    0: "Не определено", 1: "Не классифицировано", 2: "Поверхность", 3: "Низкая растительность",
    4: "Средняя растительность", 5: "Высокая растительность",
    6: "Здание", 7: "Низкий шум", 9: "Вода", 18: "Высокий шум",
}
CLASS_HEADERS = ["Неопр.", "Не клас.", "Поверх.", "Низк.раст.", "Сред.раст.",
                 "Выс.раст.", "Здание", "Шум", "Вода", "Выс.шум"]
# Цель прогона -> ID исходных классов, которые становятся меткой 1.
TARGETS = {
    "ground": (2,), "vegetation": (3, 4, 5), "noise": (7, 18), "building": (6,), "water": (9,),
}
TARGET_TITLES = {"ground": "Грунт", "vegetation": "Растительность",
                 "noise": "Ложноотражённые точки", "building": "Здания", "water": "Вода"}
FEATURE_CHANNELS = {"xyz": 3, "xyz_hag": 5}
GROUND_CELL = 1.0  # м: при ~100 точках/м² в ячейке хватает точек, чтобы найти землю
ABOVE_GROUND_THRESHOLD = 0.5  # м
AUG_SCALE = (0.9, 1.1)  # диапазон случайного масштаба патча в аугментации


def generate_data_summary(data_dir):
    """Три таблицы по всем исходным сценам."""
    paths = sorted(Path(data_dir).glob("*.npy"))
    if not paths:
        raise ValueError(f"Нет файлов .npy в {data_dir}")
    rows = []
    names = dict(CLASS_NAMES)
    headers = dict(zip(CLASS_NAMES, CLASS_HEADERS))
    totals = {class_id: 0 for class_id in CLASS_NAMES}
    presence = {class_id: 0 for class_id in CLASS_NAMES}
    for path in paths:
        try:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
        except ValueError as exc:
            raise ValueError(f"Файл повреждён или скопирован не полностью: {path} ({exc})") from exc
        if arr.ndim != 2 or arr.shape[1] < 4 or len(arr) == 0:
            raise ValueError(f"Ожидается непустой массив [N,>=4]: {path}")
        if not np.isfinite(arr[:, :4]).all():
            raise ValueError(f"NaN/Inf в {path}")
        labels, frequencies = np.unique(arr[:, LABEL_COL], return_counts=True)
        if np.any(labels != np.round(labels)):
            raise ValueError(f"Нецелый ID класса в {path}")
        counts = dict(zip([int(label) for label in labels], [int(n) for n in frequencies]))
        extent = np.ptp(arr[:, XYZ], axis=0)
        rows.append((path.stem, len(arr), counts, extent))
        for class_id, n in counts.items():
            if class_id not in names:
                LOG.warning("Неизвестный ID класса %s в %s", class_id, path.name)
                names[class_id] = f"Класс {class_id}"
                headers[class_id] = f"ID {class_id}"
                totals[class_id] = presence[class_id] = 0
            totals[class_id] += n
            presence[class_id] += 1
    names = {k: v for k, v in sorted(names.items()) if totals[k]}  # только встречающиеся классы

    lines = [
        "## Таблица 1. Сводка по сценам", "",
        "Размеры вычислены как max−min в единицах исходных координат.", "",
        "| Сцена | Точек | Классов | Размер (x × y × z) |",
        "|:---|---:|---:|---:|",
    ]
    for name, count, classes, extent in rows:
        dims = " × ".join(f"{value:.1f}" for value in extent)
        lines.append(f"| {name} | {count} | {len(classes)} | {dims} |")
    lines += ["", "## Таблица 2. Доля классов по сценам (%)", "",
              "| Сцена | " + " | ".join(headers[class_id] for class_id in names) + " | Итого |",
              "|:---|" + "---:|" * (len(names) + 1)]
    for name, count, classes, _ in rows:
        shares = [f"{100 * classes[class_id] / count:.2f}" if class_id in classes else "—"
                  for class_id in names]
        lines.append(f"| {name} | " + " | ".join(shares) + " | 100.00 |")
    grand_total = sum(totals.values())
    lines += ["", "## Таблица 3. Сводка по классам (все сцены)", "",
              "| Класс | ID | Точек всего | Доля (%) | Сцен с классом |",
              "|:---|---:|---:|---:|---:|"]
    for class_id, name in names.items():
        lines.append(f"| {name} | {class_id} | {totals[class_id]} | "
                     f"{100 * totals[class_id] / grand_total:.3f} | "
                     f"{presence[class_id]} / {len(rows)} |")
    return "\n".join(lines) + "\n"


def load_scenes(data_dir):
    """Все сцены участвуют: сцены без целевого класса — честные отрицательные примеры.

    Формат файлов проверяет generate_data_summary, она вызывается в main раньше.
    """
    paths = sorted(Path(data_dir).glob("*.npy"))
    if len(paths) < 2:
        raise ValueError(f"Для Train/Val нужны хотя бы две сцены .npy в {data_dir}")
    return paths


def split_scenes(paths, seed):
    """Train/Val по целым сценам, до нарезки патчей."""
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    n_val = max(1, math.ceil(0.2 * len(paths)))
    return paths[n_val:], paths[:n_val]


def ground_grid(xyz, cell=GROUND_CELL):
    """Сетка высот земли только по геометрии (метки не используются).

    Земля в ячейке = минимум z по ней и 8 соседним. shortcut: грубая оценка,
    на крутых склонах HAG завышается; при необходимости заменить на CSF/PMF.
    """
    origin = xyz[:, :2].min(0)
    ij = np.floor((xyz[:, :2] - origin) / cell).astype(np.int64)
    h, w = ij.max(0) + 1
    grid = np.full((h + 2, w + 2), np.inf, dtype=np.float32)
    np.minimum.at(grid, (ij[:, 0] + 1, ij[:, 1] + 1), xyz[:, 2])
    ground = np.min([grid[di:di + h, dj:dj + w] for di in range(3) for dj in range(3)], axis=0)
    return origin, ground


def hag_at(xyz, origin, ground, cell=GROUND_CELL):
    ij = np.floor((xyz[:, :2] - origin) / cell).astype(np.int64)
    ij = np.clip(ij, 0, np.array(ground.shape) - 1)
    return np.maximum(xyz[:, 2] - ground[ij[:, 0], ij[:, 1]], 0).astype(np.float32)


def compute_hag(xyz, cell=GROUND_CELL):
    return hag_at(xyz, *ground_grid(xyz, cell), cell)


def patch_radius(xy, n_points):
    """Радиус круга на n_points точек при средней плотности сцены (общий для обучения и predict)."""
    area = float(np.prod(np.maximum(np.ptp(xy, axis=0), 1e-3)))
    return max(math.sqrt(n_points * area / (math.pi * len(xy))), 1e-3)


def grow_circle(d2, r, need):
    """Индексы точек в круге (d2 — квадраты расстояний); радиус растёт ×1.25, пока точек < need."""
    idx = np.flatnonzero(d2 <= r * r)
    while len(idx) < need:
        r *= 1.25
        idx = np.flatnonzero(d2 <= r * r)
    return idx, r


def boundary_index(xyz, target, radius, space="xyz", chunk_size=100000):
    """Пограничная полоса по обе стороны границы классов (перенесено из версии Авина).

    Точка пограничная, если в радиусе radius (исходные единицы, до нормировки патча) есть точка
    другого класса. Возвращает маску, индексы таких точек и индексы ближайших точек другого класса.
    """
    from scipy.spatial import cKDTree
    if not math.isfinite(radius) or radius <= 0 or space not in ("xy", "xyz"):
        raise ValueError("boundary radius должен быть > 0, space — xy или xyz")
    coords = np.asarray(xyz, dtype=np.float64)[:, :2 if space == "xy" else 3]
    target = np.asarray(target, dtype=bool)
    mask = np.zeros(len(target), dtype=bool)
    anchors, neighbors = [], []
    if target.any() and not target.all():
        for kind in (False, True):
            own = np.flatnonzero(target == kind)
            opposite = np.flatnonzero(target != kind)
            tree = cKDTree(coords[opposite])
            for start in range(0, len(own), chunk_size):
                query = own[start:start + chunk_size]
                distance, closest = tree.query(coords[query], k=1)
                valid = distance <= radius
                mask[query[valid]] = True
                anchors.append(query[valid])
                neighbors.append(opposite[closest[valid]])
    empty = np.empty(0, dtype=np.int64)
    return (mask, np.concatenate(anchors) if anchors else empty,
            np.concatenate(neighbors) if neighbors else empty)


def quota(i, fraction):
    """1, если i-й элемент попадает в долю fraction (равномерное округление по индексам)."""
    return math.floor((i + 1) * fraction + 0.5) - math.floor(i * fraction + 0.5)


class PatchDataset:
    """Патч = точки в круге (по XY) вокруг случайного центра, метки 0/1.

    Начальный радиус рассчитан на n_points при средней плотности сцены; если точек мало,
    радиус растёт, если много — случайное прореживание до n_points. Центр и прореживание
    зависят только от (seed, epoch, index); у val epoch не меняется, и патчи фиксированы.
    Координаты патча центрируются и делятся на радиус, поэтому radius_l0 задан в долях патча.
    augment (только Train): поворот вокруг Z, отражение и масштаб от того же (seed, epoch, index);
    расстояния меняются в одно число раз, поэтому порядок FPS (и FpsCache) остаётся корректным.

    Только Train, по идеям Авина (все выключены по умолчанию — тогда патчи как раньше):
    positive_fraction — доля патчей с центром на точке цели (остальные — на фоне), сцена выбирается
    среди сцен с нужным классом; boundary_fraction — доля патчей с центром на границе классов
    (в патч гарантированно попадают обе стороны); boundary_weight > 1 — множитель веса
    пограничных точек в CE/Focal (третий элемент патча). Вид патча задаётся индексом, сцена и
    центр — (seed, epoch, index), а не batch_size, поэтому кеш FPS общий для всех trials.
    """

    def __init__(self, paths, target_ids, n_points=16384, patches_per_scene=2, seed=42,
                 features="xyz", augment=False, positive_fraction=None, boundary_fraction=0.0,
                 boundary_weight=1.0, boundary_radius=0.5, boundary_space="xyz"):
        if features not in FEATURE_CHANNELS:
            raise ValueError(f"features должен быть одним из {list(FEATURE_CHANNELS)}")
        if positive_fraction is not None and not 0 < positive_fraction < 1:
            raise ValueError("positive_fraction должна быть строго между 0 и 1")
        if not 0 <= boundary_fraction < 1 or (boundary_fraction and positive_fraction is None):
            raise ValueError("boundary_fraction в [0, 1) и только вместе с positive_fraction")
        if not math.isfinite(boundary_weight) or boundary_weight < 1:
            raise ValueError("boundary_weight должен быть >= 1")
        self.paths = list(paths)
        self.target_ids = list(target_ids)
        self.n_points = n_points
        self.patches_per_scene = patches_per_scene
        self.seed = seed
        self.features = features
        self.augment = augment
        self.epoch = 0
        # Сетки земли — один раз в основном процессе: воркеры DataLoader получают их готовыми.
        self._ground = [ground_grid(np.asarray(np.load(p, mmap_mode="r", allow_pickle=False)[:, XYZ],
                                               dtype=np.float32))
                        for p in self.paths] if features == "xyz_hag" else None
        self.positive_fraction = positive_fraction
        self.boundary_fraction = boundary_fraction
        self.boundary_weight = boundary_weight
        self._positive = self._boundary = None
        if positive_fraction is not None:
            self._positive = [np.flatnonzero(np.isin(self._labels(s), self.target_ids))
                              for s in range(len(self.paths))]
            self._scenes = [[s for s, p in enumerate(self._positive) if len(p) < len(self._labels(s))],
                            [s for s, p in enumerate(self._positive) if len(p)]]
            if not all(self._scenes):
                raise ValueError("Для баланса патчей в Train нужны и точки цели, и точки фона")
        if boundary_fraction or boundary_weight > 1:
            # (маска, пограничные точки, ближайшая точка другого класса) по каждой сцене
            self._boundary = [boundary_index(np.load(p, mmap_mode="r", allow_pickle=False)[:, XYZ],
                                             np.isin(self._labels(s), self.target_ids),
                                             boundary_radius, boundary_space)
                              for s, p in enumerate(self.paths)]
            self._scenes_boundary = [s for s, b in enumerate(self._boundary) if len(b[1])]
            if boundary_fraction and not self._scenes_boundary:
                raise ValueError("В Train нет пограничных точек; проверьте boundary_radius и единицы")
            LOG.info("Пограничных точек в Train: %s (радиус %g, %s)",
                     sum(int(b[0].sum()) for b in self._boundary), boundary_radius, boundary_space)

    def _labels(self, scene):
        return np.load(self.paths[scene], mmap_mode="r", allow_pickle=False)[:, LABEL_COL]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.paths) * self.patches_per_scene

    def kind(self, index):
        """Вид патча: None — случайный центр (как раньше), 2 — граница, 1 — цель, 0 — фон."""
        if self.positive_fraction is None:
            return None
        if quota(index, self.boundary_fraction):
            return 2
        regular = index - math.floor(index * self.boundary_fraction + 0.5)  # номер среди не граничных
        return quota(regular, self.positive_fraction)

    def __getitem__(self, index):
        import torch
        rng = np.random.default_rng([self.seed, self.epoch, index])
        kind = self.kind(index)
        scene = (index // self.patches_per_scene if kind is None else
                 int(rng.choice(self._scenes_boundary if kind == 2 else self._scenes[kind])))
        arr = np.load(self.paths[scene], mmap_mode="r", allow_pickle=False)
        xy = np.asarray(arr[:, XY], dtype=np.float32)
        n = len(xy)
        required = None
        if kind == 2:
            slot = int(rng.integers(len(self._boundary[scene][1])))
            required = np.array([self._boundary[scene][1][slot], self._boundary[scene][2][slot]])
        elif kind == 1:
            required = np.array([rng.choice(self._positive[scene])])
        elif kind == 0:
            target = np.isin(np.asarray(arr[:, LABEL_COL]), self.target_ids)
            required = np.array([rng.choice(np.flatnonzero(~target))])
        anchor = int(rng.integers(n)) if required is None else int(required[0])
        center = xy[anchor]
        d2 = ((xy - center) ** 2).sum(1)
        radius = patch_radius(xy, self.n_points)
        if kind == 2:  # круг захватывает и пару с другой стороны границы
            radius = max(radius, math.sqrt(float(d2[required[1]])) * (1 + 1e-6))
        idx, r = grow_circle(d2, radius, min(self.n_points, n))
        if len(idx) > self.n_points:
            if required is None:
                idx = np.sort(rng.choice(idx, self.n_points, replace=False))
            else:  # прореживание не выбрасывает центр (и пару на границе)
                rest = idx[~np.isin(idx, required)]
                idx = np.sort(np.concatenate((rng.choice(rest, self.n_points - len(required),
                                                         replace=False), required)))
        xyz = np.asarray(arr[idx, XYZ], dtype=np.float32)
        shift = np.array([center[0], center[1], xyz[:, 2].min()], dtype=np.float32)
        feat = (xyz - shift) / r
        scale = 1.0
        if self.augment:
            a = rng.uniform(0, 2 * math.pi)
            rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=np.float32)
            if rng.random() < 0.5:
                rot[:, 0] *= -1  # отражение по x; вместе с поворотом даёт любые отражения в XY
            scale = rng.uniform(*AUG_SCALE)
            feat = np.column_stack((feat[:, :2] @ rot.T, feat[:, 2])) * scale
        if self.features == "xyz_hag":
            hag = hag_at(xyz, *self._ground[scene])  # по исходным координатам, затем тот же масштаб
            feat = np.column_stack((feat, hag / r * scale, hag > ABOVE_GROUND_THRESHOLD))
        y = np.isin(np.asarray(arr[idx, LABEL_COL]), self.target_ids).astype(np.int64)
        result = torch.from_numpy(feat.astype(np.float32)), torch.from_numpy(y)  # колонки 0–2 — координаты
        if self.boundary_weight > 1:
            w = np.where(self._boundary[scene][0][idx], self.boundary_weight, 1).astype(np.float32)
            return (*result, torch.from_numpy(w))
        return result


class Keyed:
    """Обёртка датасета: к патчу добавляется ключ (epoch, index, точек) для FpsCache.

    Длина патча задана (epoch, index) и в ключе нужна, чтобы границы патчей считались на CPU.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        feat, y, *weight = self.dataset[index]
        return (feat, y, (self.dataset.epoch, index, len(y)), *weight)


def class_counts(paths, target_ids):
    counts = np.zeros(2, dtype=np.int64)
    for p in paths:
        labels = np.load(p, mmap_mode="r", allow_pickle=False)[:, LABEL_COL]
        count1 = int(np.count_nonzero(np.isin(labels, target_ids)))
        counts += (len(labels) - count1, count1)
    return counts


def class_weights(counts, power=1.0):
    """Веса для CrossEntropyLoss по Train (среднее = 1): редкий целевой класс весит больше.

    power < 1 смягчает слишком большой вес редкого класса (у Авина для воды 0.47: 21.5 → 4.2).
    """
    if not np.all(counts):
        raise ValueError("В Train отсутствует один из классов")
    if not 0 <= power <= 1:
        raise ValueError("Степень весов классов должна быть от 0 до 1")
    return (counts.sum() / (2 * counts.astype(np.float64))) ** power


def collate_flat(items):
    import torch
    pos = torch.cat([item[0] for item in items])
    y = torch.cat([item[1] for item in items])
    batch = torch.cat([torch.full((len(item[1]),), i, dtype=torch.long)
                       for i, item in enumerate(items)])
    keys = [item[2] for item in items] if len(items[0]) > 2 else None  # есть только у Keyed
    if len(items[0]) == 4:  # веса пограничных точек
        return pos, batch, y, keys, torch.cat([item[3] for item in items])
    return pos, batch, y, keys


def make_loader(dataset, batch_size, shuffle, seed, workers=0):
    from torch.utils.data import DataLoader
    import torch
    generator = torch.Generator().manual_seed(seed)
    # Воркеры не ломают воспроизводимость: случайность патча задана (seed, epoch, index).
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      collate_fn=collate_flat, num_workers=workers)
