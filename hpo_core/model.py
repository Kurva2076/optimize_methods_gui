"""PointNet++: сборка модели, кеш FPS, прямой проход, функция потерь, seed."""
import random

import numpy as np

MODEL_NAME = "pointnet2.s3dis-area1.openpoints"
N_POINTS_L0_MAX = 4096  # верх n_points_l0 = длина закешированного порядка FPS


def seed_everything(seed):
    """Единая точка фиксации seed: random, NumPy, PyTorch (CPU/CUDA).

    Хеширование строк текущего процесса не фиксируется: PYTHONHASHSEED читается при старте
    интерпретатора и действует только на дочерние процессы (воркеры DataLoader).
    """
    import os
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # нужно для детерминизма cuBLAS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: часть операций PyG/CUDA не имеет детерминированной версии.
    torch.use_deterministic_algorithms(True, warn_only=True)


class FpsCache:
    """Порядок FPS первого SA-слоя на n точек для каждого патча, общий для всех trials.

    FPS жадный: первые k индексов порядка на n точек совпадают с FPS на k точек, а патч зависит
    только от (epoch, index). Считается батчем на устройстве модели, в основном процессе (работает
    и с DataLoader-воркерами). random_start: старт — случайная точка патча от (seed, epoch, index),
    как random_start модели в train; иначе первая точка, как у модели в eval.
    keep=False (один trial — переиспользовать нечего): порядок той же схемы, но ровно на нужное
    число точек и без хранения, поэтому режим обучения одинаков при любом --n-trials.
    """

    def __init__(self, n, random_start, seed, keep=True):
        self.n, self.random_start, self.seed, self.keep = n, random_start, seed, keep
        self.orders = {}  # ключ -> локальные индексы [n], int32 на CPU (не занимают память GPU)

    def __call__(self, pos, keys, n):
        """Глобальные индексы [патчей, n] первых n точек FPS каждого патча батча."""
        import torch
        if self.keep and n > self.n:
            raise ValueError(f"n_points_l0={n} больше длины кеша FPS {self.n}")
        sizes = [k[2] for k in keys]
        starts = np.cumsum([0] + sizes[:-1]).tolist()  # границы патчей на CPU, без синхронизаций GPU
        todo = [b for b, k in enumerate(keys) if not self.keep or k not in self.orders]
        if todo:
            from torch_pointcloud.ops.cluster import fps
            m = self.n if self.keep else n
            # Сдвиг каждого патча по кругу делает стартовой точкой выбранную: FPS стартует с первой.
            perm = torch.cat([
                (torch.arange(sizes[b], device=pos.device) +
                 (int(np.random.default_rng([self.seed, *keys[b][:2], 1]).integers(sizes[b]))
                  if self.random_start else 0)) % sizes[b] + starts[b]
                for b in todo])
            sub_batch = torch.repeat_interleave(
                torch.arange(len(todo), device=pos.device),
                torch.tensor([sizes[b] for b in todo], device=pos.device))
            idx = perm[fps(pos[perm], sub_batch, num_nodes=m, random_start=False)].view(len(todo), m)
            if not self.keep:
                return idx  # todo = весь батч, индексы уже глобальные
            local = (idx - torch.tensor([starts[b] for b in todo], device=pos.device)[:, None]).cpu()
            for b, row in zip(todo, local.to(torch.int32)):
                self.orders[keys[b]] = row.clone()  # clone: не держать весь [патчей, n]
        order = torch.stack([self.orders[k][:n] for k in keys]).to(pos.device).long()
        return order + torch.tensor(starts, device=pos.device)[:, None]


def lovasz_softmax(probas, labels):
    """Lovász-Softmax (Berman et al., CVPR 2018) по всем точкам батча, среднее по классам батча.

    probas — [N, C]; https://github.com/bermanmaxim/LovaszSoftmax (MIT), перенесено из версии Авина.
    """
    import torch
    losses = []
    for c in range(probas.shape[1]):
        fg = (labels == c).to(probas.dtype)
        if not bool(fg.any()):
            continue
        errors, order = (fg - probas[:, c]).abs().sort(descending=True)
        fg = fg[order]
        jaccard = 1 - (fg.sum() - fg.cumsum(0)) / (fg.sum() + (1 - fg).cumsum(0))
        losses.append(torch.dot(errors, torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))))
    return torch.stack(losses).mean() if losses else probas.sum() * 0


def make_criterion(name, weights, gamma=2.0, lovasz_weight=0.5):
    """CE, Focal Loss (Lin et al., 2017) с весами классов как alpha, или (1-w)·Focal + w·Lovász.

    Вызов: criterion(logits, y, point_weight=None); point_weight множит веса точек в CE/Focal
    (Lovász весов не использует: он оптимизирует IoU напрямую).
    """
    import torch
    import torch.nn.functional as F
    if name not in ("ce", "focal", "focal_lovasz"):
        raise ValueError(f"Неизвестная функция потерь: {name}")

    def base(logits, y, point_weight=None):
        ce = F.cross_entropy(logits, y, reduction="none")
        w = weights[y] if point_weight is None else weights[y] * point_weight
        if name != "ce":
            ce = (1 - torch.exp(-ce)) ** gamma * ce
        # Нормировка как у взвешенного CE (mean): при gamma=0 focal совпадает с CrossEntropyLoss.
        return (w * ce).sum() / w.sum()

    if name != "focal_lovasz":
        return base
    return lambda logits, y, point_weight=None: (
        (1 - lovasz_weight) * base(logits, y, point_weight)
        + lovasz_weight * lovasz_softmax(logits.float().softmax(dim=1), y))


def forward(model, feat, batch, keys=None, cache=None):
    """Модель на батче; с cache первый SA-слой берёт префикс закешированного FPS."""
    pos = feat[:, :3].contiguous()
    if cache is None or keys is None:
        return model(feat, pos, batch)
    first = model.encoder.sa_blocks[0]
    idx = cache(pos, keys, first.num_points).reshape(-1)
    # Хук живёт один вызов: у модели не остаётся состояния, и прямой model(...) считает FPS сам.
    handle = first.register_forward_pre_hook(
        lambda module, args, kwargs: (args, dict(kwargs, idx=idx)), with_kwargs=True)
    try:
        return model(feat, pos, batch)
    finally:
        handle.remove()


def build_model(params, in_channels=3):
    from torch_pointcloud.models import create_model, model_info
    # Registry задаёт 4 блока; переопределяем согласованные encoder/decoder списки.
    base = model_info(MODEL_NAME, task="semantic-segmentation")["hparams"]
    depth = params["n_sa_layers"]
    channels = base["sa_channels"][:depth]
    r = params["radius_l0"]
    k = params["nsample"]
    if params["use_MSG"]:
        # Две шкалы с той же суммарной шириной выходного блока.
        channels = [[[max(8, c // 2) for c in layer]] * 2 for layer in channels]
        radii = [(r * 2 ** i, r * 2 ** (i + 1)) for i in range(depth)]
        neighbors = [(max(8, k // 2), k) for _ in range(depth)]
    else:
        radii = [r * 2 ** i for i in range(depth)]
        neighbors = [k] * depth
    model = create_model(
        MODEL_NAME, task="semantic-segmentation", pretrained=False,
        in_channels=in_channels, num_classes=2, sa_channels=channels,
        fp_channels=base["fp_channels"][-depth:],
        ratios=base["ratios"][:depth], radii=radii,
        num_neighbors=neighbors, dropout=params["dropout"],
        normalize_pos=False,
    )
    # У версии 0.0.16 сегментационный конструктор принимает ratios, но не num_points.
    # Первый SA-блок поддерживает абсолютное num_points; остальные оставляем по ratios.
    first = model.encoder.sa_blocks[0]
    first.ratio = None
    first.num_points = params["n_points_l0"]
    return model
