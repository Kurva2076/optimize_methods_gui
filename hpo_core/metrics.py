"""F1/Precision/Recall целевого класса, подбор порога, оценка на Val."""
import logging

import numpy as np

from .model import forward

LOG = logging.getLogger("hpo_pointnet")  # одно имя логгера на весь движок
BASELINE_MARGIN = 0.05  # ТЗ: модель должна обгонять «всё = 1» хотя бы на 5 п.п. F1
THRESHOLD_BINS = 100  # шаг подбора порога вероятности цели — 0.01


def binary_metrics(tp, fp, fn, n_neg):
    """F1/Precision/Recall только целевого класса и F1 константы «всё = 1» (baseline)."""
    n_pos = tp + fn
    if n_pos == 0:
        LOG.warning("В Val-патчах нет точек целевого класса: F1 не определён, записан 0")
    prevalence = n_pos / max(1, n_pos + n_neg)
    return {"f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, n_pos),
            "iou": tp / max(1, tp + fp + fn),
            "baseline_f1": 2 * prevalence / (1 + prevalence), "val_positive_points": n_pos,
            # Матрица ошибок: строки — истина (фон, цель), столбцы — предсказание.
            "confusion": [[n_neg - fp, fp], [fn, tp]]}


def confusion_counts(pred, y):
    """[tn, fp, fn, tp] на устройстве тензоров — без синхронизации на каждом батче."""
    import torch
    return torch.bincount(y.long() * 2 + pred.long(), minlength=4)


def counts_metrics(counts):
    tn, fp, fn, tp = (int(c) for c in counts)
    return binary_metrics(tp, fp, fn, tn + fp)


def threshold_metrics(hist):
    """Метрики при пороге вероятности цели с лучшим F1 (при равенстве — ближе к 0.5).

    hist — [2 * THRESHOLD_BINS]: гистограмма p(цель) у точек фона, затем у точек цели;
    точка считается целью при p >= k / THRESHOLD_BINS. f1_default — F1 при пороге 0.5.
    """
    neg, pos = np.asarray(hist, dtype=np.int64).reshape(2, THRESHOLD_BINS)
    tp, fp = np.cumsum(pos[::-1])[::-1], np.cumsum(neg[::-1])[::-1]  # [k] — точки с бином >= k
    n_pos, half = int(pos.sum()), THRESHOLD_BINS // 2
    f1 = 2 * tp / np.maximum(1, tp + fp + n_pos)
    k = min(range(1, THRESHOLD_BINS), key=lambda k: (-f1[k], abs(k - half)))
    m = binary_metrics(int(tp[k]), int(fp[k]), n_pos - int(tp[k]), int(neg.sum()))
    return dict(m, threshold=k / THRESHOLD_BINS, f1_default=float(f1[half]))


def evaluate(model, loader, device, cache=None, amp=False):
    import torch
    model.eval()
    hist = torch.zeros(2 * THRESHOLD_BINS, dtype=torch.long, device=device)
    with torch.no_grad():
        for feat, batch, y, keys in loader:
            feat, batch = feat.to(device), batch.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                out = forward(model, feat, batch, keys, cache)
            p = torch.softmax(out.float(), dim=1)[:, 1]
            b = (p * THRESHOLD_BINS).long().clamp_(max=THRESHOLD_BINS - 1)
            hist += torch.bincount(y.to(device).long() * THRESHOLD_BINS + b,
                                   minlength=2 * THRESHOLD_BINS)
    return threshold_metrics(hist.cpu().numpy())
