"""
metrics.py - confusion matrix, overall accuracy, mIoU and per-class IoU, F1, precision and recall.
Classes that are absent from the reference are left out of the mean values.
"""
import numpy as np

from config import NUM_CLASSES, CLASSES, IGNORE_INDEX


def confusion(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rows = true class, cols = predicted class. Pixels labelled IGNORE_INDEX are skipped."""
    mask = (y != IGNORE_INDEX) & (y < NUM_CLASSES)
    return np.bincount(NUM_CLASSES * y[mask].astype(np.int64) + pred[mask].astype(np.int64),
                       minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def summarize(cm: np.ndarray) -> dict:
    """OA, mIoU, mean F1 and per-class IoU / F1 / precision / recall from a confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    support = cm.sum(1)
    predicted = cm.sum(0)
    union = support + predicted - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
        prec = np.where(predicted > 0, tp / predicted, np.nan)
        rec = np.where(support > 0, tp / support, np.nan)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), np.nan)
    present = support > 0
    f = lambda a: [None if np.isnan(v) else float(v) for v in a]
    return {
        "overall_acc": float(tp.sum() / cm.sum()) if cm.sum() > 0 else float("nan"),
        "mIoU": float(np.nanmean(iou[present])) if present.any() else float("nan"),
        "mF1": float(np.nanmean(f1[present])) if present.any() else float("nan"),
        "iou": dict(zip(CLASSES, f(iou))),
        "f1": dict(zip(CLASSES, f(f1))),
        "precision": dict(zip(CLASSES, f(prec))),
        "recall": dict(zip(CLASSES, f(rec))),
        "support_pixels": dict(zip(CLASSES, [int(v) for v in support])),
        "confusion_matrix": cm.astype(np.int64).tolist(),
    }


def class_counts(label: np.ndarray) -> np.ndarray:
    v = label[(label != IGNORE_INDEX) & (label < NUM_CLASSES)]
    return np.bincount(v.ravel(), minlength=NUM_CLASSES)