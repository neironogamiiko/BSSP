import torch
import torchmetrics

def _get_coords_from_heatmap(heatmap: torch.Tensor):
    """
    Перетворює heatmap у координати ключових точок.
    hm: (B, C, H, W)
    повертає: coords (B, C, 2) у форматі (x, y)
    """
    B, C, H, W = heatmap.shape
    heatmap_flat = heatmap.view(B, C, -1)
    idx = heatmap_flat.argmax(dim=2)  # (B, C)
    xs = idx // W
    ys = idx % W
    return torch.stack([xs, ys], dim=-1).float()  # (B, C, 2)

def PCK(threshold):
    """
    Створює метрику PCK (Percentage of Correct Keypoints) для оцінки точності передбачених
    координат landmark'ів у heatmap.

    :param threshold: максимальна допустима відстань у пікселях між передбаченою та реальною
                      точкою, щоб вважати її "правильною"
    :return pck: функція метрики, яка приймає:
                    y_true (torch.Tensor): ground truth heatmaps, розмір (B, C, H, W)
                    y_pred (torch.Tensor): передбачені heatmaps, розмір (B, C, H, W)
                і повертає float — відсоток правильно передбачених ключових точок по batch.
    """

    def pck(y_true, y_pred):
        # Вирізаємо landmark'и 1–19
        y_true = y_true[:, 1:, ...]
        y_pred = y_pred[:, 1:, ...]

        coords_true = _get_coords_from_heatmap(y_true)
        coords_pred = _get_coords_from_heatmap(y_pred)

        d = torch.norm(coords_true - coords_pred, dim=-1)  # (B, C)
        return (d < threshold).float().mean()

    return pck

def NME(normalizer):
    """
    Створює метрику NME (Normalized Mean Error) для оцінки середньої відстані між передбаченими
    і справжніми координатами landmark'ів у heatmap.

    :param normalizer: значення, на яке нормалізується відстань (наприклад, висота зображення)
    :return nme: функція метрики, яка приймає:
                    y_true (torch.Tensor): ground truth heatmaps, розмір (B, C, H, W)
                    y_pred (torch.Tensor): передбачені heatmaps, розмір (B, C, H, W)
                і повертає float — середню нормалізовану помилку по batch.
    """

    def nme(y_true, y_pred):
        y_true = y_true[:, 1:, ...]
        y_pred = y_pred[:, 1:, ...]

        coords_true = _get_coords_from_heatmap(y_true)
        coords_pred = _get_coords_from_heatmap(y_pred)

        d = torch.norm(coords_true - coords_pred, dim=-1)
        return d.mean() / normalizer

    return nme

def ClassicMetrics(num_classes: int, device: torch.device):
    """
    Створює колекцію класичних метрик.

    Метрики включають:
        - Accuracy (точність передбачення класу);
        - Precision (точність позитивних передбачень);
        - Recall (повнота позитивних передбачень);
        - F1 Score (гармонічне середнє precision і recall).

    :param num_classes: кількість класів.
    :param device: де обчислювати метрики (CPU або GPU).
    :return: torchmetrics.MetricCollection, словник із результатами всіх метрик.
    """
    return torchmetrics.MetricCollection({
        "accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes),
        "precision": torchmetrics.classification.MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall": torchmetrics.classification.MulticlassRecall(num_classes=num_classes, average="macro"),
        "F1": torchmetrics.classification.MulticlassF1Score(num_classes=num_classes, average="macro")
    }).to(device)