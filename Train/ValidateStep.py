import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Callable

def val_step(model: nn.Module,
             val_loader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             pck_fn: Callable,
             nme_fn: Callable,
             classic_metrics: Dict[str, nn.Module]) -> Dict[str, float]:
    """
    Виконує один крок валідації для multi-output моделі.

    :param model: Нейронна мережа для валідації.
    :param val_loader: DataLoader з валідаційним набором даних. Повертає кортеж (images, targets).
    :param criterion: Функція втрат (наприклад, nn.CrossEntropyLoss()).
    :param device: Пристрій для обчислень (CPU або GPU).
    :param pck_fn: Функція для обчислення метрики PCK (Percentage of Correct Keypoints). Приймає (targets, predictions) і повертає float.
    :param nme_fn: Функція для обчислення метрики NME (Normalized Mean Error). Приймає (targets, predictions) і повертає float.
    :param classic_metrics: Словник з "класичними" метриками з torchmetrics. Ключ — назва метрики, значення — об'єкт метрики.
    :return: Dict[str, float]: Словник з усередненими метриками для всього val_loader, включаючи:
                          - 'loss': усереднений loss по всіх зразках
                          - 'PCK': усереднена метрика PCK по всіх зразках
                          - 'NME': усереднена метрика NME по всіх зразках
                          - інші метрики з classic_metrics, обчислені через .compute()
    """
    model.eval()
    total_loss = 0.0
    metric_sums = {'PCK': 0.0, 'NME': 0.0}
    total_samples = 0

    with torch.inference_mode():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            batch_size = images.size(0)
            total_samples += batch_size

            targets_idx = targets.argmax(dim=1)

            outs = model(images)
            if not isinstance(outs, (list, tuple)):
                outs = [outs]

            losses = []
            pck_batch, nme_batch = 0.0, 0.0

            for i, o in enumerate(outs):
                if o.shape[2:] != targets.shape[2:]:
                    o_resized = nn.functional.interpolate(
                        o, size=targets.shape[2:], mode='bilinear', align_corners=False
                    )
                else:
                    o_resized = o

                losses.append(criterion(o_resized, targets_idx))

                o_for_metrics = o_resized
                pck_batch += pck_fn(targets, o_for_metrics) * batch_size
                nme_batch += nme_fn(targets, o_for_metrics) * batch_size

                pred_idx = o_for_metrics.argmax(dim=1)
                for metric in classic_metrics.values():
                    metric.update(pred_idx, targets_idx)

            total_loss += sum(losses).item() * batch_size
            metric_sums['PCK'] += pck_batch
            metric_sums['NME'] += nme_batch

    avg_metrics = {
        'loss': total_loss / total_samples,
        'PCK': metric_sums['PCK'] / total_samples,
        'NME': metric_sums['NME'] / total_samples,
    }

    classic_results = {k: m.compute().item() for k, m in classic_metrics.items()}
    avg_metrics.update(classic_results)

    for m in classic_metrics.values():
        m.reset()

    return avg_metrics

# Висновок по val_step:
# Логіка обчислення мульти-loss і метрик повністю відповідає TF.
# Батчова агрегація PCK/NME та класичних метрик коректна.
# Можна вважати стабільною версією для валідації, готовою до тестування та порівняння зі сходженням TF моделі.