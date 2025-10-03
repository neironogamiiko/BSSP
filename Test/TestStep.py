import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Callable

def test_step(model: nn.Module,
              test_loader: DataLoader,
              device: torch.device,
              pck_fn: Callable,
              nme_fn: Callable,
              classic_metrics: Dict[str, nn.Module]) -> Dict[str, float]:
    """
    Крок тестування для multi-output моделі.
    Без loss, тільки метрики.
    """
    model.eval()
    metric_sums = {'PCK': 0.0, 'NME': 0.0}
    total_samples = 0

    with torch.inference_mode():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            batch_size = images.size(0)
            total_samples += batch_size

            outs = model(images)
            if not isinstance(outs, (list, tuple)):
                outs = [outs]

            pck_batch, nme_batch = 0.0, 0.0

            for o in outs:
                if o.shape[2:] != targets.shape[2:]:
                    o_for_metrics = nn.functional.interpolate(
                        o, size=targets.shape[2:], mode='bilinear', align_corners=False
                    )
                else:
                    o_for_metrics = o

                pck_batch += pck_fn(targets, o_for_metrics) * batch_size
                nme_batch += nme_fn(targets, o_for_metrics) * batch_size

                pred_idx = o_for_metrics.argmax(dim=1)
                for metric in classic_metrics.values():
                    metric.update(pred_idx, targets.argmax(dim=1))

            metric_sums['PCK'] += pck_batch
            metric_sums['NME'] += nme_batch

    avg_metrics = {
        'PCK': metric_sums['PCK'] / total_samples,
        'NME': metric_sums['NME'] / total_samples,
    }

    classic_results = {k: m.compute().item() for k, m in classic_metrics.items()}
    avg_metrics.update(classic_results)

    for m in classic_metrics.values():
        m.reset()

    return avg_metrics

# Висновок по test_step:
# Логіка тестового кроку повністю відповідає TF:
# Метрики PCK/NME → однаково.
# Батчове усереднення → як TF.
# Класичні метрики → аналогічно TF.
# Loss не обчислюється → як у TF тестуванні.
# Можна вважати стабільною версією, готовою до використання для порівняння точності та швидкості сходження.