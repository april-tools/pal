import numpy as np
from wmipa.integration.integrator import Integrator
import torch
from wmipa.wmi import WMI


def _integrate_batch_pytorch(self, problems, cache, factors=None):
    """Computes the integral of a batch of problems.

    Args:
        problems (list): The list of problems to integrate.
        cache (int): The cache level to use.
        factors (list, optional): A list of factor each problem should be multiplied by.
            Defaults to [1] * len(problems).

    """
    if isinstance(self.integrator, Integrator):
        results, cached = self.integrator.integrate_batch(problems, cache)  # type: ignore
    else:
        results, cached = zip(*(i.integrate_batch(problems, cache) for i in self.integrator))
    # check if results is pytorch tensor
    if isinstance(results, torch.Tensor):
        if factors is None:
            factors = torch.ones(len(problems), device=results.device)
        else:
            assert isinstance(factors, list)
            assert len(problems) == len(factors)
            factors = torch.tensor(factors, device=results.device)
        if len(results.shape) == 2:
            temp = results * factors.unsqueeze(0)
            v_plus = temp.clamp(min=0).sort(dim=-1, descending=False).values
            v_minus = temp.clamp(max=0).sort(dim=-1, descending=True).values
            volume = v_plus.sum(dim=-1) + v_minus.sum(dim=-1)
        else:
            temp = results * factors
            v_plus = temp.clamp(min=0).sort(dim=-1, descending=False).values
            v_minus = temp.clamp(max=0).sort(dim=-1, descending=True).values
            volume = v_plus.sum(dim=-1) + v_minus.sum(dim=-1)
        return volume, cached
    else:
        if factors is None:
            factors = [1] * len(problems)
        else:
            assert isinstance(factors, list)
            assert len(problems) == len(factors)
        cached = np.array(cached)
        results = np.array(results)
        volume = np.sum(results * factors, axis=-1)
        return volume, cached


WMI._integrate_batch = _integrate_batch_pytorch
