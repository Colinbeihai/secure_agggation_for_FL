import hashlib
import secrets
import torch
from typing import Dict, List

TensorDict = Dict[str, torch.Tensor]


def _stable_int(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big", signed=False)


def _prg_like(param: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed & ((1 << 63) - 1))
    return torch.randn(param.cpu().shape, dtype=param.dtype, device="cpu", generator=g)


def _zeros_like(template: TensorDict) -> TensorDict:
    return {k: torch.zeros_like(v, device="cpu") for k, v in template.items()}


def _add_inplace(dst: TensorDict, src: TensorDict, alpha: float = 1.0):
    for k in dst.keys():
        dst[k] += src[k] * alpha


def _clone_cpu(d: TensorDict) -> TensorDict:
    return {k: v.detach().cpu().clone() for k, v in d.items()}


class FastSecAgg:
    """
    Educational FastSecAgg-like approximation using single-end PRG masks.
    Not production-secure.
    """

    def __init__(self, base_seed: int | None = None):
        self.base_seed = base_seed if base_seed is not None else secrets.randbits(64)
        self.round_id = 0
        self.all_ids: List[int] = []
        self.template: TensorDict | None = None
        self.param_names: List[str] = []

    def begin_round(self, round_id: int, all_client_ids: List[int], weight_template: TensorDict):
        self.round_id = int(round_id)
        self.all_ids = sorted(list(all_client_ids))
        self.template = _clone_cpu(weight_template)
        self.param_names = list(self.template.keys())

    def _client_seed(self, i: int) -> int:
        s = f"fastsecagg|seed:{self.base_seed}|round:{self.round_id}|client:{i}"
        return _stable_int(s)

    def _mask_for_client(self, i: int) -> Dict[str, torch.Tensor]:
        assert self.template is not None
        client_seed = self._client_seed(i)
        out = {}
        for name, p in self.template.items():
            seed = _stable_int(f"{client_seed}|param:{name}")
            out[name] = _prg_like(p, seed)
        return out

    def mask_update(self, client_id: int, weighted_update: TensorDict) -> TensorDict:
        assert self.template is not None
        masked = _clone_cpu(weighted_update)
        _add_inplace(masked, self._mask_for_client(client_id), alpha=1.0)
        return masked

    def server_unmask_sum(self, sum_masked: TensorDict, online_ids: List[int]) -> TensorDict:
        assert self.template is not None
        correction = _zeros_like(self.template)
        for i in online_ids:
            _add_inplace(correction, self._mask_for_client(i), alpha=1.0)
        unmasked = _clone_cpu(sum_masked)
        _add_inplace(unmasked, correction, alpha=-1.0)
        return unmasked


def aggregate_secure(round_id: int,
                     weight_template: TensorDict,
                     updates: List[dict],
                     online_ids: List[int],
                     instance: FastSecAgg | None = None) -> TensorDict:
    proto = instance or FastSecAgg()
    all_ids = [u["client_id"] for u in updates]
    for cid in online_ids:
        if cid not in all_ids:
            all_ids.append(cid)
    proto.begin_round(round_id, all_ids, weight_template)

    total_samples = sum(int(u["num_samples"]) for u in updates)
    sum_masked = _zeros_like(weight_template)

    for u in updates:
        weighted = {k: v.cpu() * float(u["num_samples"]) for k, v in u["weights"].items()}
        masked = proto.mask_update(u["client_id"], weighted)
        _add_inplace(sum_masked, masked, alpha=1.0)

    sum_unmasked = proto.server_unmask_sum(sum_masked, online_ids)
    avg = {k: sum_unmasked[k] / float(total_samples) for k in sum_unmasked.keys()}
    return avg


