import hashlib
import secrets
import time
import torch
from typing import Dict, List, Tuple

TensorDict = Dict[str, torch.Tensor]


def _stable_int(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big", signed=False)


def _prg_like(param: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed & ((1 << 63) - 1))
    # Some torch versions do not support `generator` in randn_like; use randn with shape instead
    return torch.randn(param.cpu().shape, dtype=param.dtype, device="cpu", generator=g)


def _zeros_like(template: TensorDict) -> TensorDict:
    return {k: torch.zeros_like(v, device="cpu") for k, v in template.items()}


def _add_inplace(dst: TensorDict, src: TensorDict, alpha: float = 1.0):
    for k in dst.keys():
        dst[k] += src[k] * alpha


def _clone_cpu(d: TensorDict) -> TensorDict:
    return {k: v.detach().cpu().clone() for k, v in d.items()}


class SecAggPlus:
    """
    Educational SecAgg+ approximation using pairwise masks.
    Not a production-ready cryptographic implementation.
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

    def _pair_seed(self, i: int, j: int) -> int:
        a, b = (i, j) if i < j else (j, i)
        s = f"secaggplus|seed:{self.base_seed}|round:{self.round_id}|pair:{a}-{b}"
        return _stable_int(s)

    def _mask_for_pair(self, i: int, j: int) -> TensorDict:
        assert self.template is not None
        pair_seed = self._pair_seed(i, j)
        out = {}
        for name, p in self.template.items():
            seed = _stable_int(f"{pair_seed}|param:{name}")
            out[name] = _prg_like(p, seed)
        return out

    def client_mask(self, client_id: int) -> TensorDict:
        assert self.template is not None
        mask = _zeros_like(self.template)
        for j in self.all_ids:
            if j == client_id:
                continue
            pair_mask = self._mask_for_pair(client_id, j)
            sign = 1.0 if client_id < j else -1.0
            _add_inplace(mask, pair_mask, alpha=sign)
        return mask

    def mask_update(self, client_id: int, weighted_update: TensorDict) -> TensorDict:
        assert self.template is not None
        masked = _clone_cpu(weighted_update)
        _add_inplace(masked, self.client_mask(client_id), alpha=1.0)
        return masked

    def server_unmask_sum(self, sum_masked: TensorDict, online_ids: List[int]) -> TensorDict:
        assert self.template is not None
        online = set(online_ids)
        residual = _zeros_like(self.template)
        for i in online:
            for j in self.all_ids:
                if j == i:
                    continue
                if j not in online:
                    pair_mask = self._mask_for_pair(i, j)
                    sign = 1.0 if i < j else -1.0
                    _add_inplace(residual, pair_mask, alpha=sign)
        unmasked = _clone_cpu(sum_masked)
        _add_inplace(unmasked, residual, alpha=-1.0)
        return unmasked


def aggregate_secure(round_id: int,
                     weight_template: TensorDict,
                     updates: List[dict],
                     online_ids: List[int],
                     instance: SecAggPlus | None = None) -> Tuple[TensorDict, dict]:
    """
    updates: List of {"client_id": int, "weights": TensorDict, "num_samples": int}
    returns: averaged weights (TensorDict)
    """
    proto = instance or SecAggPlus()
    all_ids = [u["client_id"] for u in updates]
    for cid in online_ids:
        if cid not in all_ids:
            all_ids.append(cid)
    proto.begin_round(round_id, all_ids, weight_template)

    total_samples = sum(int(u["num_samples"]) for u in updates)
    sum_masked = _zeros_like(weight_template)

    t0 = time.time()
    for u in updates:
        weighted = {k: v.cpu() * float(u["num_samples"]) for k, v in u["weights"].items()}
        masked = proto.mask_update(u["client_id"], weighted)
        _add_inplace(sum_masked, masked, alpha=1.0)
    t1 = time.time()

    sum_unmasked = proto.server_unmask_sum(sum_masked, online_ids)
    t2 = time.time()

    avg = {k: sum_unmasked[k] / float(total_samples) for k in sum_unmasked.keys()}
    stats = {
        "mask_sum_time_s": round(t1 - t0, 6),
        "unmask_time_s": round(t2 - t1, 6),
        "total_clients": len(proto.all_ids),
        "online_clients": len(online_ids),
    }
    return avg, stats


