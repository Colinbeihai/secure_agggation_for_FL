import hashlib
import secrets
import time
from typing import Dict, List, Tuple

import torch

TensorDict = Dict[str, torch.Tensor]


def _stable_int(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big", signed=False)


def _clone_cpu(d: TensorDict) -> TensorDict:
    return {k: v.detach().cpu().clone() for k, v in d.items()}


def _flatten_weights(weights: TensorDict, names_in_order: List[str]) -> Tuple[torch.Tensor, List[Tuple[str, torch.Size]], List[int]]:
    shapes: List[Tuple[str, torch.Size]] = []
    lengths: List[int] = []
    flat_list: List[torch.Tensor] = []
    for name in names_in_order:
        t = weights[name].detach().cpu().view(-1)
        shapes.append((name, weights[name].shape))
        lengths.append(t.numel())
        flat_list.append(t)
    flat = torch.cat(flat_list) if flat_list else torch.tensor([], dtype=torch.float32)
    return flat.to(dtype=weights[names_in_order[0]].dtype if names_in_order else torch.float32), shapes, lengths


def _unflatten_vector(vec: torch.Tensor, template: TensorDict, shapes: List[Tuple[str, torch.Size]], lengths: List[int]) -> TensorDict:
    out: TensorDict = {}
    cursor = 0
    for (name, shape), length in zip(shapes, lengths):
        if length == 0:
            out[name] = torch.zeros(shape, dtype=template[name].dtype, device="cpu")
            continue
        part = vec[cursor:cursor + length].view(shape).to(dtype=template[name].dtype)
        out[name] = part.clone()
        cursor += length
    return out


class ScSecAgg:
    """
    Staircase-code-based secure aggregation (single-process simulation).
    Builds an encoder C ∈ R^{N×N} (Cauchy-like), places message blocks in a
    staircase matrix M ∈ R^{N×L}, encodes Y = C @ M. Any R available rows
    suffice (via pseudoinverse) to reconstruct M_sum, then blocks are merged
    back to the sum vector.
    """

    def __init__(self, num_servers: int = 5, read_threshold: int = 3, storage_factor: int = 2, base_seed: int | None = None, server_points: List[float] | None = None):
        assert num_servers >= 1
        assert 1 <= read_threshold <= num_servers
        self.N = int(num_servers)
        self.R = int(read_threshold)
        self.Kc = int(storage_factor)
        self.base_seed = base_seed if base_seed is not None else secrets.randbits(64)
        self.round_id = 0
        self.template: TensorDict | None = None
        self.param_names: List[str] = []
        self.total_len: int = 0
        if server_points is not None:
            assert len(server_points) == self.N
            self.alphas = [float(a) for a in server_points]
        else:
            self.alphas = [float(i + 1) for i in range(self.N)]  # 1..N
        self.C: torch.Tensor | None = None  # (N, N)
        # staircase parameters
        self.G: int = 0
        self.alpha_list: List[int] = []
        self.beta_list: List[int] = []
        self._blocks: List[Tuple[int, int, int]] = []  # (start, end, rows_used)

    def _build_generator(self, dtype: torch.dtype) -> torch.Tensor:
        # Use Cauchy-like matrix (N x N) as encoder
        x_values = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15][:self.N], dtype=dtype)
        f_values = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16][:self.N], dtype=dtype)
        C = torch.zeros((self.N, self.N), dtype=dtype, device="cpu")
        for i in range(self.N):
            for j in range(self.N):
                C[i, j] = 1.0 / (x_values[i] - f_values[j])
        return C

    def _init_staircase_params(self):
        # Based on secure_aggregation_sc.py and RDCDS intuition
        # G = N - R + 1; alpha_i = N - R + Kc + 1 - i; beta_i = N + 1 - i
        self.G = self.N - self.R + 1
        if self.G < 1:
            self.G = 1
        self.alpha_list = [self.N - self.R + self.Kc + 1 - i for i in range(1, self.G + 1)]
        self.beta_list = [self.N + 1 - i for i in range(1, self.G + 1)]
        self._blocks = []

    def _staircase_generate(self, vector: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        """
        Simplified staircase placement: produce M ∈ R^{N×L} with step-like blocks.
        """
        L = int(vector.numel())
        M = torch.zeros((self.N, L), dtype=vector.dtype, device="cpu")
        block_size = L // self.G
        remainder = L % self.G
        current_pos = 0
        for i in range(self.G):
            current_block_size = block_size + (1 if i < remainder else 0)
            if current_block_size == 0:
                continue
            block_data = vector[current_pos:current_pos + current_block_size]
            rows_to_use = max(1, min(max(self.alpha_list[i], 0), self.N))
            # place data in top rows of current step
            for row in range(rows_to_use):
                end_pos = current_pos + min(current_block_size, L - current_pos)
                if current_pos < end_pos:
                    M[row, current_pos:end_pos] = block_data[:end_pos - current_pos]
            # optional noise fill for remaining rows in the step (disabled by default)
            if noise_std > 0.0:
                max_rows = min(max(self.beta_list[i], 0), self.N)
                for row in range(rows_to_use, max_rows):
                    noise = torch.normal(0, noise_std, (current_block_size,))
                    end_pos = current_pos + min(current_block_size, L - current_pos)
                    if current_pos < end_pos:
                        M[row, current_pos:end_pos] = noise[:end_pos - current_pos]
            self._blocks.append((current_pos, current_pos + current_block_size, rows_to_use))
            current_pos += current_block_size
        return M

    def begin_round(self, round_id: int, weight_template: TensorDict):
        self.round_id = int(round_id)
        self.template = _clone_cpu(weight_template)
        self.param_names = list(self.template.keys())
        flat, shapes, lengths = _flatten_weights(self.template, self.param_names)
        self.total_len = int(flat.numel())
        self._shapes = shapes
        self._lengths = lengths
        self._dtype = flat.dtype
        # initialize staircase params and generator
        self._init_staircase_params()
        self.C = self._build_generator(dtype=self._dtype)

    def _encode_client_matrix(self, secret_vec: torch.Tensor) -> torch.Tensor:
        assert self.C is not None
        # Build staircase matrix M (N x L), then encode with C (N x N)
        M = self._staircase_generate(secret_vec, noise_std=0.0)
        Y = torch.matmul(self.C, M)  # (N, L)
        return Y


def aggregate_secure(round_id: int,
                     weight_template: TensorDict,
                     updates: List[dict],
                     online_ids: List[int],
                     instance: ScSecAgg | None = None) -> Tuple[TensorDict, dict]:
    proto = instance or ScSecAgg()
    proto.begin_round(round_id, weight_template)

    total_samples = sum(int(u["num_samples"]) for u in updates)
    if total_samples == 0:
        return weight_template, {"mask_sum_time_s": 0.0, "unmask_time_s": 0.0, "total_servers": proto.N, "read_threshold": proto.R}

    # determine code dimensions (staircase)
    L = int(proto.total_len)
    R = int(proto.R)
    code_cols = L

    # aggregated shares for each simulated server
    agg_shares = torch.zeros((proto.N, code_cols), dtype=proto._dtype, device="cpu")

    t0 = time.time()
    # accumulate encoded shares
    for u in updates:
        flat_u, _, _ = _flatten_weights({k: v.detach().cpu() for k, v in u["weights"].items()}, proto.param_names)
        flat_u = flat_u * float(u["num_samples"])  # sample-weighted
        Y = proto._encode_client_matrix(flat_u)
        agg_shares += Y
    t1 = time.time()

    # choose first R servers for decoding (simulation)
    sel_idx = list(range(R))
    # C is (N x N); select R rows -> (R x N), solve least-squares per column
    C_sel = proto.C.index_select(0, torch.tensor(sel_idx, dtype=torch.long, device="cpu"))  # (R, N)
    Y_sel = agg_shares.index_select(0, torch.tensor(sel_idx, dtype=torch.long, device="cpu"))  # (R, L)
    # Use pseudoinverse for stable reconstruction
    C_pinv = torch.linalg.pinv(C_sel)  # (N, R)
    M_sum = torch.matmul(C_pinv, Y_sel)  # (N, L)
    t2 = time.time()

    # flatten and trim to sum vector
    # Recompose sum vector from staircase blocks: average across rows used per block
    sum_vec = torch.zeros((L,), dtype=proto._dtype, device="cpu")
    for (start, end, rows_used) in proto._blocks:
        if rows_used <= 0:
            continue
        block = torch.mean(M_sum[0:rows_used, start:end], dim=0)
        sum_vec[start:end] = block

    avg_vec = sum_vec / float(total_samples)
    avg = _unflatten_vector(avg_vec, proto.template, proto._shapes, proto._lengths)

    stats = {
        "mask_sum_time_s": round(t1 - t0, 6),
        "unmask_time_s": round(t2 - t1, 6),
        "total_servers": proto.N,
        "read_threshold": proto.R,
        "storage_factor": proto.Kc,
        "code_rows": proto.R,
        "code_cols": code_cols,
        "mode": "staircase",
        "G": getattr(proto, "G", 0),
    }
    return avg, stats


