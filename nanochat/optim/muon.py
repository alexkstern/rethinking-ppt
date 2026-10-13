"""
Muon optimizer from Keller et al.
Also a lot of borrowing of ideas from modded-nanogpt.
"""
import torch
from torch import Tensor
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration computing the zeroth power / orthogonalization of G.
    The quintic coefficients are chosen to maximize the slope at zero, pushed past the
    point where the iteration converges to one everywhere on the interval, which minimizes
    the step count. The result is therefore not UV^T but roughly US'V^T with
    S_{ii}' ~ Uniform(0.5, 1.5), which does not hurt model performance.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz.
    https://kellerjordan.github.io/posts/muon/

    SGD-momentum, then an orthogonalization post-processing step replacing each 2D
    parameter's update with the nearest orthogonal matrix, via a Newton-Schulz
    iteration that runs stably in bfloat16 on the GPU.

    Not for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those belong to a standard method such as AdamW.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)


class DistMuon(torch.optim.Optimizer):
    """Muon that performs its own distributed synchronization:
      - reduce_scatter(AVG) for gradient averaging
      - all_gather to replicate updated weights

    2D parameters only. Momentum buffers live only on each parameter's 'owner'
    rank (block-cyclic assignment, see `step`), so optimizer state checkpointed
    from a single rank must be consolidated first.
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"
        rank = dist.get_rank()
        shapes = sorted({p.shape for p in params}) # deterministic ordering across ranks
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)
            if rank == 0:
                print(f"Muon: Grouping {len(group_params)} params of shape {shape}, device {device}, dtype {dtype}")
            param_groups.append(dict(params=group_params, zero_buffer=torch.zeros_like(group_params[0])))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), "All params must have grads"

        # Average gradients across ranks, one async reduce_scatter per group of world_size params.
        all_reduce_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            for base_i in range(0, len(params), world_size):
                # The compute owner of param i is rank i % world_size; each rank
                # receives the averaged grad of the one param it owns.
                owner_idx = base_i + rank
                rs_input = [p.grad for p in params[base_i:base_i + world_size]]
                rs_input.extend([zero_buffer] * (world_size - len(rs_input)))  # pad the final group
                rs_output = params[owner_idx].grad if owner_idx < len(params) else torch.empty_like(zero_buffer)
                work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
                all_reduce_futures.append(work)

        # Each owner rank computes its update, then all_gather replicates it everywhere.
        future_idx = 0
        all_gather_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            for base_i in range(0, len(params), world_size):
                owner_idx = base_i + rank
                all_reduce_futures[future_idx].wait()
                future_idx += 1
                if owner_idx < len(params):
                    p = params[owner_idx]
                    g = p.grad  # averaged across ranks
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1.0 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                    p.add_(g, alpha=-group["lr"] * scale)
                ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
                ag_output = params[base_i:base_i + world_size]
                ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))]) # pad
                work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
                all_gather_futures.append(work)

        torch.futures.collect_all(all_gather_futures).wait()
