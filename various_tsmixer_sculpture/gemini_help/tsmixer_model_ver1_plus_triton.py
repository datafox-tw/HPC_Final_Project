"""
HPC Optimized Time-Series Mixer (TSMixer) with Triton
-----------------------------------------------------
Optimized with:
1. Channel-First memory layout for Conv1d speedup.
2. Custom Triton Kernels for RMSNorm (Memory Bandwidth Optimization).
3. Hybrid Layout Management.
4. Fixes 'OptimizedModule not iterable' by using nn.Sequential.
"""

from typing import Callable, Optional, Union, Tuple
import torch
from torch import nn
from torch.autograd import Function

from darts.logging import get_logger
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.data.torch_datasets.utils import PLModuleInput, TorchTrainingSample

logger = get_logger(__name__)

# ==========================================
# 1. Triton Kernel Definitions
# ==========================================

HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    pass

if HAS_TRITON:
    @triton.jit
    def _rms_norm_fwd_kernel(
        X_ptr, Y_ptr, W_ptr, Stride,
        N, eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        X_ptr += row_idx * Stride
        Y_ptr += row_idx * Stride

        mean_square = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            mean_square += x * x
        
        mean_square = tl.sum(mean_square, axis=0) / N
        rstd = 1 / tl.sqrt(mean_square + eps)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            w = tl.load(W_ptr + cols, mask=mask, other=1.0)
            x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = x * rstd * w
            tl.store(Y_ptr + cols, y, mask=mask)

    @triton.jit
    def _rms_norm_bwd_kernel_dx(
        DX_ptr, DY_ptr, X_ptr, W_ptr, Stride,
        N, eps,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        DX_ptr += row_idx * Stride
        DY_ptr += row_idx * Stride
        X_ptr += row_idx * Stride

        mean_square = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            mean_square += x * x
        mean_square = tl.sum(mean_square, axis=0) / N
        rstd = 1 / tl.sqrt(mean_square + eps)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            dy = tl.load(DY_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            dx = dy * w * rstd
            tl.store(DX_ptr + cols, dx, mask=mask)

    class TritonRMSNormFunc(Function):
        @staticmethod
        def forward(ctx, x, weight, eps):
            M, N = x.shape
            y = torch.empty_like(x)
            stride = x.stride(0)
            
            BLOCK_SIZE = 1024
            num_warps = 4
            if N >= 4096: BLOCK_SIZE = 4096; num_warps = 8
            elif N >= 2048: BLOCK_SIZE = 2048; num_warps = 8

            grid = (M,)
            _rms_norm_fwd_kernel[grid](
                x, y, weight, stride,
                N, eps,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps
            )
            ctx.save_for_backward(x, weight)
            ctx.eps = eps
            ctx.stride = stride
            ctx.N = N
            ctx.BLOCK_SIZE = BLOCK_SIZE
            return y

        @staticmethod
        def backward(ctx, dy):
            x, weight = ctx.saved_tensors
            dx = torch.empty_like(dy)
            grid = (x.shape[0],)
            _rms_norm_bwd_kernel_dx[grid](
                dx, dy, x, weight, ctx.stride,
                ctx.N, ctx.eps,
                BLOCK_SIZE=ctx.BLOCK_SIZE
            )
            dw = (dy * (x / (x.norm(2, dim=-1, keepdim=True) * (x.shape[-1]**-0.5) + ctx.eps))).sum(0)
            return dx, dw, None

# ==========================================
# 2. Optimized Modules
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_triton: bool = True):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.use_triton = use_triton and HAS_TRITON

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton and x.is_cuda:
            x_t = x.transpose(1, 2).contiguous() 
            shape_t = x_t.shape
            x_flat = x_t.view(-1, shape_t[-1])
            y_flat = TritonRMSNormFunc.apply(x_flat, self.scale, self.eps)
            y = y_flat.view(shape_t).transpose(1, 2)
            return y
        else:
            norm_x = x.norm(2, dim=1, keepdim=True)
            d_x = x.size(1)
            rms_x = norm_x * (d_x ** -0.5)
            return self.scale.view(1, -1, 1) * x / (rms_x + self.eps)

class _FastTimeMixing(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, dropout: float, activation: nn.Module):
        super().__init__()
        self.norm = RMSNorm(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(sequence_length, sequence_length),
            activation,
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = self.fc(x)
        return res + x

class _FastFeatureMixing(nn.Module):
    def __init__(self, input_dim: int, ff_size: int, dropout: float, activation: nn.Module):
        super().__init__()
        self.norm = RMSNorm(input_dim)
        self.conv_net = nn.Sequential(
            nn.Conv1d(input_dim, ff_size, kernel_size=1),
            activation,
            nn.Dropout(dropout),
            nn.Conv1d(ff_size, input_dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = self.conv_net(x)
        return res + x

class _FastConditionalMixerLayer(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, static_cov_dim: int, 
                 ff_size: int, activation: nn.Module, dropout: float):
        super().__init__()
        # If we have static covariates, we must project them.
        # But for Sequential compatibility, we must handle the `x_static` input carefully.
        # So we split: Standard blocks go into Sequential, Conditional part stays outside if needed.
        # HOWEVER, torch.compile prefers simple call graphs.
        
        # Strategy:
        # We will compile the main mixing block. The static addition happens before it.
        
        if static_cov_dim > 0:
            self.static_mixing = nn.Sequential(
                nn.Linear(static_cov_dim, input_dim),
                activation,
                nn.Dropout(dropout)
            )
        else:
            self.static_mixing = None

        self.block = nn.Sequential(
            _FastTimeMixing(sequence_length, input_dim, dropout, activation),
            _FastFeatureMixing(input_dim, ff_size, dropout, activation)
        )

    def forward(self, x: torch.Tensor, x_static: Optional[torch.Tensor]) -> torch.Tensor:
        if self.static_mixing is not None and x_static is not None:
            if x_static.ndim == 3: x_static = x_static.squeeze(1)
            s_emb = self.static_mixing(x_static).unsqueeze(-1)
            x = x + s_emb 
        return self.block(x)


# ==========================================
# 3. Main Module
# ==========================================

class _TSMixerModule(PLForecastingModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        past_cov_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        nr_params: int,
        hidden_size: int,
        ff_size: int,
        num_blocks: int,
        activation: str,
        dropout: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.past_cov_dim = past_cov_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        
        act_cls = getattr(nn, activation)
        act_obj = act_cls()

        # Projections
        total_hist_dim = input_dim + past_cov_dim + future_cov_dim
        self.hist_proj = nn.Linear(total_hist_dim, hidden_size)
        
        if future_cov_dim > 0:
            self.future_proj = nn.Linear(future_cov_dim, hidden_size)
        else:
            self.future_proj = None

        self.fc_time_transform = nn.Linear(self.input_chunk_length, self.output_chunk_length)

        # FIX: Blocks handling for torch.compile
        # We cannot iterate over an OptimizedModule.
        # Instead, we define self.blocks as a ModuleList, but we compile EACH block individually
        # OR we compile a wrapper Sequential if they were purely sequential.
        # But they are Conditional (take 2 args: x, x_static).
        # So we cannot use nn.Sequential directly.
        
        # SOLUTION: Compile the list elements, but keep the list as a Python list of modules.
        # torch.compile on a ModuleList compiles the __getitem__ access, but iteration fails.
        # We will compile each layer instance.
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            layer = _FastConditionalMixerLayer(
                sequence_length=self.output_chunk_length,
                input_dim=hidden_size,
                static_cov_dim=static_cov_dim,
                ff_size=ff_size,
                activation=act_obj,
                dropout=dropout
            )
            # Compile each layer individually.
            try:
                # [FIX]: 移除 mode="reduce-overhead"。
                # "reduce-overhead" 會啟用 CUDA Graphs，導致 Residual Connection 的記憶體覆寫錯誤。
                # 預設模式依然會使用 Triton 進行算子融合，效能依然很快且更穩定。
                layer = torch.compile(layer) 
            except Exception:
                pass 
            
            self.blocks.append(layer)
        if HAS_TRITON:
            logger.info("TSMixer optimized with Triton RMSNorm & torch.compile")
        else:
            logger.info("TSMixer optimized with PyTorch RMSNorm & torch.compile")

        self.head = nn.Linear(hidden_size, output_dim * nr_params)

    @io_processor
    def forward(self, x_in: PLModuleInput) -> torch.Tensor:
        x_hist, x_future, x_static = x_in
        
        # Project & Transform
        x = self.hist_proj(x_hist) 
        x = x.transpose(1, 2) # (B, H, T_in)
        x = self.fc_time_transform(x) # (B, H, T_out)
        
        if self.future_proj is not None and x_future is not None:
            x_f = self.future_proj(x_future).transpose(1, 2)
            x = x + x_f

        # Blocks (x is B, C, T)
        # Now we iterate over the ModuleList, where each element IS an OptimizedModule
        # This is valid Python.
        for block in self.blocks:
            x = block(x, x_static=x_static)

        # Output
        x = x.transpose(1, 2) 
        x = self.head(x)
        x = x.view(-1, self.output_chunk_length, self.output_dim, self.nr_params)
        
        return x

class TSMixerModel(MixedCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        hidden_size: int = 64,
        ff_size: int = 64,
        num_blocks: int = 2,
        activation: str = "ReLU",
        dropout: float = 0.1,
        norm_type: str = "RMSNorm", 
        normalize_before: bool = False,
        use_static_covariates: bool = True,
        **kwargs,
    ) -> None:
        model_kwargs = {key: val for key, val in self.model_params.items()}
        super().__init__(**self._extract_torch_model_params(**model_kwargs))

        self.pl_module_params = self._extract_pl_module_params(**model_kwargs)
        self.ff_size = ff_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.activation = activation
        self.hidden_size = hidden_size
        self.use_static_covariates = use_static_covariates

    def _create_model(self, train_sample: TorchTrainingSample) -> nn.Module:
        (past_target, past_covariates, historic_future_covariates, future_covariates, static_covariates, future_target) = train_sample
        input_dim = past_target.shape[1]
        output_dim = future_target.shape[1]
        static_cov_dim = (static_covariates.shape[0] * static_covariates.shape[1] if static_covariates is not None else 0)
        future_cov_dim = (future_covariates.shape[1] if future_covariates is not None else 0)
        past_cov_dim = past_covariates.shape[1] if past_covariates is not None else 0
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _TSMixerModule(
            input_dim=input_dim,
            output_dim=output_dim,
            future_cov_dim=future_cov_dim,
            past_cov_dim=past_cov_dim,
            static_cov_dim=static_cov_dim,
            nr_params=nr_params,
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            num_blocks=self.num_blocks,
            activation=self.activation,
            dropout=self.dropout,
            **self.pl_module_params,
        )

    @property
    def supports_static_covariates(self) -> bool:
        return True