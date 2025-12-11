"""
HPC Optimized Time-Series Mixer (TSMixer) - v2 Convergence Optimized
--------------------------------------------------------------------
Optimized with:
1. Channel-First memory layout (Conv1d) for throughput.
2. GELU Activation & Kaiming Init for faster convergence.
3. MonteCarloDropout for better generalization (Darts compatibility).
"""

from typing import Callable, Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import init

from darts.logging import get_logger
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.data.torch_datasets.utils import PLModuleInput, TorchTrainingSample
# [Importance] Use Darts' MonteCarloDropout for consistency
from darts.utils.torch import MonteCarloDropout 

logger = get_logger(__name__)

# ==========================================
# 1. Normalization Components
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) - Normalize over Channel dimension
        norm_x = x.norm(2, dim=1, keepdim=True)
        d_x = x.size(1)
        rms_x = norm_x * (d_x ** -0.5)
        return self.scale.view(1, -1, 1) * x / (rms_x + self.eps)

class ChannelLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C) -> Norm -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)

def get_norm_layer(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "RMSNorm":
        return RMSNorm(dim)
    elif norm_type == "LayerNorm":
        return ChannelLayerNorm(dim)
    elif norm_type == "BatchNorm":
        return nn.BatchNorm1d(dim)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

# ==========================================
# 2. Optimized Blocks (GELU + Init + MC Dropout)
# ==========================================

class _FastTimeMixing(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, dropout: float, activation: nn.Module, norm_type: str):
        super().__init__()
        self.norm = get_norm_layer(norm_type, input_dim)
        
        # [Opt] Use GELU for smoother gradients
        self.fc = nn.Sequential(
            nn.Linear(sequence_length, sequence_length),
            nn.GELU(), 
            MonteCarloDropout(dropout) # [Opt] Use MC Dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = self.fc(x) 
        return res + x

class _FastFeatureMixing(nn.Module):
    def __init__(self, input_dim: int, ff_size: int, dropout: float, activation: nn.Module, norm_type: str):
        super().__init__()
        self.norm = get_norm_layer(norm_type, input_dim)
        
        # [Opt] Use GELU & MC Dropout
        self.conv_net = nn.Sequential(
            nn.Conv1d(input_dim, ff_size, kernel_size=1),
            nn.GELU(),
            MonteCarloDropout(dropout),
            nn.Conv1d(ff_size, input_dim, kernel_size=1),
            MonteCarloDropout(dropout)
        )
        # [Opt] Initialize weights for faster convergence
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = self.conv_net(x)
        return res + x

class _FastConditionalMixerLayer(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, static_cov_dim: int, 
                 ff_size: int, activation: nn.Module, dropout: float, norm_type: str):
        super().__init__()
        
        if static_cov_dim > 0:
            self.static_mixing = nn.Sequential(
                nn.Linear(static_cov_dim, input_dim),
                nn.GELU(),
                MonteCarloDropout(dropout)
            )
        else:
            self.static_mixing = None

        self.block_time = _FastTimeMixing(sequence_length, input_dim, dropout, activation, norm_type)
        self.block_feature = _FastFeatureMixing(input_dim, ff_size, dropout, activation, norm_type)

    def forward(self, x: torch.Tensor, x_static: Optional[torch.Tensor]) -> torch.Tensor:
        if self.static_mixing is not None and x_static is not None:
            if x_static.ndim == 3: x_static = x_static.squeeze(1)
            s_emb = self.static_mixing(x_static).unsqueeze(-1)
            x = x + s_emb 

        x = self.block_time(x)
        x = self.block_feature(x)
        return x

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
        norm_type: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nr_params = nr_params
        
        # We ignore the user passed activation string and enforce GELU for HPC v2
        # act_cls = getattr(nn, activation)
        act_obj = nn.GELU() 

        # Projections
        total_hist_dim = input_dim + past_cov_dim + future_cov_dim
        self.hist_proj = nn.Linear(total_hist_dim, hidden_size)
        
        if future_cov_dim > 0:
            self.future_proj = nn.Linear(future_cov_dim, hidden_size)
        else:
            self.future_proj = None

        self.fc_time_transform = nn.Linear(self.input_chunk_length, self.output_chunk_length)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                _FastConditionalMixerLayer(
                    sequence_length=self.output_chunk_length,
                    input_dim=hidden_size,
                    static_cov_dim=static_cov_dim,
                    ff_size=ff_size,
                    activation=act_obj,
                    dropout=dropout,
                    norm_type=norm_type
                )
            )

        self.head = nn.Linear(hidden_size, output_dim * nr_params)
        logger.info(f"HPC-TSMixer v2: {norm_type}, GELU, KaimingInit, MC-Dropout")

    @io_processor
    def forward(self, x_in: PLModuleInput) -> torch.Tensor:
        x_hist, x_future, x_static = x_in

        # 1. Project History
        x = self.hist_proj(x_hist) 
        x = x.transpose(1, 2) # (B, H, T_in)
        x = self.fc_time_transform(x) # (B, H, T_out)
        
        # 2. Future Covariates
        if self.future_proj is not None and x_future is not None:
            x_f = self.future_proj(x_future).transpose(1, 2)
            x = x + x_f

        # 3. HPC Blocks
        for block in self.blocks:
            x = block(x, x_static=x_static)

        # 4. Output
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
        activation: str = "GELU", # Default to GELU now
        dropout: float = 0.1,
        norm_type: str = "LayerNorm", # Recommend LayerNorm for stability first
        normalize_before: bool = False,
        use_static_covariates: bool = True,
        **kwargs,
    ) -> None:
        """
        HPC Optimized TSMixer Model v2.
        Defaulting to LayerNorm + GELU for best convergence balance.
        """
        model_kwargs = {key: val for key, val in self.model_params.items()}
        super().__init__(**self._extract_torch_model_params(**model_kwargs))

        self.pl_module_params = self._extract_pl_module_params(**model_kwargs)
        self.ff_size = ff_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.activation = activation
        self.hidden_size = hidden_size
        self.norm_type = norm_type
        self.use_static_covariates = use_static_covariates

    def _create_model(self, train_sample: TorchTrainingSample) -> nn.Module:
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            future_target,
        ) = train_sample

        input_dim = past_target.shape[1]
        output_dim = future_target.shape[1]

        static_cov_dim = (
            static_covariates.shape[0] * static_covariates.shape[1]
            if static_covariates is not None
            else 0
        )
        future_cov_dim = (
            future_covariates.shape[1] if future_covariates is not None else 0
        )
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
            norm_type=self.norm_type,
            **self.pl_module_params,
        )

    @property
    def supports_static_covariates(self) -> bool:
        return True