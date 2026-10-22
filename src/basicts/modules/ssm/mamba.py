"""Mamba state-space mixer backed by the pure-PyTorch mambapy package."""

from torch import nn

try:
    from mambapy.mamba import MambaBlock, MambaConfig
except ImportError as exc:  # pragma: no cover - exercised by optional installs
    raise ImportError(
        "Mamba models require mambapy. Install it with "
        "`python -m pip install --no-deps mambapy==1.2.0`."
    ) from exc


class MambaMixer(nn.Module):
    """Single Mamba mixer with the constructor used by mamba-ssm models."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_fast_path: bool = False,
        **_kwargs,
    ) -> None:
        super().__init__()
        del use_fast_path
        config = MambaConfig(
            d_model=d_model,
            n_layers=1,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand,
            use_cuda=False,
            pscan=True,
        )
        self.mixer = MambaBlock(config)

    def forward(self, inputs):
        return self.mixer(inputs)
