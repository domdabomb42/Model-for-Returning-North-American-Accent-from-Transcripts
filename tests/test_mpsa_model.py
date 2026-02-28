import torch

from conase_geo.models.mpsa_densenet import MPSAConfig, MPSADenseNet


def test_mpsa_densenet_forward_shape() -> None:
    model = MPSADenseNet(
        num_classes=7,
        config=MPSAConfig(
            block_config=(2, 2, 2, 2),
            growth_rate=16,
            num_init_features=32,
            bn_size=2,
            drop_rate=0.0,
            in_channels=2,
        ),
    )
    x = torch.randn(2, 2, 64, 96)
    out = model(x)
    assert "accent" in out
    assert out["accent"].shape == (2, 7)
