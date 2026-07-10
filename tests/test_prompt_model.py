from pathlib import Path

import torch

from src.ml.prompt_dataset import CLASS_NAMES
from src.ml.prompt_model import (
    build_prompt_model,
    freeze_for_head_training,
    load_prompt_checkpoint,
    save_prompt_checkpoint,
    unfreeze_last_feature_block,
)


def test_model_output_shape_is_batch_by_four() -> None:
    model = build_prompt_model(pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.zeros(2, 3, 96, 320))
    assert output.shape == (2, 4)


def test_model_forwards_on_cpu() -> None:
    model = build_prompt_model(pretrained=False).cpu().eval()
    with torch.inference_mode():
        output = model(torch.ones(1, 3, 96, 320))
    assert output.device.type == "cpu"


def test_head_stage_freezes_features_only() -> None:
    model = build_prompt_model(pretrained=False)
    freeze_for_head_training(model)
    assert not any(parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_finetune_stage_unfreezes_only_last_feature_block() -> None:
    model = build_prompt_model(pretrained=False)
    unfreeze_last_feature_block(model)
    assert any(parameter.requires_grad for parameter in model.features[-1].parameters())
    assert not any(parameter.requires_grad for parameter in model.features[:-1].parameters())


def test_checkpoint_round_trip_preserves_class_order_and_logits(tmp_path: Path) -> None:
    model = build_prompt_model(pretrained=False).eval()
    checkpoint = tmp_path / "model.pt"
    save_prompt_checkpoint(checkpoint, model, epoch=2, validation_macro_f1=0.8)
    loaded, metadata = load_prompt_checkpoint(checkpoint)
    sample = torch.zeros(1, 3, 96, 320)
    with torch.inference_mode():
        assert torch.equal(model(sample), loaded(sample))
    assert metadata["class_names"] == list(CLASS_NAMES)
    assert metadata["epoch"] == 2
