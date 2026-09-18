"""PEFT/LoRA configuration builders."""
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def build_lora_config(cfg: dict) -> LoraConfig:
    """Build LoraConfig from a dictionary."""
    return LoraConfig(
        r=cfg.get("r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        target_modules=cfg.get("target_modules"),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias=cfg.get("bias", "none"),
        task_type=cfg.get("task_type", "CAUSAL_LM"),
    )


def apply_peft(
    model,
    lora_cfg: dict,
    use_gradient_checkpointing: bool = True,
):
    """Prepare model for k-bit training and apply LoRA."""
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=use_gradient_checkpointing
    )
    lora_config = build_lora_config(lora_cfg)
    model = get_peft_model(model, lora_config)
    return model