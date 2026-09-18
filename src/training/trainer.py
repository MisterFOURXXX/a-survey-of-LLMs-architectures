"""Training orchestration."""
import math

import pandas as pd
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# NOTE: absolute import, not relative.
# The notebooks add `src/` to sys.path, so `training` is a top-level package.
from utils.monitoring import EpochMonitor


def _resolve_strategy_conflicts(
    save_strategy: str,
    eval_strategy: str,
    load_best_model_at_end: bool,
) -> tuple[str, str, bool]:
    """
    HF requires:
      * load_best_model_at_end=True  ⇒  save_strategy == eval_strategy
      * save_strategy="no"           ⇒  load_best_model_at_end must be False
    """
    if save_strategy == "no":
        load_best_model_at_end = False
    elif load_best_model_at_end and save_strategy != eval_strategy:
        save_strategy = eval_strategy
    return save_strategy, eval_strategy, load_best_model_at_end


def build_training_args(cfg: dict) -> TrainingArguments:
    """Build TrainingArguments from a config dictionary."""
    t = cfg.get("training", {})
    m = cfg.get("model", {})

    eval_strategy = t.get("eval_strategy", "epoch")
    save_strategy = t.get("save_strategy", "epoch")
    load_best = t.get("load_best_model_at_end", True)

    save_strategy, eval_strategy, load_best = _resolve_strategy_conflicts(
        save_strategy, eval_strategy, load_best
    )

    warmup_steps = t.get("warmup_steps", 100)
    if "warmup_ratio" in t and "warmup_steps" not in t:
        approx_total_steps = 50
        warmup_steps = max(1, int(approx_total_steps * float(t["warmup_ratio"])))

    return TrainingArguments(
        output_dir=m["output_dir"],
        run_name=t.get("run_name", None),
        num_train_epochs=t.get("num_train_epochs", 3),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
        learning_rate=t.get("learning_rate", 2e-5),
        weight_decay=t.get("weight_decay", 0.01),
        warmup_steps=warmup_steps,
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        max_grad_norm=t.get("max_grad_norm", 0.5),
        logging_strategy=t.get("logging_strategy", "epoch"),
        logging_steps=t.get("logging_steps", 10),
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=t.get("save_total_limit", 2),
        load_best_model_at_end=load_best,
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        greater_is_better=t.get("greater_is_better", False),
        report_to=t.get("report_to", "none"),
        bf16=t.get("bf16", True),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        optim=t.get("optim", "adamw_torch"),
        remove_unused_columns=t.get("remove_unused_columns", False),
        dataloader_pin_memory=t.get("dataloader_pin_memory", False),
        group_by_length=t.get("group_by_length", False),
        ddp_find_unused_parameters=t.get("ddp_find_unused_parameters", False),
    )


def run_training(model, tokenizer, dataset_dict, cfg: dict):
    """Execute training and return (trainer, monitor, train_result)."""
    training_args = build_training_args(cfg)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    monitor = EpochMonitor()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=data_collator,
        callbacks=[monitor],
    )

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(cfg["model"]["output_dir"])

    return trainer, monitor, train_result


def print_training_results(trainer, monitor, train_result=None):
    """Print a formatted summary of training results."""
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)

    log_history = trainer.state.log_history
    val_losses = [log.get("eval_loss") for log in log_history if "eval_loss" in log]

    rows = []
    for i, ep in enumerate(monitor.epoch_data):
        val_loss = val_losses[i] if i < len(val_losses) else None
        rows.append({
            "Epoch": math.ceil(ep["epoch"]),
            "Validation Loss": f"{val_loss:.4f}" if val_loss is not None else "N/A",
            "Time (s)": f"{ep['time']:.1f}",
            "Avg CPU (%)": f"{ep['avg_cpu']:.1f}",
            "Avg Mem (%)": f"{ep['avg_memory']:.1f}",
        })

    if rows:
        print(pd.DataFrame(rows).to_string(index=False))

    if monitor.epoch_data:
        total_time = sum(e["time"] for e in monitor.epoch_data)
        avg_time = total_time / len(monitor.epoch_data)
        avg_cpu = sum(e["avg_cpu"] for e in monitor.epoch_data) / len(monitor.epoch_data)
        avg_mem = sum(e["avg_memory"] for e in monitor.epoch_data) / len(monitor.epoch_data)
        print(f"\nSummary Statistics:")
        print(f"  Total training time  : {total_time:,.1f} seconds")
        print(f"  Average epoch time   : {avg_time:.1f} seconds")
        print(f"  Average CPU usage    : {avg_cpu:.1f}%")
        print(f"  Average memory usage : {avg_mem:.1f}%")

    if train_result and train_result.metrics:
        print("\nFinal Training Metrics:")
        for k, v in train_result.metrics.items():
            try:
                print(f"  {k}: {float(v):.4f}")
            except (TypeError, ValueError):
                print(f"  {k}: {v}")

    print("\nTraining completed successfully!")