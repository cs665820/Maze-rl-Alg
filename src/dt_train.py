import os

from datasets import DatasetDict
from transformers import DecisionTransformerConfig, Trainer, TrainingArguments

from DataCollector import DecisionTransformerGymDataCollator
from TrainableDT import TrainableDT

if __name__ == '__main__':
    os.environ["WANDB_PROJECT"] = "maze-rl-examples"
    dataset = DatasetDict.load_from_disk('episode_data/maze_dataset')
    collector = DecisionTransformerGymDataCollator(dataset["train"])

    config = DecisionTransformerConfig(
        state_dim=collector.state_dim,
        act_dim=collector.act_dim
    )
    model = TrainableDT(config)

    output_dir = "models/DT/maze_dt"

    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        num_train_epochs=120,
        per_device_train_batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        optim="adamw_torch",
        max_grad_norm=0.25,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=collector,
    )

    trainer.train()