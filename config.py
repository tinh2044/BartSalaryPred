import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    repo_id: str = "tinh2312/Bart-salary-pred"
    token: Optional[str] = None
    max_position_embeddings: int = 1024
    vocab_size: int = 2230
    
@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    output_dir: str = "./Bart-salary-pred"
    eval_strategy: str = "steps"
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    num_train_epochs: int = 100
    weight_decay: float = 0.01
    logging_dir: str = './logs'
    logging_steps: int = 100
    save_strategy: str = "steps"
    save_total_limit: int = 2
    remove_unused_columns: bool = False
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: int = 32
    dataloader_num_workers: int = 0
    report_to: str = "tensorboard"
    warmup_ratio: float = 0.5

@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    max_gen_length: int = 32
    num_beams: int = 6
    use_early_stopping: bool = True
    batch_size: int = 32
    output_path: str = "./"

def get_default_configs():
    """Get default configurations for training."""
    model_config = ModelConfig()
    training_config = TrainingConfig(
        hub_model_id=model_config.repo_id,
        hub_token=model_config.token
    )
    eval_config = EvaluationConfig()
    
    return model_config, training_config, eval_config 