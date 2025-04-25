import os
import warnings
import torch
from transformers import (Seq2SeqTrainingArguments, BartForConditionalGeneration, 
                          BartConfig)
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from config import get_default_configs
from tokenizer import Tokenizer
from data_collator import SalaryPredCollator
from trainer import CustomSeq2SeqTrainer
from evaluation import evaluate_model
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model_and_tokenizer(model_config):
    """
    Load model and tokenizer from the specified configuration.
    
    Args:
        model_config: ModelConfig object with model parameters
        
    Returns:
        Tuple of (model, tokenizer)
    """
    prune_id_file = hf_hub_download(
        repo_id=model_config.repo_id, 
        filename="map_ids.json", 
        local_dir="./"
    )
    
    config = BartConfig.from_pretrained(model_config.repo_id)
    config.max_position_embeddings = model_config.max_position_embeddings
    config.vocab_size = model_config.vocab_size
    
    model = BartForConditionalGeneration.from_pretrained(model_config.repo_id)
    print(f"Model memory footprint: {model.get_memory_footprint()/1024**2:.2f} MB")
    
    tokenizer = Tokenizer(model_config.repo_id, prune_id_file)
    
    return model, tokenizer

def prepare_datasets():
    """
    Load and prepare datasets.
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    data = load_dataset("tinh2312/vn_jobs")
    train_dataset = data['train']
    val_dataset = data['validation']
    
    info_max_len = max(len(x.split(" ")) for x in train_dataset['info'] + val_dataset['info'])
    tgt_max_len = max(len(x.split(" ")) for x in val_dataset['salary'] + val_dataset['salary'])
    
    print(f"Maximum input length: {info_max_len}")
    print(f"Maximum target length: {tgt_max_len}")
    
    return train_dataset, val_dataset

def train(model_config=None, training_config=None, eval_config=None):
    """
    Main training function.
    
    Args:
        model_config: ModelConfig object or None to use defaults
        training_config: TrainingConfig object or None to use defaults
        eval_config: EvaluationConfig object or None to use defaults
        
    Returns:
        Trained model and evaluation results
    """
    if model_config is None or training_config is None or eval_config is None:
        model_config, training_config, eval_config = get_default_configs()
        
    if training_config.hub_token:
        os.environ["HF_HUB_TOKEN"] = training_config.hub_token
        
    if training_config.push_to_hub and training_config.hub_token is None:
        raise ValueError("Hub token is required when pushing to hub")
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    train_dataset, val_dataset = prepare_datasets()
    
    data_collator = SalaryPredCollator(tokenizer)
    
    training_args_kwargs = {
        "output_dir": training_config.output_dir,
        "eval_strategy": training_config.eval_strategy,
        "learning_rate": training_config.learning_rate,
        "per_device_train_batch_size": training_config.per_device_train_batch_size,
        "per_device_eval_batch_size": training_config.per_device_eval_batch_size,
        "num_train_epochs": training_config.num_train_epochs,
        "weight_decay": training_config.weight_decay,
        "logging_dir": training_config.logging_dir,
        "logging_steps": training_config.logging_steps,
        "save_strategy": training_config.save_strategy,
        "save_total_limit": training_config.save_total_limit,
        "remove_unused_columns": training_config.remove_unused_columns,
        "push_to_hub": training_config.push_to_hub,
        "hub_model_id": training_config.hub_model_id,
        "hub_token": training_config.hub_token,
        "dataloader_num_workers": training_config.dataloader_num_workers,
        "report_to": training_config.report_to,
        "warmup_ratio": training_config.warmup_ratio,
    }
    
    # Only add prefetch_factor and persistent_workers if num_workers > 0
    if training_config.dataloader_num_workers > 0:
        training_args_kwargs["dataloader_prefetch_factor"] = training_config.dataloader_prefetch_factor
        training_args_kwargs["dataloader_persistent_workers"] = training_config.dataloader_persistent_workers
    
    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)
    
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    warnings.filterwarnings("ignore")
    trainer.train(resume_from_checkpoint=None)
    
    if training_config.push_to_hub:
        trainer.push_to_hub()
    
    print("Evaluating model...")
    val_results, train_results, val_wer, train_wer = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_path=eval_config.output_path
    )
    
    return model, (val_results, train_results, val_wer, train_wer)

if __name__ == "__main__":
    model, evaluation_results = train()