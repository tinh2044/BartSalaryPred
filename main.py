#!/usr/bin/env python
"""
Command-line interface for training the Vietnamese salary prediction model.
"""
import argparse
import os
from training import train 
from config import ModelConfig, TrainingConfig, EvaluationConfig
from pprint import pprint

def parse_args():
    """Parse command-line arguments for model training."""
    parser = argparse.ArgumentParser(
        description="Train a Bart model for Vietnamese salary prediction"
    )
    
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--repo-id", 
        type=str, 
        default="tinh2312/Bart-salary-pred",
        help="HuggingFace token for accessing the model repository (when pushing model to your own repo"
    )
    model_group.add_argument(
        "--token", 
        type=str, 
        default=None,
        help="HuggingFace token for accessing the model repository"
    )
    model_group.add_argument(
        "--max-position-embeddings", 
        type=int, 
        default=1024,
        help="Maximum position embeddings for the model"
    )
    model_group.add_argument(
        "--vocab-size", 
        type=int, 
        default=2230,
        help="Vocabulary size for the model"
    )
    
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--output-dir", 
        type=str, 
        default="./Bart-salary-pred",
        help="Directory to save model checkpoints"
    )
    train_group.add_argument(
        "--eval-strategy",
        type=str,
        default="steps",
        help="Evaluation strategy (steps or epoch)"
    )
    train_group.add_argument(
        "--learning-rate", 
        type=float, 
        default=2e-5,
        help="Learning rate for training"
    )
    train_group.add_argument(
        "--train-batch-size", 
        type=int, 
        default=16,
        help="Batch size for training"
    )
    train_group.add_argument(
        "--eval-batch-size", 
        type=int, 
        default=16,
        help="Batch size for evaluation"
    )
    train_group.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--weight-decay", 
        type=float, 
        default=0.01,
        help="Weight decay for optimizer"
    )
    train_group.add_argument(
        "--logging-dir", 
        type=str, 
        default="./logs",
        help="Directory for logging"
    )
    train_group.add_argument(
        "--logging-steps", 
        type=int, 
        default=100,
        help="Number of steps between logging"
    )
    train_group.add_argument(
        "--save-strategy", 
        type=str, 
        default="steps",
        help="Strategy for saving checkpoints"
    )
    train_group.add_argument(
        "--save-total-limit", 
        type=int, 
        default=2,
        help="Maximum number of checkpoints to save"
    )
    train_group.add_argument(
        "--warmup-ratio", 
        type=float, 
        default=0.5,
        help="Ratio of training steps for learning rate warmup"
    )
    train_group.add_argument(
        "--push-to-hub", 
        action="store_true",
        default=False,
        help="Enable pushing model to HuggingFace Hub"
    )
    train_group.add_argument(
        "--report-to",
        type=str,
        default="tensorboard",
        help="Reporting platform (tensorboard, wandb, etc.)"
    )
    train_group.add_argument(
        "--num-workers", 
        type=int, 
        default=0,
        help="Number of workers for data loading"
    )
    train_group.add_argument(
        "--prefetch-factor", 
        type=int, 
        default=4,
        help="Prefetch factor for dataloader"
    )
    train_group.add_argument(
        "--persistent-workers",
        action="store_true",
        default=False,
        help="Enable persistent workers for dataloader"
    )
    
    eval_group = parser.add_argument_group("Evaluation Configuration")
    eval_group.add_argument(
        "--max-gen-length", 
        type=int, 
        default=32,
        help="Maximum generation length for evaluation"
    )
    eval_group.add_argument(
        "--num-beams", 
        type=int, 
        default=6,
        help="Number of beams for beam search during evaluation"
    )
    eval_group.add_argument(
        "--no-early-stopping", 
        action="store_true",
        help="Disable early stopping during beam search"
    )
    eval_group.add_argument(
        "--eval-output-path", 
        type=str, 
        default="./",
        help="Path to save evaluation results"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for training."""
    args = parse_args()
    
    model_config = ModelConfig(
        repo_id=args.repo_id,
        token=args.token,
        max_position_embeddings=args.max_position_embeddings,
        vocab_size=args.vocab_size
    )
    if args.num_workers == 0:
        args.prefetch_factor = False
        args.persistent_workers = False
        
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        eval_strategy=args.eval_strategy,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.repo_id,
        hub_token=args.token,
        dataloader_num_workers=args.num_workers,
        dataloader_prefetch_factor=args.prefetch_factor,
        dataloader_persistent_workers=args.persistent_workers,
        report_to=args.report_to
    )
    
    eval_config = EvaluationConfig(
        max_gen_length=args.max_gen_length,
        num_beams=args.num_beams,
        use_early_stopping=not args.no_early_stopping,
        batch_size=args.eval_batch_size,
        output_path=args.eval_output_path
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.eval_output_path, exist_ok=True)
    
    print(f"Starting training with the following configuration:")
    pprint(f"Model: {vars(model_config)}")
    pprint(f"Training: {vars(training_config)}")
    pprint(f"Evaluation: {vars(eval_config)}")
    
    model, (val_results, train_results, val_wer, train_wer) = train(
        model_config=model_config,
        training_config=training_config,
        eval_config=eval_config
    )
    
    print("\nTraining completed!")
    print(f"Validation WER: {val_wer['WER'].mean():.4f}")
    print(f"Training WER: {train_wer['WER'].mean():.4f}")
    print(f"Results saved to {args.eval_output_path}")

if __name__ == "__main__":
    main() 