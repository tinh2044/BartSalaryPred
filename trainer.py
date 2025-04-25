from transformers import Seq2SeqTrainer
from torch.utils.data import DataLoader

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom Seq2SeqTrainer that supports different collators for training and evaluation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_train_dataloader(self):
        """
        Returns a custom DataLoader for training with the ability to specify
        different collation modes.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            persistent_workers=self.args.dataloader_persistent_workers,
            collate_fn=lambda x: self.data_collator(x, mode="train")
        )
        
    def get_eval_dataloader(self, eval_dataset=None):
        """
        Returns a custom DataLoader for evaluation.
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            persistent_workers=self.args.dataloader_persistent_workers,
            collate_fn=lambda x: self.data_collator(x, mode="val")
        ) 