import numpy as np
from data_augmentation import augment_data

class SalaryPredCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch, mode="val"):
        """
        Collate function for the salary prediction model.
        
        Args:
            batch: List of samples to process
            mode: Training or validation mode ('train' or 'val')
            
        Returns:
            Dictionary with model inputs: input_ids, labels, attention_mask
        """
        infos = []
        salarys = []

        for sample in batch:
            info = sample['info'].lower()
            salary = sample['salary'].lower()
            
            if np.random.uniform(0, 1) < 0.5 and mode == 'train':
                aug_info = augment_data(info)
            else:
                aug_info = info
                
            infos.append(aug_info)
            salarys.append(salary)

        model_inputs = self.tokenizer.encode_src(infos)
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        labels = self.tokenizer.encode_tgt(salarys)['labels']
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        } 