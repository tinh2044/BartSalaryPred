import pandas as pd
from tqdm import tqdm
from jiwer import wer, compute_measures

def create_results_df(dataset, model, tokenizer, batch=16):
    """
    Create a DataFrame with model predictions and ground truth.
    
    Args:
        dataset: The dataset to evaluate on
        model: The model to use for predictions
        tokenizer: The tokenizer for encoding/decoding
        batch: Batch size for evaluation
        
    Returns:
        DataFrame with columns: info, ref, hyp
    """
    device = model.device
    result = {"info": [], "ref": [], "hyp": []}
    batch_data = [dataset[i:i+batch] for i in range(0, len(dataset), batch)]
    
    for batch in tqdm(batch_data, total=len(batch_data)):
        info = batch['info']
        tgt = batch['salary']
        input_ids = tokenizer.encode_src(info)['input_ids'].to(device)
        
        output_ids = model.generate(input_ids, max_length=32, num_beams=6, early_stopping=True)
        output_text = tokenizer.batch_decode(output_ids)
        
        result['info'].extend(info)
        result['ref'].extend(tgt)
        result['hyp'].extend(output_text)
        
    return pd.DataFrame(result)

def calculate_wer_for_df(df, reference_col, hypothesis_col):
    """
    Calculate Word Error Rate metrics for the given DataFrame.
    
    Args:
        df: DataFrame with prediction results
        reference_col: Column name for reference (ground truth) text
        hypothesis_col: Column name for hypothesis (prediction) text
        
    Returns:
        DataFrame with WER metrics for each sample
    """
    results = []
    
    for _, row in df.iterrows():
        reference = row[reference_col]
        hypothesis = row[hypothesis_col]
        
        if not isinstance(reference, str) or not isinstance(hypothesis, str) or not reference.strip() or not hypothesis.strip():
            results.append({
                "reference": reference,
                "hypothesis": hypothesis,
                "WER": None,
                "Substitutions": None,
                "Deletions": None,
                "Insertions": None,
                "Hits": None
            })
        else:
            wer_score = wer(reference, hypothesis)
            measures = compute_measures(reference, hypothesis)
            
            results.append({
                "reference": reference,
                "hypothesis": hypothesis,
                "WER": wer_score,
                "Substitutions": measures["substitutions"],
                "Deletions": measures["deletions"],
                "Insertions": measures["insertions"],
                "Hits": measures["hits"]
            })
    
    return pd.DataFrame(results)

def evaluate_model(model, tokenizer, train_dataset, val_dataset, output_path="./"):
    """
    Evaluate the model on training and validation datasets.
    
    Args:
        model: The trained model
        tokenizer: Tokenizer for encoding/decoding
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_path: Path to save evaluation results
        
    Returns:
        Tuple of DataFrames (val_results, train_results, val_wer, train_wer)
    """
    results_val_df = create_results_df(val_dataset, model, tokenizer, batch=32)
    val_wer = calculate_wer_for_df(results_val_df, "ref", "hyp")
    val_wer.to_csv(f"{output_path}/val_wer.csv", index=False)
    
    results_train_df = create_results_df(train_dataset, model, tokenizer, batch=32)
    train_wer = calculate_wer_for_df(results_train_df, "ref", "hyp")
    train_wer.to_csv(f"{output_path}/train_wer.csv", index=False)
    
    return results_val_df, results_train_df, val_wer, train_wer 