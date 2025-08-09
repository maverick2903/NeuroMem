import os
from models import MultiScaleEncoder
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
from scipy.stats import spearmanr
from torch.nn.functional import cosine_similarity as cos_sim

def run_quantitative_evaluation():
    if not os.path.exists(CONFIG['model_save_path']):
        print("\nModel file not found. Skipping quantitative evaluation.")
        return

    print("\n--- Running Simplified Quantitative Evaluation on STS Benchmark ---")

    # 1. Load Models
    print("Loading models...")
    muses_model = MultiScaleEncoder(
        model_name=CONFIG['model_name'], token_dim=CONFIG['token_dim'],
        sentence_dim=CONFIG['sentence_dim'], doc_dim=CONFIG['doc_dim']
    )
    muses_model.load_state_dict(torch.load(CONFIG['model_save_path']))
    muses_model.to(CONFIG['device'])

    baseline_model = SentenceTransformer(CONFIG['model_name'], device=CONFIG['device'])

    # 2. Load Test Data
    print("Loading STS benchmark dataset...")
    stsb_test = load_dataset("stsb_multi_mt", name="en", split="test")
    gold_scores = [score / 5.0 for score in stsb_test['similarity_score']]
    
    # 3. Define the manual evaluation function
    def evaluate_model(model):
        print(f"Encoding with {model.__class__.__name__}...")
        # Encode sentences in batches
        embeddings1 = model.encode(stsb_test['sentence1'], show_progress_bar=True)
        embeddings2 = model.encode(stsb_test['sentence2'], show_progress_bar=True)

        # Calculate cosine similarities
        model_scores = cos_sim(embeddings1, embeddings2).diag().tolist()

        # Calculate Spearman correlation
        spearman, _ = spearmanr(gold_scores, model_scores)
        return spearman

    # 4. Evaluate both models and get raw scores
    baseline_score = evaluate_model(baseline_model)
    muses_score = evaluate_model(muses_model)

    # 5. Print results
    print("\n--- Evaluation Results (Raw Spearman Correlation) ---")
    print(f"| {'Model':<25} | {'STS Benchmark Score':<25} |")
    print(f"| {'-'*25} | {'-'*25} |")
    print(f"| {'Baseline (all-MiniLM-L6-v2)':<25} | {baseline_score:<25.4f} |")
    print(f"| {'Your Custom MuSES Model':<25} | {muses_score:<25.4f} |")
    print("-" * 55)

    improvement = muses_score - baseline_score
    if improvement > 0.001:
        print(f"🎉 Your model shows an improvement of {improvement:.4f} points! 🎉")
    elif improvement < -0.001:
        print(f"Your model's performance is {abs(improvement):.4f} points below the baseline.")
    else:
        print("Your model's performance is about the same as the baseline.")