import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from models import MultiScaleEncoder
from losses import MuSESLoss
import os
import tqdm
import random
from torch.utils.data import Dataset

class DocGroupedDataset(Dataset):
    """
    Minimal wrapper: groups sentences by doc_id and supports sampling K docs x P sentences per doc.
    """
    def __init__(self, input_examples, min_sents_per_doc=2):
        self.doc2sents = {}
        for ex in input_examples:
            doc_id = int(ex.label)
            self.doc2sents.setdefault(doc_id, []).append(ex.texts[0])
        # keep only docs with enough sentences
        self.docs = [d for d, s in self.doc2sents.items() if len(self.doc2sents[d]) >= min_sents_per_doc]
        # fallback: keep a flattened list too (not used for grouped sampling)
        self.all = input_examples

    def sample_batch(self, K, P):
        # If not enough docs, sample with replacement to reach K
        if len(self.docs) >= K:
            chosen_docs = random.sample(self.docs, K)
        else:
            chosen_docs = random.choices(self.docs, k=K)
        texts, doc_ids = [], []
        for d in chosen_docs:
            sents = self.doc2sents[d]
            # if doc has fewer than P sents, sample with replacement
            if len(sents) >= P:
                chosen_sents = random.sample(sents, P)
            else:
                chosen_sents = random.choices(sents, k=P)
            texts.extend(chosen_sents)
            doc_ids.extend([d] * P)
        return texts, torch.LongTensor(doc_ids)


def train(train_examples, test_examples):
    print("\nInitializing model, loss, and optimizer...")
    model = MultiScaleEncoder(model_name=CONFIG['model_name'], token_dim=CONFIG['token_dim'], sentence_dim=CONFIG['sentence_dim'], doc_dim=CONFIG['doc_dim']).to(CONFIG['device'])
    loss_fn = MuSESLoss(temperature=CONFIG['temperature'], device=CONFIG['device'])
    # --- two-LR optimizer: backbone smaller LR, heads larger LR ---
    backbone_params = list(model.base_model.parameters())
    head_params = [p for n,p in model.named_parameters() if (not n.startswith('base_model')) and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': CONFIG['learning_rate_backbone']},
        {'params': head_params, 'lr': CONFIG['learning_rate_heads']}
    ], weight_decay=0.01)

    # collate_fn used only for test_loader
    def collate_fn(batch):
        texts, doc_ids = [b.texts[0] for b in batch], torch.LongTensor([b.label for b in batch])
        features = model.base_model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return features, doc_ids

    # --- Create grouped dataset for training and keep the existing test_loader ---
    grouped_dataset = DocGroupedDataset(train_examples, min_sents_per_doc=2)
    P = 4  # sentences per doc in a batch (you can tune)
    K = CONFIG['batch_size'] // P
    print(f"Grouped sampling will use K={K} docs per batch, P={P} sentences per doc => batch_size={K*P}")

    test_loader = DataLoader(test_examples, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)

    best_eval_loss = float('inf')
    steps_per_epoch = max(1, len(train_examples) // CONFIG['batch_size'])
    for epoch in range(CONFIG['epochs']):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['epochs']} ---")
        model.train()
        total_train_loss, total_t2s_loss, total_s2d_loss = 0, 0, 0

        # Manual grouped sampling loop (replaces the DataLoader training loop)
        for step in tqdm.tqdm(range(steps_per_epoch), desc="Training (grouped batches)"):
            texts, doc_ids = grouped_dataset.sample_batch(K, P)
            features = model.base_model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            features, doc_ids = {k: v.to(CONFIG['device']) for k, v in features.items()}, doc_ids.to(CONFIG['device'])

            optimizer.zero_grad()
            embeddings = model(features)
            loss, t2s_loss, s2d_loss = loss_fn(embeddings, doc_ids)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_t2s_loss += t2s_loss.item()
            total_s2d_loss += s2d_loss.item() if torch.is_tensor(s2d_loss) else s2d_loss

        print(f"Avg Train Loss: {total_train_loss / steps_per_epoch:.4f} | Token-Sentence Loss: {total_t2s_loss / steps_per_epoch:.4f} | Sentence-Doc Loss: {total_s2d_loss / steps_per_epoch:.4f}")

        # --- Evaluation: keep same evaluation code but use test_loader ---
        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for features, doc_ids in tqdm.tqdm(test_loader, desc="Evaluating"):
                features, doc_ids = {k: v.to(CONFIG['device']) for k, v in features.items()}, doc_ids.to(CONFIG['device'])
                embeddings = model(features)
                loss, _, _ = loss_fn(embeddings, doc_ids)
                total_eval_loss += loss.item()
        avg_eval_loss = total_eval_loss / len(test_loader)
        print(f"Average Evaluation Loss: {avg_eval_loss:.4f}")
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            print(f"New best model saved to {CONFIG['model_save_path']} with loss: {best_eval_loss:.4f}")