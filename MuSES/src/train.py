import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from models import MultiScaleEncoder
from losses import MuSESLoss
import os

def train(train_examples, test_examples):
    print("\nInitializing model, loss, and optimizer...")
    model = MultiScaleEncoder(model_name=CONFIG['model_name'], token_dim=CONFIG['token_dim'], sentence_dim=CONFIG['sentence_dim'], doc_dim=CONFIG['doc_dim']).to(CONFIG['device'])
    loss_fn = MuSESLoss(temperature=CONFIG['temperature'], device=CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    def collate_fn(batch):
        texts, doc_ids = [b.texts[0] for b in batch], torch.LongTensor([b.label for b in batch])
        features = model.base_model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return features, doc_ids
    train_loader = DataLoader(train_examples, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_examples, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
    best_eval_loss = float('inf')
    for epoch in range(CONFIG['epochs']):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['epochs']} ---")
        model.train()
        total_train_loss, total_t2s_loss, total_s2d_loss = 0, 0, 0
        for features, doc_ids in tqdm.tqdm(train_loader, desc="Training"):
            features, doc_ids = {k: v.to(CONFIG['device']) for k, v in features.items()}, doc_ids.to(CONFIG['device'])
            optimizer.zero_grad()
            embeddings = model(features)
            loss, t2s_loss, s2d_loss = loss_fn(embeddings, doc_ids)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_t2s_loss += t2s_loss.item()
            total_s2d_loss += s2d_loss.item() if torch.is_tensor(s2d_loss) else s2d_loss
        print(f"Avg Train Loss: {total_train_loss / len(train_loader):.4f} | Token-Sentence Loss: {total_t2s_loss / len(train_loader):.4f} | Sentence-Doc Loss: {total_s2d_loss / len(train_loader):.4f}")

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