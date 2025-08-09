import torch
from torch import nn
from sentence_transformers import SentenceTransformer
import tqdm

class MultiScaleEncoder(nn.Module):
    def __init__(self, model_name, token_dim, sentence_dim, doc_dim):
        super().__init__()
        self.base_model = SentenceTransformer(model_name)
        base_embedding_dim = self.base_model.get_sentence_embedding_dimension()

        self.token_proj = nn.Sequential(
            nn.Linear(base_embedding_dim, base_embedding_dim), nn.GELU(), nn.Linear(base_embedding_dim, token_dim)
        )
        self.sentence_proj = nn.Sequential(
            nn.Linear(base_embedding_dim, base_embedding_dim), nn.GELU(), nn.Linear(base_embedding_dim, sentence_dim)
        )
        self.doc_proj = nn.Sequential(
            nn.Linear(base_embedding_dim, base_embedding_dim), nn.GELU(), nn.Linear(base_embedding_dim, doc_dim)
        )

    def _mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, features):
        outputs = self.base_model(features)
        raw_token_embeds = outputs['token_embeddings']
        attention_mask = features['attention_mask']
        raw_sentence_embeds = self._mean_pooling(raw_token_embeds, attention_mask)

        projected_tokens = self.token_proj(raw_token_embeds)
        projected_sentence = self.sentence_proj(raw_sentence_embeds)
        projected_doc = self.doc_proj(raw_sentence_embeds) # Doc embeds are projected from the same raw sentence embeds

        return {
            'token_embeds': projected_tokens,
            'sentence_embeds': projected_sentence,
            'doc_embeds': projected_doc,
            'attention_mask': attention_mask
        }

    def encode(self, sentences, batch_size=32, show_progress_bar=False, **kwargs):
        self.eval()
        all_embeddings = []
        progress_bar = kwargs.get('show_progress_bar', show_progress_bar)

        for start_index in tqdm.trange(0, len(sentences), batch_size, desc="Encoding", disable=not progress_bar):
            sentences_batch = sentences[start_index:start_index + batch_size]

            features = self.base_model.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors='pt')
            features = {k: v.to(next(self.parameters()).device) for k, v in features.items()}

            with torch.no_grad():
                model_output = self.forward(features)
                embeddings = model_output['sentence_embeds']
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()