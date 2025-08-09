import torch
from torch import nn
import torch.nn.functional as F

class MuSESLoss(nn.Module):
    def __init__(self, temperature, device):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.token_proj_for_loss = nn.Linear(CONFIG['token_dim'], CONFIG['sentence_dim']).to(device)

    def _calculate_sent_doc_loss(self, anchors, positives):
        anchors = F.normalize(anchors, p=2, dim=-1)
        positives = F.normalize(positives, p=2, dim=-1)
        logits = torch.matmul(anchors, positives.T) / self.temperature
        batch_size = anchors.shape[0]
        labels = torch.arange(batch_size).to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, labels)

    def forward(self, embeddings, doc_ids):
        token_embeds = embeddings['token_embeds']
        sentence_embeds = embeddings['sentence_embeds']
        attention_mask = embeddings['attention_mask']
        batch_size, seq_len, _ = token_embeds.shape
        tokens_for_loss = self.token_proj_for_loss(token_embeds)
        tokens_norm = F.normalize(tokens_for_loss, p=2, dim=-1)
        sents_norm = F.normalize(sentence_embeds, p=2, dim=-1)
        tokens_flat = tokens_norm.view(batch_size * seq_len, -1)
        logits = torch.matmul(tokens_flat, sents_norm.T) / self.temperature
        labels = torch.arange(batch_size).to(self.device).repeat_interleave(seq_len)
        loss_unmasked = self.cross_entropy(logits, labels)
        attention_mask_flat = attention_mask.view(-1)
        token_to_sentence_loss = (loss_unmasked * attention_mask_flat).sum() / attention_mask_flat.sum()

        unique_docs, doc_counts = torch.unique(doc_ids, return_counts=True)
        valid_docs = unique_docs[doc_counts > 1]
        if len(valid_docs) > 0:
            doc_anchors, doc_positives = [], []
            for doc_id in valid_docs:
                mask = (doc_ids == doc_id)
                doc_sents = sentence_embeds[mask]
                for i in range(len(doc_sents)):
                    anchor = doc_sents[i]
                    if len(doc_sents) > 1:
                        positive = torch.cat([doc_sents[:i], doc_sents[i+1:]]).mean(dim=0)
                    else: # Should not happen with the valid_docs check, but as a safeguard
                        positive = doc_sents[i]
                    doc_anchors.append(anchor)
                    doc_positives.append(positive)

            sentence_to_document_loss = self._calculate_sent_doc_loss(torch.stack(doc_anchors), torch.stack(doc_positives))
        else:
            sentence_to_document_loss = torch.tensor(0.0, device=self.device)

        combined_loss = (CONFIG['loss_alpha'] * token_to_sentence_loss) + ((1 - CONFIG['loss_alpha']) * sentence_to_document_loss)
        return combined_loss, token_to_sentence_loss, sentence_to_document_loss
