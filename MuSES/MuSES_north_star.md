
---

# **NeuroMem Module 1: Multi-Scale Embedding System**
### **Technical Blueprint & Implementation Guide**

This document details the project idea, proposed solution, novelty, and technical specifications for building the foundational module of the NeuroMem architecture.

---

## 🎯 **Project Vision & Core Problem**

### The Problem with "Flat" Embeddings

Standard embedding models like `sentence-transformers` are incredibly powerful, but they operate on a single, "flat" level of understanding. They take a piece of text (a word, sentence, or paragraph) and map it to a single vector. This process loses the rich, hierarchical structure of language.

For example, the model doesn't inherently know that the meaning of a sentence is derived from the meaning of its words, or that a document's theme is an aggregation of its sentences' ideas. This is a critical limitation for advanced reasoning and memory systems.

### Our Proposed Solution: Hierarchical Embeddings

We will build a custom embedding system that represents information at **three distinct scales** simultaneously:
1.  **Token Level**: Captures the meaning of individual words or subwords.
2.  **Sentence Level**: Captures the semantic meaning of a complete sentence.
3.  **Document Level**: Captures the overarching theme or topic of a larger block of text.

The key is that these embeddings will be **linked**. We will train the model to understand that a sentence's embedding is composed of its token embeddings, creating a true hierarchical representation of the text.



---

## ✨ **Why This is Novel**

The innovation here is not just creating embeddings of different sizes, but training them to be **aware of their own hierarchy** through a specialized contrastive learning process.

| Standard Approach | Our Novel Approach |
| :--- | :--- |
| Single vector per input. | A single input text yields **multiple, inter-related vectors** at different scales. |
| Ignores the compositional nature of language. | Explicitly models the **compositional hierarchy** (tokens form sentences). |
| Similarity is "flat". | Similarity can be checked at the word, sentence, or document level for more nuanced retrieval. |

This creates a much richer data structure for the downstream memory system, allowing an LLM to "zoom in" on word-level details or "zoom out" to get the gist of a conversation.

---

## 🛠️ **Technical Deep Dive**

This section outlines the specific architecture, tools, and processes you'll use.

### Model Architecture

Our `MultiScaleEncoder` will be a custom PyTorch module that wraps a pre-trained `sentence-transformer` model. The data will flow through it in three main steps:



1.  **Base Transformer Model**: We'll use a pre-trained model like `all-MiniLM-L6-v2` as our foundation. Given an input sentence, it outputs embeddings for each token.
2.  **Pooling Layer**: We will implement a simple **mean pooling** layer. To get the sentence-level representation, we will average the embeddings of all its tokens (respecting the attention mask).
3.  **Projection Heads**: The raw embeddings from the base model and pooling layer are not yet in our desired final dimensions, nor are they optimized for our hierarchical task. We'll add small neural networks (projection heads, typically `Linear -> ReLU -> Linear`) to transform these raw embeddings into our final, multi-scale output vectors (`token_embeds`, `sentence_embeds`, etc.).

### The Training Process: Hierarchical Contrastive Learning

This is the heart of the module. We will use a contrastive loss function to teach the model about the hierarchy. For a given batch of sentences:

* **Positive Pair 👍**: The embedding for a token is a "positive" match with the embedding of the sentence it belongs to.
* **Negative Pair 👎**: The embedding for that same token is a "negative" match with the embeddings of all *other* sentences in the batch.

The model's goal is to maximize the similarity of positive pairs while minimizing the similarity of negative pairs. We achieve this using the **InfoNCE loss** formula:

$$
L_{i} = -\log \frac{\exp(\text{sim}(t_i, s_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(t_i, s_j) / \tau)}
$$

Where:
* $t_i$ is the embedding of a token from sentence $i$.
* $s_i$ is the embedding of the sentence $i$ (the positive pair).
* $s_j$ represents all sentence embeddings in the batch of size $N$ (the negative pairs).
* `sim()` is the cosine similarity (calculated via dot product of normalized vectors).
* $\tau$ is the temperature hyperparameter, which controls the sharpness of the distribution.

### Key Components & Tools

* **Primary Library**: **PyTorch**
* **Transformer Models**: **`sentence-transformers`** library by Hugging Face
* **GPU Acceleration**: **CUDA**
* **Experimentation**: **Jupyter Notebooks** (for testing and visualization)
* **Hardware Strategy**: Develop locally on your RTX 3050, train larger jobs on Google Colab.

### Project Directory Structure

A clean, organized folder structure is essential for long-term maintainability.

```bash
neuromem_module_1/
├── configs/
│   └── M1_config.yaml
├── data/
│   └── sample_corpus.txt
├── src/
│   ├── data_loader.py
│   ├── models.py
│   ├── losses.py
│   └── train.py
├── notebooks/
│   └── 01_embedding_analysis.ipynb
└── README.md
```

### Implementation Workflow

Follow these steps to build the module:

1.  **Setup**: Create the directory structure above.
2.  **Data Loading (`data_loader.py`)**: Implement a PyTorch `Dataset` that reads your text corpus. The `sentence-transformers` library has specialized data loaders you can use.
3.  **Model (`models.py`)**: Implement the `MultiScaleEncoder` class as described in the architecture section.
4.  **Loss Function (`losses.py`)**: Implement the `HierarchicalContrastiveLoss` class.
5.  **Training Script (`train.py`)**: Write the main script that initializes the model, loss function, and optimizer, and then runs the training loop.
6.  **Debug**: Run the training script with a very small batch size (e.g., 2) and a small subset of your data to ensure everything works end-to-end.
7.  **Train**: Once debugged, run the full training process on Google Colab for best performance.