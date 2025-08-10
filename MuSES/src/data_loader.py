import random
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import InputExample
import tqdm
from models import MultiScaleEncoder
from losses import MuSESLoss

def prepare_data():
    """
    Loads and processes the Wikipedia dataset to create a hierarchical structure
    of documents (paragraphs) and sentences.
    """
    # Download the 'punkt' sentence tokenizer from NLTK if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK sentence tokenizer (punkt)...")
        nltk.download('punkt')

    print("\nLoading Wikipedia dataset...")
    # We use 'train[:2%]' to load only the first 2% of the dataset for faster experimentation.
    # For a full run, you could increase this or remove the slice altogether.
    wiki_dataset = load_dataset('wikipedia', '20220301.en', split='train[:500]', trust_remote_code=True)

    train_examples = []
    # Each paragraph with multiple sentences will be treated as a unique "document"
    # for the purpose of the sentence-to-document loss.
    doc_id_counter = 0

    print("Processing articles into paragraphs and sentences...")
    for article in tqdm.tqdm(wiki_dataset):
        # Split the article's text into paragraphs based on double newlines
        paragraphs = article['text'].split('\n\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            # Filter out very short paragraphs or section titles
            if len(paragraph) < 100:
                continue

            # Use NLTK to robustly split the paragraph into sentences
            sentences = sent_tokenize(paragraph)

            # We only care about paragraphs that have more than one sentence
            if len(sentences) > 1:
                for sentence in sentences:
                    # The label is the ID of the paragraph (our "document")
                    train_examples.append(InputExample(texts=[sentence], label=doc_id_counter))
                # Increment the ID for the next paragraph
                doc_id_counter += 1

    # Split all the sentences we've gathered into a train and test set.
    train_size = int(len(train_examples) * 0.9)
    train_data, test_data = train_examples[:train_size], train_examples[train_size:]

    print(f"\nTotal paragraphs processed as 'documents': {doc_id_counter}")
    print(f"Total sentences loaded: {len(train_examples)}")
    print(f"Number of training examples (sentences): {len(train_data)}")
    print(f"Number of testing examples (sentences): {len(test_data)}")
    return train_data, test_data