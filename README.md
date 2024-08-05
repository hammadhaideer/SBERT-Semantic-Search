# SBERT Semantic Search: Advanced Sentence Embeddings for Natural Language Processing

This repository contains a comprehensive example of using Sentence-BERT (SBERT) for semantic search. SBERT is a powerful model for generating sentence embeddings, which can be used for a variety of NLP tasks, including semantic search, text clustering, and more.

## Requirements

- Python 3.7+
- Sentence Transformers library

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/sbert-semantic-search.git
cd sbert-semantic-search
pip install -r requirements.txt
```
## ***Usage***

Run the semantic_search.py script to perform semantic search on a list of example sentences:

```bash
python semantic_search.py
```
You can add your own sentences to the data/example_sentences.txt file to customize the search.

## ***Code Explanation***

The semantic_search.py script loads a pre-trained SBERT model, generates embeddings for a list of sentences, and performs semantic search using cosine similarity.

## ***License***

This project is licensed under the MIT License.
