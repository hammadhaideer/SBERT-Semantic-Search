
import torch
from sentence_transformers import SentenceTransformer, util

def load_sentences(file_path):
    with open(file_path, 'r') as file:
        sentences = file.readlines()
    sentences = [sentence.strip() for sentence in sentences]
    return sentences

def main():
    # Load the SBERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Load example sentences
    sentences = load_sentences('data/example_sentences.txt')

    # Generate embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Perform semantic search
    query = "How is AI changing the industry?"
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)

    # Print results
    print(f"Query: {query}")
    for i in range(len(sentences)):
        print(f"Sentence: {sentences[i]}, Similarity Score: {cosine_scores[0][i].item():.4f}")

if __name__ == "__main__":
    main()
