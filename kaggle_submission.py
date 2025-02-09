import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

def load_data():
    """Load datasets and test files"""
    # Load BeIR datasets
    corpus_name = "BeIR/nfcorpus"
    rel_name = "BeIR/nfcorpus-qrels"
    
    dataset_queries = load_dataset(corpus_name, "queries")
    dataset_docs = load_dataset(corpus_name, "corpus")
    dataset_train = load_dataset(rel_name)["train"]  # 使用qrels数据集获取训练数据

    # Extract queries and documents
    queries = dataset_queries["queries"]["text"]
    query_ids = dataset_queries["queries"]["_id"]
    documents = dataset_docs["corpus"]["text"]
    document_ids = dataset_docs["corpus"]["_id"]

    # Create mappings for quick lookup
    query_id_to_text = dict(zip(query_ids, queries))
    doc_id_to_text = dict(zip(document_ids, documents))

    # Prepare training examples
    train_examples = []
    for item in dataset_train:
        query_id = item["query-id"]
        pos_doc_id = item["corpus-id"]
        
        if query_id in query_id_to_text and pos_doc_id in doc_id_to_text:
            query = query_id_to_text[query_id]
            positive = doc_id_to_text[pos_doc_id]
            
            # 随机选择一个负例
            neg_doc_id = np.random.choice(document_ids)
            while neg_doc_id == pos_doc_id:
                neg_doc_id = np.random.choice(document_ids)
            negative = doc_id_to_text[neg_doc_id]
            
            train_examples.append(InputExample(
                texts=[query, positive, negative]
            ))

    # Load test data
    test_queries = pd.read_csv("test_query.csv")["Query"].values.tolist()
    test_document_ids = pd.read_csv("test_documents.csv")["Doc"].values.tolist()
    test_document_ids = set(test_document_ids)
    test_documents = [doc for did, doc in zip(document_ids, documents) if did in test_document_ids]

    return documents, document_ids, test_queries, test_documents, train_examples

def fine_tune_model(train_examples, batch_size=32, num_epochs=3):
    """Fine-tune the SentenceTransformer model"""
    model_name = "all-mpnet-base-v2"
    model = SentenceTransformer(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # 训练模型
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=100,
        show_progress_bar=True
    )
    
    return model

def generate_embeddings(documents, test_queries, test_documents, model):
    """Generate embeddings using fine-tuned model"""
    # Move computation to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    document_embeddings = model.encode(documents, batch_size=32, show_progress_bar=True, device=device)
    test_query_embeddings = model.encode(test_queries, batch_size=32, show_progress_bar=True, device=device)
    return document_embeddings, test_query_embeddings

def generate_rankings(test_queries, test_query_embeddings, document_embeddings, document_ids):
    """Generate document rankings for each test query"""
    test_query_to_ranked_doc_ids = []
    
    for query, query_embedding in zip(test_queries, test_query_embeddings):
        cosine_similarities = cosine_similarity([query_embedding], document_embeddings)[0]
        sorted_doc_indices = np.argsort(cosine_similarities)[::-1][:10]
        test_query_to_ranked_doc_ids.append([
            query, 
            " ".join([document_ids[i] for i in sorted_doc_indices])
        ])
    
    return test_query_to_ranked_doc_ids

def main():
    print("Loading data...")
    documents, document_ids, test_queries, test_documents, train_examples = load_data()

    print("Fine-tuning model...")
    model = fine_tune_model(train_examples)

    print("Generating embeddings...")
    document_embeddings, test_query_embeddings = generate_embeddings(
        documents, test_queries, test_documents, model
    )

    print("Generating rankings...")
    test_query_to_ranked_doc_ids = generate_rankings(
        test_queries, 
        test_query_embeddings, 
        document_embeddings, 
        document_ids
    )

    print("Saving results...")
    submission = pd.DataFrame.from_records(
        test_query_to_ranked_doc_ids, 
        columns=["Query", "Doc_ID"]
    )
    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()