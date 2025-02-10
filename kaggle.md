# Document Ranking System with SentenceTransformer

This code implements a document ranking system using SentenceTransformer for semantic search. Here's a breakdown of the process:

1. **Data Loading (`load_data` function)**
   - Loads BeIR datasets (nfcorpus) containing queries and documents
   - Creates mappings between IDs and text for queries and documents
   - Prepares training examples with positive and negative pairs
   - Loads test queries and documents

2. **Model Fine-tuning (`fine_tune_model` function)**
   - Uses the "all-mpnet-base-v2" SentenceTransformer model
   - Configures training with MultipleNegativesRankingLoss
   - Fine-tunes model on training examples for specified epochs
   - Utilizes GPU if available

3. **Embedding Generation (`generate_embeddings` function)**
   - Generates embeddings for documents and test queries
   - Uses batched processing for efficiency
   - Leverages GPU acceleration when available

4. **Ranking Generation (`generate_rankings` function)**
   - Computes cosine similarity between query and document embeddings
   - Ranks documents for each test query
   - Returns top 10 most relevant documents per query

5. **Main Execution Flow**
   - Loads required data
   - Fine-tunes the model
   - Generates embeddings
   - Creates document rankings
   - Saves results to submission.csv

The system uses modern deep learning techniques to create semantic representations of text and find relevant documents based on query similarity.

## Experiments

We experimented with different SentenceTransformer models and configurations to improve the ranking system. Here are the results of our experiments:

|Exp| MAP|
|---|----|
|all-MiniLM-L12-v2| 0.23310 |
|all-mpnet-base-v2| 0.25553 |
|rerank with mixedbread-ai/mxbai-rerank-base-v1r | 0.26385 |
| **fine-tuned all-mpnet-base-v2** | **0.28786** |

We started with the `all-MiniLM-L12-v2` model and achieved a MAP score of 0.23310. We then switched to the advanced `all-mpnet-base-v2` model and improved the score to 0.25553. We then experimented with retrieval-reranking paradigms,using the `mixedbread-ai/mxbai-rerank-base-v1r` model to re-ranked the top 100 documents for each query retrieved by the `all-mpnet-base-v2` model, and boosted the score to 0.26385. 

Finally, we fine-tuned the `all-mpnet-base-v2` model on the `[anchor, positive, negative]` training examples with `MultipleNegativesRankingLoss` and achieved the best score of **0.28786**.

## Results

Our best way:
we fine tuned the sentence transformer model all-mpnet-base-v2 on the training data and then used the model to generate embeddings for the test data. It runs on 8 A6000 GPUs in 40 minutes.

Our best score: 0.28786
![Result](result.png)
