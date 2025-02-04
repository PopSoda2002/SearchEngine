# Mini Project 1 Part 2 Report

**Contribution**:

@Huapeng Zhou:

@Junfeng Zhou:

## Ranking Document Report

### Comparison of Encoding Methods

> Q: Compare GloVe embeddings vs. Sentence Transformer embeddings.
> A: 
> - The GloVe embeddings are pre-trained on a general corpus for word embeddings, which are more suitable for word similarity tasks.
> - The Sentence Transformer embeddings are trained for sentence embeddings, which are better to capture the semantic meaning of sentences.

> Q: Which method ranked documents better?
> A: The Mean Average Precision (MAP) of the Sentence Transformer embeddings is 0.4586, which is higher than the MAP of the GloVe embeddings, 0.0509.


> Q: Did the top-ranked documents make sense?
> Yes, the 1st ranked document to the query "Breast Cancer Cells Feed on Cholesterol" is, "While many factors are involved in the etiology of cancer, it has been clearly established that diet significantly impacts one’s risk for this disease. More recently, specific food components have been identified which are uniquely beneficial in mitigating the risk of specific cancer subtypes. Plant sterols are well known for their effects on blood cholesterol levels, however research into their potential role in mitigating cancer risk remains in its infancy. As outlined in this review, the cholesterol modulating actions of plant sterols may overlap with their anti-cancer actions. Breast cancer is the most common malignancy affecting women and there remains a need for effective adjuvant therapies for this disease, for which plant sterols may play a distinctive role."
> The document is relevant to the query and provides information about the relationship between cholesterol and breast cancer.

> Q: How does cosine similarity behave with different embeddings?
> A: The cosine similarity is a measure of similarity between two non-zero vectors of an inner product space. The Sentence Transformer embeddings have a higher cosine similarity with the query than the GloVe embeddings.

### Observations on Cosine Similarity & Ranking

> Q: Did the ranking appear meaningful?
> A: The ranking of the documents is meaningful. The top-ranked documents are relevant to the query.
> For example, the 1st ranked document to the query "Breast Cancer Cells Feed on Cholesterol" is relevant to the query, which is about the relationship between cholesterol and breast cancer.

> Q: Were there cases where documents that should be highly ranked were not?
> A: No, the top-ranked documents are relevant to the query.

>Q: What are possible explanations for incorrect rankings?
> A: Although the top10 documents are relevant to the query, the possible explanations for incorrect rankings are:
> - Sentence Transformer embeddings are not fine-tuned on the dataset.
> - The cosine similarity may be not the best distance metric for ranking documents.
> - The documents are not preprocessed correctly (e.g., removing stopwords).

### Possible Improvements

> Q: What can be done to improve document ranking?
> A: 1. Fine-tune the Sentence Transformer embeddings on the dataset. 2. Preprocess the documents (e.g., removing stopwords) before ranking.

> Q: Would a different distance metric (e.g., Euclidean, Manhattan) help?
> A: Yes, a different distance metric may help improve ranking. For example, the Euclidean distance metric may be better for ranking documents.

> Q: Would preprocessing the queries or documents (e.g., removing stopwords) improve ranking?
> A: Yes, preprocessing the queries or documents may improve ranking. For example, removing stopwords can reduce noise in the documents and help the model focus on the important words.

## Fine-Tuning Report