This process can be broken down further into three main steps: chunking of information, embedding models, and storing it in a vector database.
1. Chunking
2. Encoding chunked information into a vector using a embedding model (e.g. BERT, seq2seq, text2vec)
3. Storing the encoded information in a vector database. 

### Chunking

This is the first step in the ingestion process. The raw data can come in various forms.  which could be a large corpus of text, is divided into manageable chunks or segments. The size of these chunks can vary depending on the specific requirements of the task at hand. Chunking helps in reducing the complexity of the data and makes it easier for the model to process the information.

### Embedding models

Embedding models are a type of machine learning model used to convert discrete variables, such as words or items, into continuous vectors, often in a lower-dimensional space. The goal of an embedding model is to place similar items closer together in the embedding space, while dissimilar items are placed further apart.

For the current project, OpenAI's `text-embedding-ada-002` model is used since it is a powerful language model used to convert discrete text data into continuous vector representations, often referred to as embeddings. These embeddings capture the semantic meaning of the text data in a high-dimensional space, where similar items are placed closer together and dissimilar items are placed further apart. The `text-embedding-ada-002 model` is trained using a variant of the Transformer architecture, which is a type of deep learning model. The Transformer model uses self-attention mechanisms to weigh the importance of different words in a text when generating the embedding for a particular word.

The objective function for the Transformer model can be written as:


$$
J(\theta) = - \frac{1}{N} \sum_{n=1}^{N} \log p(y_n | x_n; \theta)
$$

where $x_n$ is the input text, $y_n$ is the target text (which is often the same as the input text in unsupervised learning), N is the total number of texts, and theta are the parameters of the model.

After training, each piece of text is represented by a vector, which can be used as a feature in other machine learning models or for tasks such as finding similar texts or clustering.

#### Word2Vec Model

One of the most common examples of an embedding model is Word2Vec, which is used to create word embeddings. Word2Vec uses a shallow neural network to learn the relationships between words based on their context.

The Word2Vec model can be trained using one of two architectures: Continuous Bag of Words (CBOW) or Skip-gram. In the CBOW architecture, the model predicts a target word from its context words, while in the Skip-gram architecture, the model predicts context words from a target word.

The objective function for the CBOW model can be written as:

$$
J(\theta) = \frac{1}{T} \sum_{t=1}^{T} \log p(w_t | w_{t-j}, ..., w_{t+j}; \theta)
$$

where $w_t$ is the target word, $w_{t-j}$, ..., $w_{t+j}$ are the context words, $T$ is the total number of words, and theta are the parameters of the model.

The objective function for the Skip-gram model can be written as:

$$
J(\theta) = \frac{1}{T} \sum_{t=1}^{T} \sum_{-j \leq k \leq j, k \neq 0} \log p(w_{t+k} | w_t; \theta)
$$

### Vector Database

Vector databases, also known as vector search engines or similarity search engines, are databases designed to handle high-dimensional vector data efficiently. They are particularly useful in machine learning and AI applications, where data is often represented as high-dimensional vectors, such as embeddings.

Traditional databases are designed to handle structured data like text, numbers, and dates, and they use exact matching or numerical comparisons for queries. However, these methods are not efficient for high-dimensional vector data, where the concept of similarity is more complex and is often defined by a distance metric in the vector space, such as Euclidean distance or cosine similarity.

Vector databases address this problem by using indexing techniques that are designed for high-dimensional vector data. These techniques, such as k-d trees, ball trees, and HNSW (Hierarchical Navigable Small World), allow the database to quickly narrow down the search space and find the most similar vectors without having to compare the query vector to every vector in the database.

The objective function for a vector database query can be written as:

$$
\text{argmin}_{v \in V} ~ d(q, v)
$$

where `q` is the query vector, `V` is the set of vectors in the database, `d` is a distance metric, and argmin returns the vector `v` that minimizes the distance to the query vector.

By using these indexing techniques, vector databases can handle queries on large-scale vector data much more efficiently than traditional databases, making them a key component in many machine learning and AI systems.

The figure below shows the process of ingestion in detail. 