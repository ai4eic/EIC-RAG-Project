1. **User Query**: The process begins with a user query. This query is typically a text string that the user wants to know more about.

2. **Query to LLM**: The user query is then passed to a Language Model (LLM), which determines if more information is needed from the Vector Database to provide a comprehensive answer.

3. **Vector Database Query**: If the LLM determines that more information is needed, the user query is converted into a vector using the same embedding model that was used to create the vectors in the database. This query vector is then used to retrieve the most similar vectors from the database. The similarity is typically measured using a distance metric in the vector space, such as cosine similarity or Euclidean distance.

4. **Information Curation**: The sentences corresponding to the most similar vectors, along with their sources, are then curated and rewritten cohesively. This curation process may involve removing duplicate information, filtering out irrelevant information, and ordering the information in a way that makes sense for the user query.

5. **Self Evaluation by LLM**: The curated information is then passed back to the LLM, which self-evaluates the retrieved information. If the LLM determines that more information is needed, it provides keywords for a new search.

6. **Second Vector Database Query**: If more information is needed, the keywords provided by the LLM are used to retrieve additional similar sentences from the vector database.

7. **Return Information**: Finally, all retrieved and curated information is returned back to the control, providing a comprehensive answer to the user query.

This process allows for efficient and accurate information retrieval from a vector database, enabling users to get comprehensive answers to their queries even when the database contains a large amount of high-dimensional vector data.

Below is flow diagram explaining this process in detail