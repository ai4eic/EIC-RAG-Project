# Retrieval Augmented Generation for EIC

This is a project that is currently being developed to build a RAG based system for the upcoming EIC. 

There are three main parts to the RAG pipeline. 

## Ingestion 

Ingestion in Retrieval-Augmented Generation (RAG) is a crucial process that involves the preparation and organization of data to be used by the model. This process can be broken down further into three main steps: chunking of information, embedding models, and storing it in a vector database.
1. Chunking
2. Encoding chunked information into a vector using a embedding model (e.g. BERT, seq2seq, text2vec)
3. Storing the encoded information in a vector database.

### Chunking

This is the first step in the ingestion process. The raw data can come in various forms.  which could be a large corpus of text, is divided into manageable chunks or segments. The size of these chunks can vary depending on the specific requirements of the task at hand. Chunking helps in reducing the complexity of the data and makes it easier for the model to process the information.

## Retrieval
## Content Fusion and Generation

# Types of RAG system

A very recent survey paper. summarizes the types of RAG system[^1]. There are three types of RAG architecture broadly based on where LLM being used in the pipeline

## Project Milestones

1. Building a Naive RAG for EIC using the 200 papers from arxiv on EIC. ✅
    * Backend is a relatively straight forward RAG architecture. Where ingestion of data is done using PyPDF.
    * Frontend is a simple web interface that allows for the user to upload a PDF and get back a list of papers that are relevant to the input.
    * Report evaulated RAGAS metrics for the built architecture. 
    * Publish this in the proceeding for AI4EIC-2023. 🧑‍🏭
2. Going beyong Naive RAG. Towards building a RAG architecture with Testable Evaulation Metrics. 🧑‍🏭    
    * This requires going beyond 
3. Multi modal output as a Proof of concept.
    * Storing meta data information about table etc.
    * Using Agents in Langchain to __build__ a latex report. 
    * 

## References

[^1]: [Types of RAG](https://export.arxiv.org/pdf/2312.10997)
[^2]: [RAGAS](https://arxiv.org/pdf/2203.03416.pdf)
[^3]: [LangChain](https://python.langchain.com/docs/get_started/introduction)

# How tos

## Running the webapp

In order to run the `streamlit` app do the following 

1. `git clone https://github.com/wmdataphys/EIC-RAG-Project.git` & `cd EIC-RAG-Project`
2. It is better to have a seperate python environment incase of any version mismanagement with other projects. I use conda env `conda create --name env_RAG-EIC python=3.10` This creates a python version `3.10` as this was stable when I started building the app. Once created activate the env as `conda activate env_RAG-EIC`
3. Now install `pip` before installing all other packages. `conda install pip`
4. Now install all the requirements `pip install -r requirements.txt`
5. Ask `karthik18495@gmail.com` about the `secrets.toml` and `config.toml`
6. Create a folder named `.streamlit` in the parent directory and move the files `secrets.toml` and `config.toml` in there. 
7. Now run `streamlit run streamlit_app/AI4EIC-RAGAS4EIC.py`. This should run on a `http://localhost:8050` 

## Updating `requirements.txt`

1. If any new library has been used in the app that requires installation through pip. Make sure to use the `--format freeze` when updating the `requirements.txt`
2. The command is `pip list --format freeze > requirements.txt`