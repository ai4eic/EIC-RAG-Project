

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
    LatexTextSplitter,
)
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
import argparse, os, arxiv

os.environ["OPENAI_API_KEY"] = "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"

embeddings = OpenAIEmbeddings()

from langchain.vectorstores import LanceDB
from lancedb.pydantic import Vector, LanceModel
from Typing import List
from datetime import datetime
import lancedb

global embedding_out_length
embedding_out_length = 1536
class Content(LanceModel):
    id: str
    arxiv_id: str
    vector: Vector(embedding_out_length)
    text: str
    uploaded_date: datetime
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    url: str

def PyPDF_to_Vector(table: LanceDB, embeddings: OpenAIEmbeddings, src_dir: str, n_threads: int = 1):
    
    pass    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Create Vector DB and perform ingestion from source files")
    argparser.add_argument('-s', '--src_dir', type=str, required=True, help = "Source directory where arxiv sources are stored")
    argparser.add_argument('-db', '--db_name', type=str, required=True, help = "Name of the LanceDB database to be created")
    argparser.add_argument('-t', '--table_name', type=str, required=False, help = "Name of the LanceDB table to be created", default = "EIC_archive")
    argparser.add_argument('-openai_key', '--openai_api_key', type=str, required=True, help = "OpenAI API key")
    argparser.add_argument('-c', '--chunking', type = str, required=False, help = "Type of Chunking PDF or LATEX", default = "PDF")
    argparser.add_argument('-n', '--nthreads', type=int, default=-1)
    
    args = argparser.parse_args()
    
    SRC_DIR = args.src_dir
    DB_NAME = args.db_name
    TABLE_NAME = args.table_name
    OPENAI_API_KEY = args.openai_api_key
    NTHREADS = args.nthreads
    
    db = lancedb.connect(DB_NAME)
    table = db.create_table(TABLE_NAME, schema=Content, mode="overwrite")
    

db = lancedb.connect()
meta_data =  {"arxiv_id": "1", "title": "EIC LLM",
                          "category" : "N/A",
                "authors": "N/A",
                "sub_categories": "N/A",
                "abstract": "N/A",
                "published": "N/A",
                "updated": "N/A",
                "doi": "N/A"
                },
table = db.create_table(
    "EIC_archive",
    data=[
        {
            "vector": embeddings.embed_query("EIC LLM"),
            "text": "EIC LLM",
            "id": "1",
            "arxiv_id" : "N/A",
            "title" : "N/A",
            "category" : "N/A",
            "published" : "N/A"
        }
   
    ],
    mode="overwrite",
)

vectorstore = LanceDB(connection = table, embedding = embeddings)

sourcedir = "PDFs"
count = 0
for source in os.listdir(sourcedir):
    if not os.path.isdir(os.path.join("PDFs", source)):
        continue
    print (f"Adding the source document {source} to the Vector DB")
    import arxiv
    client = arxiv.Client()
    search = arxiv.Search(id_list=[source])
    paper = next(arxiv.Client().results(search))
    meta_data = {"arxiv_id": paper.entry_id, 
                "title": paper.title, 
                "category" : categories[paper.primary_category],
                "published": paper.published
                }
    for file in os.listdir(os.path.join(sourcedir, source)):
        if file.endswith(".tex"):
            latex_file = os.path.join(sourcedir, source, file)
            print (source, latex_file)
            documents = TextLoader(latex_file, encoding = 'latin-1').load()
            latex_splitter = LatexTextSplitter(
                chunk_size=120, chunk_overlap=10
            )
            documents = latex_splitter.split_documents(documents)
            for doc in documents:
                for k, v in meta_data.items():
                    doc.metadata[k] = v
            vectorstore.add_documents(documents = documents)
            count+=len(documents)