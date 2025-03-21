'''1. read the arxive document from the src_dir
   2. Define the metadata for the document
   3. Use Recursive chunking
   4. Use ollama model  '''

import os, glob, arxiv, argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import Pinecone as LangPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter

def SaveFromPDF(args):
	DB_NAME = args.persistent_directory
	SRC_DIR = args.src_dir
	COLLECTION_NAME = args.table_name
	CHUNKING_TYPE = args.chunking

	## recursive chunking 
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
	## Ollama embedding model
	embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

	## not able to extract the pdf id
	# AllPDFs = [os.path.join(f, f.split("/")[-1] + ".pdf") for f in glob.glob(SRC_DIR + "/*")]
	AllPDFs = [os.path.join(f, os.path.basename(f) + ".pdf") for f in glob.glob(os.path.join(SRC_DIR, "*"))]

	#check if the Vector DB persistent directory exists or not before looping through the pdfs
	if (args.db_type.lower() == "chroma"):
		if os.path.exists(DB_NAME):
			#the code is same for load and create while loading for a path exist or not
			# Load exising database
			db = Chroma(persist_directory=DB_NAME, embedding_function=embeddings, collection_name=COLLECTION_NAME)
			print("Chroma DB loaded successfully")
		else:
		# Create a new database
			db = Chroma(persist_ditectory=DB_NAME, embedding_function=embeddings, collection_name=COLLECTION_NAME)
			print("Chroma DB created successfully")
	elif (args.db_type.lower() == "pinecone"):
		LangPinecone.from_documents(texts, embeddings, index_name=COLLECTION_NAME)

	for pdf in AllPDFs:
		arxiv_id = os.path.basename(pdf).replace(".pdf", "")
		# Check whether the arxiv_id exists in db or not
		existing_docs = db.get(where={"arxiv_id": arxiv_id}, include=["metadatas"])

		#if the doc does not exists in the DB
		if not existing_docs['ids']:
			#load and split the document into pages
			loader = PyPDFLoader(pdf)
			data = loader.load_and_split()
			search = arxiv.Search(id_list=[arxiv_id])
			paper = next(arxiv.Client().results(search))

			#define the metadata for that document
			meta_data = {"arxiv_id": os.path.basename(paper.entry_id), 
					"title": paper.title, 
					"categories" : '\n'.join([f'{i+1}. {cat}' for i, cat in enumerate(paper.categories)]),
					"primary_category": paper.primary_category,
					"published": str(paper.published),
					"authors": '\n'.join([f'{i+1}. {auth.name}' for i, auth in enumerate(paper.authors)])
					}
			print ("Title is :" + """{}""".format(meta_data["title"]))

			#loop through the splited page within the document
			for page in data:
				print ("Processing page " + str(page.metadata['page']))
				##chunking
				texts = text_splitter.create_documents([page.page_content], metadatas = [meta_data])
				## create the embeddings
				db.add_documents(texts)
				print("new data is added")
			del data, loader, page
		else:
			print("Data already exists")
			continue
    
                

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Create Vector DB and perform ingestion from source files")
    argparser.add_argument('-s', '--src_dir', type=str, required=True, help = "Source directory where arxiv sources are stored")
    argparser.add_argument('-db_type', '--db_type', type=str, required=True, help = "Type of VectorDB to be created", default="chroma")
    argparser.add_argument('-db', '--db_name', type=str, required=True, help = "Name of the database to be created")
    argparser.add_argument('-t', '--table_name', type=str, required=False, help = "Name of the table to be created", default = "EIC_archive")
    # argparser.add_argument('-openai_key', '--openai_api_key', type=str, required=True, help = "OpenAI API key")
    argparser.add_argument('-c', '--chunking', type = str, required=False, help = "Type of Chunking PDF or LATEX", default = "PDF")
    argparser.add_argument('-n', '--nthreads', type=int, default=-1)
    argparser.add_argument('-db_api_key', '--db_api_key', type=str, required=False, help = "VectorDB API key")
    argparser.add_argument('-persistent_dir', '--persistent_directory', type=str, required=False, help="Directory to store persistent ChromaDB", default="./chroma_db")
    args = argparser.parse_args()
    SaveFromPDF(args)
    
    
    
    
