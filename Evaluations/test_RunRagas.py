
import toml, os, sys
import time
import torch  # Added for GPU availability check

# Check if GPU is available
if not torch.cuda.is_available():
    print("Warning: GPU not available, falling back to CPU")
else:
    print(f"GPU available: {torch.cuda.get_device_name(0)}")

# -------------------------------
sys.path.append(os.path.realpath("../"))

# with open("../../.streamlit/secrets.toml") as f:
#     secrets = toml.load(f)
    
# os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
# if secrets.get("LANGCHAIN_API_KEY"):
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
print(langchain_api_key)
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = secrets["LANGCHAIN_ENDPOINT"]
print(f"langchain_api_key :" , {langchain_api_key})

pinecone_api_key = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY") #secrets["PINECONE_API_KEY"]

#--------------------------
import nest_asyncio

nest_asyncio.apply()

# Get the dataset
from langsmith import Client
from langsmith.utils import LangSmithError

client = Client()

#-----------------------
import pandas as pd
df = pd.read_csv("AI4EIC_sample_dataset.csv", sep = ",")

#---------------------
# from langchain_openai import OpenAIEmbeddings
# from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from streamlit_app.app_utilities import *
from streamlit_app.LangChainUtils.LLMChains import *
from langchain import callbacks
from langsmith import Client
from langchain_core.tracers.context import tracing_v2_enabled
from langchain.callbacks.tracers import LangChainTracer
from ragas.run_config import RunConfig
import json


# def RunQuery(input_question, max_k, sim_score):
def RunQuery(input_question, max_k, sim_score,
             collection_name=None, db_name=None, table_name=None):
    
    #for generating output form prompt
    llm = ChatOllama(model="llama3.2:latest", temperature=0, num_predict=4096)

    ## Configure LLM with GPU-optimized parameters
    # llm = ChatOllama(
    #     model="llama3.2:latest",
    #     temperature=0,
    #     num_predict=4096,
    #     num_gpu=999,  # Use maximum available GPU layers
    #     num_threads=8  # Adjust based on your GPU/CPU setup
    # )

    # embeddings = OpenAIEmbeddings()
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    
    # Defining some props of DB
    SimilarityDict = {"Cosine similarity" : "similarity", "MMR" : "mmr"}

    #create an instance of the DB  
    DBProp = {"PINECONE" : {
                        "vector_config" : {"db_api_key" :pinecone_api_key, "index_name" : "llm-project", "embedding_function" : embeddings},
                            "search_config" : {"metric" : sim_score, "search_kwargs" : {"k" : max_k}},
                            "available_metrics" : ["Cosine similarity", "MMR"]
                },
                "CHROMA": {
                        "vector_config": {"db_name": db_name, "embedding_function": embeddings, "collection_name": collection_name},
                        "search_config": {"metric": sim_score, "search_kwargs": {"k": max_k}},
                        "available_metrics": ["Cosine similarity", "MMR"],
                },
                "LANCE": {
                    "vector_config": {"db_name": db_name, "table_name": table_name},
                    "search_config": {"metric": sim_score, "search_kwargs": {"k": max_k}},
                    "available_metrics": ["Cosine similarity", "MMR"],
                },
            }
    
    #create an instance of the VectorDB
    retriever = GetRetriever("CHROMA", DBProp["CHROMA"]["vector_config"], DBProp["CHROMA"]["search_config"])
    print("output of Getretriever")

    # project name for tracing in Langsmith
    project_name = f"RAG-CHAT-tapasi"

    # Create a LangChain tracer for tracing the run
    tracer = LangChainTracer(project_name = project_name)
    print("out of LangChainTracer")

    run_name = "Evaluation-testings"
    trace_metadata = {"DBType": "CHROMA", 
                    "similarity_score": sim_score, 
                    "max_k": max_k
                    }
    RUNCHAIN = RunChatBot(llm, retriever, "../Templates"
                        ).with_config({"callbacks": [tracer], 
                                        "run_name": run_name,
                                        "metadata": trace_metadata}
                                        )
    print("out of RunCHatBot")
    trace_id = ""
    response = ""
    runid = ""
    with tracing_v2_enabled(project_name) as cb:
        with callbacks.collect_runs() as ccb:
            output = RUNCHAIN.invoke(input_question)

            ## modify to ensure json format
            # response = output["answer"]
            # Ensure output is a JSON-compatible dictionary
            if isinstance(output, dict) and "answer" in output:
                response = json.dumps({"answer": output["answer"]})
            else:
                response = json.dumps({"answer": str(output)})

            print (output)
            print (len(ccb.traced_runs))
            for run in ccb.traced_runs:
                runid = run.id
                print (run.name)
                print (run.id)
                print (run.inputs)
                print (run.outputs)
                print (run.trace_id)
                trace_id = run.trace_id
    return response, trace_id, client.share_run(runid)

def RunLLM(input_question, MODEL = "llama3.2:latest"):
    # model_name = f"gpt-3.5-turbo-1106" if GPTMODEL == 3 else "gpt-4-0125-preview"
    print (f"input_question, {input_question}")
    # llm = ChatOpenAI(model_name=model_name, temperature=0,
    #                 max_tokens = 4096
    #                 )
    llm = ChatOllama(model_name=MODEL, temperature=0, num_predict=4096)
    # # For GPU
    # llm = ChatOllama(
    #     model_name=MODEL,
    #     temperature=0,
    #     num_predict=4096,
    #     num_gpu=999  # Maximize GPU usage
    # )
    output = llm.invoke(input_question).content
    ## for json format
    # return output

    print(f"output of llm : {output}")


#---------------------------------------------
import pickle
from datasets import Dataset

from langchain_ollama import OllamaEmbeddings, ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
# from langchain_openai import OpenAIEmbeddings
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)

import ragas
'''RAGAS metrics uses openai models by default. So we explicitly define llm and embedding model of ollama and use to
compute evaluation metrics'''

# required for ollama 
ollama_embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
ollama_llm = ChatOllama(model="llama3.2:latest", temperature=0)

# wrap the above defined llm and embedding model
wrapped_llm = LangchainLLMWrapper(ollama_llm)
wrapped_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)

ANSWER_CORRECTNESS = ragas.metrics.AnswerCorrectness(name = "ANSWER_CORRECTNESS",
                                                     weights = [0.90, 0.10],
                                                     llm = wrapped_llm,
                                                     embeddings = wrapped_embeddings
                                                     )
ANSWER_RELEVANCY = ragas.metrics.AnswerRelevancy(name = "ANSWER_RELEVANCY",
                                                strictness = 5,
                                                llm = wrapped_llm,
                                                embeddings = wrapped_embeddings
                                                )
CONTEXT_ENTITY_RECALL = ragas.metrics.ContextEntityRecall(name = "CONTEXT_ENTITY_RECALL",
                                                          llm = wrapped_llm
                                                         )
CONTEXT_PRECISION = ragas.metrics.ContextPrecision(name = "CONTEXT_PRECISION",
                                                   llm = wrapped_llm
                                                    )
CONTEXT_RECALL = ragas.metrics.ContextRecall(name = "CONTEXT_RECALL",
                                             llm = wrapped_llm
                                            )
## new RAGAS doesn't have this evaluation metric                                             
# CONTEXT_RELEVANCY = ragas.metrics.ContextRelevancy(name = "CONTEXT_RELEVANCY")
                                                   
FAITHFULNESS = ragas.metrics.Faithfulness(name = "FAITHFULNESS",
                                          llm = wrapped_llm)

import pandas as pd
df = pd.read_csv("AI4EIC_sample_dataset.csv", sep = ",")

from ragas import evaluate
dataset = {"question": [],
            "answer": [], 
            "contexts": [],
            "ground_truth": [],
            "arxiv_id": [],
            "input_arxiv_id": [], 
            "trace_links": []
            }

# no of chunks to be retrieved
max_k = 20
sim_score = "mmr"
db_name="../ingestion/myChromaDB"
collection_name = "EIC_archive"
table_name = "arxiv_table"

if (os.path.exists(f"results_k_{max_k}_sim_{sim_score}.csv")):
    os.system(f"rm -f results_k_{max_k}_sim_{sim_score}.csv")

for index, row in df.iterrows():
    question = row["input_question"]
    answer, trace_id, trace_link = RunQuery(question, max_k, sim_score, db_name=db_name, collection_name=collection_name)
    print(f" RunQuery : answer : {answer}, trace_id : {trace_id}, trace_link : {trace_link}")

    project_name = f"RAG-CHAT-tapasi"
    run_name = "Evaluation-testings"

    # if verbose==1:
    #     print(f"before langsmith is called")

    runs = client.list_runs(project_name = project_name, trace_id = trace_id)
    print(f"after langsmith client is called : , {runs}")
    contexts = []
    cite_arxiv_ids = []
    for run in runs:
        if (run.run_type.lower() == "retriever"):
            print (run.name)
            print (run.id)
            print (run.inputs)
            print (run.outputs)
            for i in run.outputs['documents']:
                contexts.append(i["page_content"])
                cite_arxiv_ids.append(i["metadata"]["arxiv_id"].split("/")[-1].strip())
            print (run.trace_id)
            print ("-----")

    # Parse JSON answer and clean it
    try:
        answer_dict = json.loads(answer)
        cleaned_answer = answer_dict["answer"].strip()
    except json.JSONDecodeError:
        cleaned_answer = answer.strip()

    dataset["question"].append(question)
    print (answer.split("http://")[0].strip("\n"))
    #adjustment for json
    # dataset["answer"].append(answer.split("http://")[0].strip("\n"))
    dataset["answer"].append(cleaned_answer)

    dataset["contexts"].append(contexts)
    dataset["ground_truth"].append(row["output_complete_response"])
    dataset["input_arxiv_id"].append(row["input_arxiv_id"])
    dataset["arxiv_id"].append(cite_arxiv_ids)
    dataset["trace_links"].append(trace_link)
    
    with open(f"dataset_k_{max_k}_sim_{sim_score}.pkl", "wb") as f:
        pickle.dump(dataset, f)
    
    tmpdataset = {}
    for key, value in dataset.items():
        tmpdataset[key] = [value[-1]]
    DATASET = Dataset.from_dict(tmpdataset)

    # Start time
    start_time = time.time()
    print(start_time)

    # Configure run_config with custom timeout
    
    run_config = RunConfig(
        timeout=600,           # Set timeout to 10 minutes (600 seconds)
        max_workers=1     # Sequential processing to avoid Ollama overload
    )

    result = evaluate(DATASET,
                  metrics = [
                    #   FAITHFULNESS,
                    # #   CONTEXT_RELEVANCY,
                      CONTEXT_ENTITY_RECALL
                    #   CONTEXT_PRECISION,
                    #   CONTEXT_RECALL,
                    #   ANSWER_RELEVANCY,
                    #   ANSWER_CORRECTNESS
                  ],
                  run_config = run_config
                  )
    result_df = result.to_pandas()
    if (os.path.exists(f"results_k_{max_k}_sim_{sim_score}.csv")):
        df = pd.read_csv(f"results_k_{max_k}_sim_{sim_score}.csv", sep = ",")
        result_df = pd.concat([df, result_df])
    result_df.to_csv(f"results_k_{max_k}_sim_{sim_score}.csv", index = False)

    # End time
    end_time = time.time()
    delta_time = end_time - start_time
    print(f"time taken : {delta_time}")


# -------------------------------------------------------

    # import asyncio
    # from ragas import evaluate

    # async def run_evaluation(dataset, metrics):
    #     return evaluate(dataset, metrics=metrics)

    # try:
    #     result = asyncio.run(asyncio.wait_for(
    #         run_evaluation(
    #             DATASET,
    #             [
    #                 FAITHFULNESS,
    #                 CONTEXT_ENTITY_RECALL,
    #                 CONTEXT_PRECISION,
    #                 CONTEXT_RECALL,
    #                 ANSWER_RELEVANCY,
    #                 ANSWER_CORRECTNESS
    #             ]
    #         ),
    #         timeout=600  # 300 seconds
    #     ))
    #     result_df = result.to_pandas()
    #     if os.path.exists(f"results_k_{max_k}_sim_{sim_score}.csv"):
    #         df = pd.read_csv(f"results_k_{max_k}_sim_{sim_score}.csv", sep=",")
    #         result_df = pd.concat([df, result_df])
    #     result_df.to_csv(f"results_k_{max_k}_sim_{sim_score}.csv", index=False)
    # except asyncio.TimeoutError:
    #     print(f"Evaluation timed out after 300 seconds")
    #     continue