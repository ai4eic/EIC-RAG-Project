from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

import streamlit as st
import numpy as np
import arxiv, os, random
import pandas as pd

from app_utilities import num_tokens_from_prompt, SetHeader
from langchain import callbacks
from langsmith import Client

from LangChainUtils.LLMChains import RunQuestionGeneration

Summary = """
        # IDEA OF THIS SCRIPT
            This script is to help the developer to generate questions for the AI4EIC-RAG project.
            Check if the user is logged in. If not ask them to login, 
            If they are logged in make sure if they have the right to access the generate questions. 
            This is done by checking their level using their log in information
        ## GENERATING A RANDOM ARTICLE 
            Ask the user if they want to load a random article from database. Then Select which version of GPT do they want to select.
            Else, user can select the targetted article as well. 
            Once selected and clicked load. If random artucle is selected, then generate a random number generator using random.choices (__Thanks to @Neeltje for this elegent solution__)
        ## GENERATING QUESTIONS
        """
if st.secrets.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
    
if "LANGCHAIN_PROJECT" not in os.environ:
    os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_EVAL_PROJECT"]
if "LANGCHAIN_RUN_NAME" not in os.environ:
    os.environ["LANGCHAIN_RUN_NAME"] = "QA_Generation"
articles = pd.read_csv(st.secrets.SOURCES_DETAILS, sep = ",")

def compute_lim(GPT_CONTEXT_LEN:int = 12_000, CHAR_PER_TOKEN:int = 4):
    return GPT_CONTEXT_LEN * CHAR_PER_TOKEN


GPTDict = {"4"  : {"model" : "gpt-4-0125-preview", "context_lim": 128_000, "temperature": 0, "max_tokens": 4096, "WORD_LIM": compute_lim(120_000, 4)},
           "3.5": {"model": "gpt-3.5-turbo-1106",  "context_lim":  16_385, "temperature": 0, "max_tokens": 4096, "WORD_LIM": compute_lim( 10_000, 4)}, 
           }

client = Client()
SetHeader("RAG Generate Questions")

# Some explanations to do 

st.header("Using LLM to generate QA bencmarks dataset 🧞")
with st.expander("Expand to see detailed explanation"):
    st.markdown(open("streamlit_app/Resources/Markdowns/QA_Generation.md", "r").read())

if not st.session_state.get("user_name"):
    st.error("Please login to your account first to further continue and generate questions.")
    st.stop()
# Start by defining article load as false
article_keys = ["article_loaded", "article_primary_category", 
                "article_categories", "article_title", 
                "article_abstract", "article_url", "article_doi", 
                "article_authors", "article_content", "article_id",
                "article_date", "article_num_tokens", "article_pages",
                "full_content"
                ]


def LoadRandomArticle(df: pd.DataFrame):
    """_summary_
    This is a adhoc function. Can be better.

    Args:
        df (pd.DataFrame): The source details csv dataframe file.

    Returns:
        str: Choice of arxiv_id selected
    """
    return random.choices(df["arxiv_id"].values, weights = df["used_num_times"].values)[0]
def LoadArticle(LoadRandom: bool, article_keys: list):
    for key in article_keys:
        st.session_state[key] = None
    if st.session_state.get("load_random_article"):
        st.session_state.article_id = LoadRandomArticle(df = articles)
    elif st.session_state.get("selected_article"):
        #print (f"Going to query the article {st.session_state.selected_article}from the database")
        st.session_state.article_id = articles.query(f'title=="{st.session_state.selected_article}"')["arxiv_id"].values[0]
        #print (st.session_state.article_id)
    elif not LoadRandom:
        pass
    else:
        st.error("Please select an article to load")
        st.stop()
    st.session_state.questions = []
    st.session_state.load_article = True
    st.session_state.generation_count = 0

# mode = 0 for user, 1 for annotator, 2 for developer   
if st.session_state.get("user_mode", -1) > 0:
    with st.sidebar:
        st.toggle("Contribute to Evaluation", 
                  value = False, 
                  key = "contribute_to_eval", 
                  help = "Toggle this to contribute to evaluation for each response"
                  )
        if (st.session_state.contribute_to_eval):
            with st.form("Langsmith details"):
                datagen_run = st.text_input("RUN NAME", value = os.environ["LANGCHAIN_RUN_NAME"], placeholder =  st.session_state.get("dataset_name", None))
                dataset_name = st.text_input("DATASET NAME", 
                                             value = st.session_state.get("dataset_name", st.session_state.get("user_name").upper() + "_DATASETS"), 
                                             placeholder =  st.session_state.get("dataset_name", st.session_state.get("user_name").upper() + "_DATASETS")
                                             )
                run_eval_name = st.text_input("EVAL PROJECT NAME", 
                                              value = os.environ.get("LANGCHAIN_PROJECT", "QA-BENCHMARK")
                                              )
                submit_dataset = st.form_submit_button("Submit", help = "Submit the dataset name and run name to start generating questions")
                if (submit_dataset):
                    os.environ["LANGCHAIN_PROJECT"] = run_eval_name
                    os.environ["LANGCHAIN_RUN_NAME"] = datagen_run
                    st.session_state.dataset_name = dataset_name
                    LoadArticle(False, article_keys + ["dataset_id"])
                    datasets = client.list_datasets(dataset_name_contains = dataset_name)
                    for dataset in datasets:
                        if dataset.name == dataset_name:
                            st.session_state.dataset_id = dataset.id
                            #print (f"Dataset id is {st.session_state.dataset_id}")
                            break
                    if not st.session_state.dataset_id:
                        dataset = client.create_dataset(dataset_name, description = f"Dataset for QA Benchmarks generated by {st.session_state.user_name}")
                        st.session_state.dataset_id = dataset.id
                    

if "load_article" not in st.session_state:
    st.session_state.load_article = False

st.header("Select GPT Version and load an Article from arxiv database to generate questions", divider = "rainbow")
col_ll2, col_bb1, col_bb2, col_rr2 = st.columns([1, 2, 2, 1])
with col_bb1:
    st.checkbox("Select a Random Article if needed", 
                key = "load_random_article", 
                help = "This generates a random article based on previous usage, it loads an article that has not been used much"
                )
with col_bb2:
    st.selectbox("Select GPT Version", GPTDict.keys(), 
                 key = "gpt_version", 
                 help = "Select the GPT version to generate questions",
                 index = 1
                 )

if not st.session_state.get("load_random_article"):
    col_aa1, col_aa2 = st.columns([1, 4])
    with col_aa1:
        cat = st.selectbox("ARXIV primary category", st.session_state.get("arxiv_id", articles["primary_category"].unique().tolist()), 
                        key = "primary_category",
                        help = "Select the primary category of the article to load",
                        index = None
                        )
    with col_aa2:
        st.selectbox("ARXIV title", sorted(articles[(articles["primary_category"] == cat)]["title"].to_list(), key = lambda x: x.lower()),
                    key = "selected_article",
                    format_func = lambda x: f"""{x}""",
                    help = "Select the title of the article to load",
                    index = None
                    )

col_Al, col_A, colAr = st.columns([1, 4, 1])
with col_A:
    st.button("Load Article from arxiv....", 
              on_click = LoadArticle , 
              args = [True, article_keys], 
              help = "Load the article from arxiv database", 
              disabled = not (st.session_state.get("load_random_article") or st.session_state.get("selected_article"))
              )
GPTDICT = GPTDict[st.session_state.get("gpt_version", "3.5")]    
WORD_LIM = GPTDICT["WORD_LIM"]

st.header("", divider = "rainbow")
if st.session_state.load_article:
    with st.status("Loading article...", expanded = True, state = "running") as status:
        if (st.session_state.get("load_random_article")):
            st.write("Selecting a random article....")
        article = st.session_state.get("article_id")
        st.write(f"Searching {article} ID from arxiv.org...")
        try:
            search = arxiv.Search(id_list=[article])
            paper = next(arxiv.Client().results(search))
        except:
            status.update(label = "Unable to load article. Please try again.", state = "error", expanded = True)
            st.stop()
        st.session_state["article_id"] = article
        st.session_state["article_title"] = paper.title
        st.session_state["article_abstract"] = paper.summary
        st.session_state["article_url"] = paper.pdf_url
        st.session_state["article_doi"] = paper.doi
        st.session_state["article_authors"] = '\n'.join([f'{i+1}. {auth.name}' for i, auth in enumerate(paper.authors)])
        st.session_state["article_date"] = paper.published
        st.session_state["article_primary_category"] = paper.primary_category
        st.session_state["article_categories"] = ', '.join([f'{cat}' for i, cat in enumerate(paper.categories)])
        st.write(f"Loading article {article} in memory...")
        docs = PyPDFLoader(paper.pdf_url).load()
        st.session_state.article_pages = len(set([doc.metadata.get('page') for doc in docs]))
        full_content = "\n".join([doc.page_content for doc in docs])
        if (len(full_content) > WORD_LIM):
            st.warning("Too large article to load in memory. Will Chunk it when running the QA Generation.")
        st.session_state["full_content"] = full_content
        st.write(f"Article {article} successfully loaded in memory. Precounting tokens now...")
        num_tokens = num_tokens_from_prompt(full_content, "gpt-3.5-turbo-1106")
        st.session_state["article_num_tokens"] = num_tokens
        st.session_state.article_loaded = True
        st.session_state.load_article = False
        status.update(label = f"Article {article} Successfully loaded and ready, Tokens = {num_tokens}!", state = "complete", expanded = False)

if not st.session_state.get("article_id"):
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Article ID", divider = "rainbow")
    st.write(st.session_state.get("article_id", ""))
    st.subheader("Paper Summary", divider = "rainbow")
    st.write(st.session_state.get("article_abstract", ""))
    st.subheader("Publication Date", divider = "rainbow")
    st.write(st.session_state.get("article_date", ""))
    st.subheader("Primary Category", divider = "rainbow")
    st.write(st.session_state.get("article_primary_category", ""))
    
with col2:
    st.subheader("Paper Title", divider = "rainbow")
    st.write(st.session_state.get("article_title", ""))
    st.subheader("Paper Authors", divider = "rainbow")
    st.write(st.session_state.get("article_authors", ""))
    st.subheader("Link to PDF", divider = "rainbow")
    st.write(st.session_state.get("article_url", ""))
    st.subheader("Published journal (if any)", divider = "rainbow")
    st.write(st.session_state.get("article_journal", "Not Published Yet"))
    st.subheader("Categories", divider = "rainbow")
    st.write(st.session_state.get("article_categories", ""))
    st.subheader("Num of Pages and tokens", divider = "rainbow")
    st.markdown(r"__Pages__: " + str(st.session_state.get("article_pages", "")) + r"  &  __Tokens__: " + str(st.session_state.get("article_num_tokens", "")))

if st.session_state.get("article_loaded") and len(st.session_state.get("full_content", [-1])) < WORD_LIM:
    article_container = st.container(border = True)
    article_container.subheader("Article Content", divider = "rainbow")
    with st.expander("Expand to show", expanded = False):
        st.write(st.session_state.get("full_content", ""))


prefix = open("Templates/QA_Generations/example_01.template").read()

def gen_submit(generate: bool):
    st.session_state.response_container = st.empty()
    st.session_state["Generate"] = generate
    st.session_state.generation_count += 1

for ques in st.session_state.get("questions", []):
    with st.expander("Question - " + ques["qnum"], expanded = False):
        st.header("", divider = "rainbow")
        st.subheader(ques["question"])
        st.subheader("Answer")
        st.write(ques["answer"])
        if (len(ques["content"]) > WORD_LIM):
            st.subheader("Content")
            st.write(ques["content"])
        st.subheader("""Link to trace [🛠️](""" + ques["trace_link"] + ")")
        st.header("", divider = "rainbow")
        

def add_to_dataset():
    INPUTS = {"NCLAIMS": int(st.session_state.get("dataset_nclaims")),
              "ARXIV_ID": st.session_state.get("dataset_arxiv_id"),
              "CATEGORY": st.session_state.get("article_primary_category"),
              "QUESTION": st.session_state.get("dataset_questions"),
              "RUN_TRACE": st.session_state.share_run_url
              }
    OUTPUTS = {"NCLAIMS": int(st.session_state.get("dataset_nclaims")),
               "CLAIMS" : eval(st.session_state.get("dataset_claims")),
               "COMPLETE_RESPONSE": st.session_state.dataset_complete_response,
               "INDIVIDUAL_RESPONSE": eval(st.session_state.get("dataset_individual_response"))
               }
    metadata = {"username": st.session_state.user_name,
                "linked_run": st.session_state.share_run_url,
                "private_link": st.session_state.run_url,
                "n_claims": int(st.session_state.get("dataset_nclaims")),
                "arxiv_id": st.session_state.get("dataset_arxiv_id")
                }
    #print (INPUTS, OUTPUTS)
    client.create_example(
        inputs = INPUTS,
        outputs = OUTPUTS,
        dataset_id = st.session_state.get("dataset_id"),
        metadata = metadata
        )  
    # SINCE ADDED NOW REDUCE THE COUNT 
    #df = pd.read_csv(st.secrets.SOURCES_DETAILS, sep = ",")
    #df[df["arxiv_id"] == st.session_state.get("dataset_arxiv_id")]["used_num_times"] -= 1
    #df["used_num_times"].mask(df['used_num_times'] < 0, 0, inplace = True)
    #df.to_csv(st.secrets.SOURCES_DETAILS, sep = ",", index = False)

with st.container(border = True):
    st.title("Lets Start Generating Questions")
    with st.form("Generate Question"):
        f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
        with f_col1:
            n_questions = st.number_input("Number of questions to be generated", min_value = 1, max_value = 5)
        with f_col2:
            n_claims = st.number_input("Number of claims to be generated in each question", min_value = 1, max_value = 10)
            st.form_submit_button("Generate", on_click = gen_submit, args = (True,))
        with f_col3:
            st.markdown("Selected GPT Version: " + st.session_state.get("gpt_version"))
             
    if st.session_state.get("Generate"):
        st.session_state["Generate"] = False
        GPTVersion = st.session_state.get("gpt_version")
        llm = ChatOpenAI(model_name=GPTDict[GPTVersion]["model"], 
                         temperature=GPTDict[GPTVersion]["temperature"], 
                         max_tokens=GPTDict[GPTVersion]["max_tokens"],
                         )
        chain = RunQuestionGeneration(llm).with_config({"run_name" : os.environ["LANGCHAIN_RUN_NAME"]
                                }
                               )
        #print ("WORD LIMIT: ", WORD_LIM)
        if (len(st.session_state["full_content"]) > WORD_LIM):
            with st.status(label = f"Article {st.session_state.article_id} is too large to load in memory.", state = "running", expanded = True) as status:
                status.update(label = "Chunking it smaller to fit it.", state = "running", expanded = True)
                _start = np.random.randint(0, len(st.session_state["full_content"]) - WORD_LIM)
                st.session_state["article_content"] = st.session_state["full_content"][_start:_start + WORD_LIM]
                status.update(label = "Chunking is done. Calculating the number of tokens for this QA Generation.....", state = "running", expanded = True)
                num_tokens = num_tokens_from_prompt(st.session_state["article_content"], "gpt-3.5-turbo-1106")
                if (num_tokens > WORD_LIM/4.):
                    status.update(label = "Still too large to fit in memory. Shrinking futher.", state = "running", expanded = True)
                    st.session_state["article_content"] = st.session_state["full_content"][_start:_start + WORD_LIM - (num_tokens - WORD_LIM//4)*4]
                num_tokens = num_tokens_from_prompt(st.session_state["article_content"], "gpt-3.5-turbo-1106")
                status.update(label = f"Chunking done. Tokens = {num_tokens}", state = "complete", expanded = False)
        else:
            st.session_state["article_content"] = st.session_state["full_content"]
        for i in range(n_questions):
            full_response = ""
            st.header("Question " + str(i+1) + " from " + st.session_state["article_id"] + " at " + st.session_state["article_url"])
            message_placeholder = st.empty()
            metadata = {"username": st.session_state.user_name, 
                        "article_id": st.session_state["article_id"],
                        "article_url": st.session_state["article_url"],
                        "claims" : n_claims
                        }
            tags = [f"claims-{n_claims}", st.session_state["article_id"], GPTVersion]
            with callbacks.collect_runs() as cb:
                for chunks in chain.stream({"prefix" : prefix, "NCLAIMS":n_claims, 
                                            "CONTEXT": st.session_state.get("article_content")
                                            }, 
                                           {"metadata": metadata, 
                                            "tags": tags
                                            }
                                           ):
                    full_response += (chunks or "")
                    message_placeholder.write(full_response + "▌")
                st.session_state.DataGen_run_id = cb.traced_runs[0].id
                st.session_state.run_url = client.read_run(st.session_state.DataGen_run_id).url
                st.session_state.share_run_url = client.share_run(st.session_state.DataGen_run_id)
            message_placeholder.write(full_response) 
            st.session_state.questions.append({"qnum" : f"Gen: {st.session_state.generation_count}, Q: {i}", 
                                               "content" : st.session_state["article_content"],
                                               "question": full_response.split("A:")[0], 
                                               "answer": full_response.split("A:")[-1],
                                               "trace_link": st.session_state.run_url,
                                               "share_link": st.session_state.share_run_url
                                               }
                                              )
            _, tmp_coll,__ = st.columns([1, 2, 1])
            with tmp_coll:
                st.subheader(f"""Link to trace [🛠️]({st.session_state.run_url})""")
            if st.session_state.get("contribute_to_eval") and st.session_state.get("dataset_name"):
                with st.form(f"Add to DataSet {st.session_state.dataset_name}", border = True):
                    QAINFO = st.session_state.questions[-1]
                    QANSWER = QAINFO["answer"].replace(":", "").replace("```", "").strip(",").strip("\n")
                    NCLAIMS = QANSWER.split("\"n_claims\"")[-1].split(",")[0].replace(" ","")
                    CLAIMS = QANSWER.split("\"claims\"")[-1].split("],")[0].strip("\n").strip(",") + "]"
                    COMPLETE_RESPONSE = QANSWER.split("\"complete_response\"")[-1].split("\"answers\"")[0].replace("\"", "").strip(",").strip("\n")
                    INDIVIDUAL_RESPONSE = QANSWER.split("\"answers\"")[-1].split("]\n}")[0].split("]}")[0] + "]"
                    #for kt in QANSWER.split("\"answers\""):
                    #    print (kt)
                    #    print ("------")
                    #print (INDIVIDUAL_RESPONSE)
                    INDIVIDUAL_RESPONSE = eval(INDIVIDUAL_RESPONSE)
                    st.header("Add Question to DataSet")
                    st.subheader("INPUT")
                    st.text_area("QUESTION", value = QAINFO["question"].split("Q:")[-1].strip("\n"), key = "dataset_questions")
                    st.text_input("NCLAIMS", value = NCLAIMS, key = "dataset_nclaims")
                    st.text_input("ARXIV_ID", value = st.session_state["article_id"], key = "dataset_arxiv_id")
                    st.subheader("OUTPUT")
                    st.text_area("CLAIMS", value = CLAIMS, key = "dataset_claims")
                    st.text_area("INDIVIDUAL_RESPONSE", value = str(INDIVIDUAL_RESPONSE), key = "dataset_individual_response")
                    st.text_area("COMPLETE_RESPONSE", value = COMPLETE_RESPONSE, key = "dataset_complete_response")
                    
                    submit = st.form_submit_button("Add to dataset", on_click = add_to_dataset)
            st.header("", divider = "rainbow")
        