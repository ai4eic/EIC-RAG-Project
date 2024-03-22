import streamlit as st
import os, random
from datetime import datetime, timedelta
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
#from langchain_community.callbacks import TrubricsCallbackHandler
#from langchain_core.output_parsers import StrOutputParser
#from langchain.prompts import PromptTemplate
#from langchain_core.runnables import RunnableBranch
from app_utilities import *
from LangChainUtils.LLMChains import *
from langchain import callbacks
from langsmith import Client
from langchain_core.tracers.context import tracing_v2_enabled
from langchain.callbacks.tracers import LangChainTracer

client = Client()
SetHeader("AI4EIC-RAG ChatBot")

# Include some explanations

if not st.session_state.get("user_name"):
    st.error("Please login to your account first to further continue.")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TRUBRICS_EMAIL"] = st.secrets["TRUBRICS_EMAIL"]
os.environ["TRUBRICS_PASSWORD"] = st.secrets["TRUBRICS_PASSWORD"]


if st.secrets.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
# Creating OpenAIEmbedding()

embeddings = OpenAIEmbeddings()
# Defining some props of DB
SimilarityDict = {"Cosine similarity" : "similarity", "MMR" : "mmr"}

DBProp = {"LANCE" : {"vector_config" : {"db_name" : st.secrets["LANCEDB_DIR"], 
                                        "table_name" : "EIC_archive", 
                                        "embedding_function" : embeddings
                                        },
                     "search_config" : {"metric" : "similarity", 
                                        "search_kwargs" : {"k" : 100}
                                        },
                     "available_metrics" : ["Cosine similarity"]
                     },
          "CHROMA" : {"vector_config" : {"db_name" : st.secrets["CHROMADB_DIR"], 
                                         "embedding_function" : embeddings, 
                                         "collection_name" : "ARXIVS"
                                         },
                      "search_config" : {"metric" : "similarity",
                                         "search_kwargs" : {"k" : 100}
                                         },
                      "available_metrics" : ["Cosine similarity", "MMR"]
                      },
          "PINECONE" : {"vector_config" : {"db_api_key" : st.secrets["PINECONE_API_KEY"], 
                                            "index_name" : "llm-project", 
                                            "embedding_function" : embeddings
                                            },
                        "search_config" : {"metric" : "similarity", 
                                           "search_kwargs" : {"k" : 5}
                                           },
                        "available_metrics" : ["Cosine similarity", "MMR"]
                        },
          }
# Creating retriever
if "retriever_init" not in st.session_state:
    retriever = GetRetriever("PINECONE", DBProp["PINECONE"]["vector_config"], DBProp["PINECONE"]["search_config"])

@st.cache_data(ttl = 300)
def GetRunList(name):
    runInfo = []
    if not client.has_project(name):
        return runInfo
    run_list = client.list_runs(project_name = name, 
                                start_time = datetime.now() - timedelta(days = 7),
                                execution_order = 1
                                )
    for runs in run_list:
        if (client.run_is_shared(runs.id)):
            runInfo.append({"runID": runs.id, 
                            "runName" : runs.name, 
                            "runLink" : client.read_run_shared_link(runs.id)
                            }
                           )
    return runInfo

def OpenHistoryPage(run_id):
    st.session_state["selected_run_id"] = run_id
    st.switch_page("pages/4_View_History.py")
 
with st.sidebar:
    if (st.session_state.get("user_name")):
        with st.container():
            st.info("Select VecDB and Properties")
            db_type = st.selectbox("Vector DB", ["PINECONE"], key = "db_type")
            similiarty_score = st.selectbox("Retrieval Metric", DBProp[db_type]["available_metrics"], key = "similiarty_score")
            max_k = st.select_slider("Max K", options = [3, 5, 10, 20, 30, 40, 50, 100, 120], value = 10, key = "max_k")
            if st.button("Select Vector DB"):
                DBProp[db_type]["search_config"]["search_kwargs"]["k"] = max_k
                DBProp[db_type]["search_config"]["metric"] = SimilarityDict[similiarty_score]
                retriever = GetRetriever(db_type, DBProp[db_type]["vector_config"], DBProp[db_type]["search_config"])
                st.session_state["retriever_init"] = True
        with st.container(border = True, height = 400):
            st.header("Previous QA Chats")
            run_list = GetRunList(f"RAG-CHAT-{st.session_state.user_name}")
            for runs in run_list:
                #st.button(runs["runName"], key = str(runs["runID"]), on_click = OpenHistoryPage, args = (runs["runID"],),
                #              help = f"Click to open the chat history of " + runs["runName"] + " and you can find more information in langsmith here: " + runs["runLink"]
                #              )
                st.link_button(label = runs["runName"], url = runs["runLink"], help = f"Click to find more information in langsmith here: " + runs["runLink"]
                               )
            

if "retriever_init" in st.session_state:
    db_type = st.session_state["db_type"]
    similiarty_score = st.session_state["similiarty_score"]
    max_k = st.session_state["max_k"]
    DBProp[db_type]["search_config"]["search_kwargs"]["k"] = max_k
    DBProp[db_type]["search_config"]["metric"] = SimilarityDict[similiarty_score]
    retriever = GetRetriever(db_type, DBProp[db_type]["vector_config"], DBProp[db_type]["search_config"])

sss = """
llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0, 
                 callbacks=[
                     TrubricsCallbackHandler(
                         project="EIC-RAG-TestRun",
                         config_model={
                             "model" : "gpt-3.5-turbo-1106",
                             "temperature" : 0,
                             "max_tokens" : 4096,
                             "prompt_template" : "rag_prompt_custom",
                         },
                         tags = ["EIC-RAG-TestRun"],
                         user_id = st.session_state["user_name"],
                                             )
                     ], 
                 max_tokens=4096)
                 """
llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0,
                 max_tokens = 4096
                 )

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        _cola, _colb = st.columns([1, 1])
        with _cola:
            if message.get("trace_link"):
                st.subheader(f"[View the trace 🛠️]({message['trace_link']})")
        with _colb:
            if message.get("feedback"):
                st.subheader("Thanks for the feedback 📝")

def submit_feedback():
    client.create_feedback(run_id = st.session_state.run_id, 
                           key = "output_render",  
                           score = 1 if fdbk_output_render == "Yes" else 0,
                           created_at = datetime.now(),
                           created_by = st.session_state.user_name
                           )
    client.create_feedback(run_id = st.session_state.run_id, 
                           key = "output_quality", 
                           score = fdbk_output_quality,
                           created_at = datetime.now(),
                           created_by = st.session_state.user_name,
                           correction = {"COMPLETE_RESPONSE": st.session_state.get("feedback_expected_response", "")},
                           source_info = {"arxiv_id": st.session_state.get("feedback_source_info", "")}
                           )
    GetRunList.clear()

if "user_avatar" not in st.session_state:
    st.session_state["user_avatar"] = random.choice(["😊", "😉", "🤗"])
if "ai_avatar" not in st.session_state:
    st.session_state["ai_avatar"] = random.choice(["🐉", "🐙", "🍄"])

if prompt := st.chat_input("What is up? Ask anything about the Electron Ion Collider (EIC)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar = st.session_state["user_avatar"]):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar = st.session_state["ai_avatar"]):
        full_response = ""
        allchunks = None
        with st.spinner("Hmmmm deciding if I need to use Knowledge bank for this query..."):
            outdecide = DecideChain(llm).invoke({"question": prompt})
        infocontainer = st.empty()
        message_placeholder = st.empty()
        trace_link = st.empty()
        feedback_container = st.empty()
        if "more info" in outdecide.lower():
            infocontainer.info("Gathering info from Knowledge Bank for this query...")
            run_name = llm.invoke(f"""
                                  Rewrite the question below such that I can name the the question in a tab. 
                                  It has to be without any new lines. Make it no more than 2 words.
                                  {prompt}
                                  """).content
            trace_metadata = {"DBType": st.session_state.db_type, 
                              "similarity_score": st.session_state.similiarty_score, 
                              "max_k": st.session_state.max_k
                              }
            project_name = f"RAG-CHAT-{st.session_state.user_name}"
            tracer = LangChainTracer(project_name = project_name)
            RUNCHAIN = RunChatBot(llm, retriever
                                  ).with_config({"callbacks": [tracer],
                                                 "run_name": run_name, 
                                                 "metadata": trace_metadata
                                                 }
                                                )
            with tracing_v2_enabled(project_name) as cb:
                with callbacks.collect_runs() as ccb:
                    for chunk in RUNCHAIN.stream(prompt):
                        full_response += (chunk.get("answer") or "")
                        message_placeholder.markdown(full_response + "▌")
                    st.session_state.run_id = ccb.traced_runs[0].id
                    st.session_state.chat_run_id = ccb.traced_runs[0].id
                    st.session_state.share_run_url = client.share_run(st.session_state.run_id)
        elif "enough info" in outdecide.lower():
            st.session_state.share_run_url = None
            infocontainer.warning("I am going to answer this question with my knowledge.")
            for chunk in CreativeChain(llm).stream({"question" : prompt}):
                full_response += (chunk.get("answer") or "")
                message_placeholder.markdown(full_response + "▌")
        else:
            st.session_state.share_run_url = None
            infocontainer.error("I am not sure if I can answer this question. I will try to answer it with my knowledge.")
            for chunk in GeneralChain(llm).stream({"question" : prompt}):
                full_response += (chunk.get("answer") or "")
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response + "▌")
        if st.session_state.get("share_run_url"):
            trace_link.subheader(f"[View the trace 🛠️]({st.session_state.share_run_url})")
    with feedback_container.container(border = True):
        with st.form("Feedback for the generation"):
            _colf1, _colf2 = st.columns([1, 1])
            with _colf1:
                fdbk_output_render = st.selectbox("Has the output been displayed properly?", 
                                                ["Yes", "No"], index = None,
                                                key = "feedback_output_render"
                                                )
            with _colf2:
                fdbk_output_quality = st.number_input("Rate the quality of the output, Min = 0, Max = 5", 
                                                    min_value = 0, max_value = 5, 
                                                    key = "feedback_output_quality"
                                                    )
            fdbk_expected_response = st.text_area("What was your expected response?", value = "",
                                                  key = "feedback_expected_response",
                                                  placeholder = "What is an ideal response for this query?",
                                                  help = "What is an ideal response for this query?"
                                                  )
            fdbk_source_info = st.text_input("What was the source of the question?", value = "",
                                             key = "feedback_source_info",
                                             placeholder = "Where did you get this query from, ideally this is the arxiv_id that is used for testing purpose?",
                                             help = "Where did you get this query from?"
                                             )
            submit = st.form_submit_button("Submit Feedback", on_click = submit_feedback)
    st.session_state.messages.append({"role": "assistant", 
                                      "content": full_response, 
                                      "trace_link": st.session_state.share_run_url,
                                      "feedback": True if st.session_state.get("feedback_output_render") else False
                                      }
                                     )
