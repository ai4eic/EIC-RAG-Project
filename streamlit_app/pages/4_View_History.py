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
SetHeader("Viewing QA logs")

# Include some explanations

if not st.session_state.get("user_name"):
    st.error("Please login to your account first to further continue.")
    st.stop()

if st.session_state.get("user_mode", -1) <= 0:
    st.error("Seems like you do not have access to see these logs. Contact ai4eic team for further assistance")
    st.stop()

if st.secrets.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]

@st.cache_resource(ttl = 300)
def GetRunList(name, latestID):
    runs = {}
    if not client.has_project(name):
        return None
    run_list = client.list_runs(project_name = name, 
                                start_time = datetime.now() - timedelta(days = 7),
                                execution_order = 1
                                )
    for run in run_list:
        if (client.run_is_shared(run.id)):
            runs[run.id] = {"run" : run, 
                            "runLink" : client.read_run_shared_link(run.id)
                            }
    return runs
# get the run from client

client = Client()
run_list = GetRunList(f"RAG-CHAT-{st.session_state.user_name}", st.session_state.get("chat_run_id", ""))

if not run_list:
    st.warning("No QA Chat logs found")
    st.stop()

selected_run_id = st.selectbox("Select the QA Chat",
                            [k for k in run_list.keys()],
                            key = "selected_run_id",
                            format_func = lambda x: f"""{run_list[x]["run"].name}""",
                            help = "Select the QA Chat to load"
                            )

selected_run = run_list[selected_run_id]["run"]

st.header(f"Viewing logs for run_id: [{selected_run.id}]" + "({})".format(run_list[selected_run_id]["runLink"]), 
          divider = "rainbow"
          )

st.subheader("INPUTS")

with st.container():
    for k, v in selected_run.inputs.items():
        st.write(f"{k}:")
        st.write(f"{v}")
    st.markdown("---")

st.subheader("OUTPUTS")
with st.container():
    for k, v in selected_run.outputs.items():
        st.write(f"{k}")
        st.write(f"{v}")
st.write("---")


