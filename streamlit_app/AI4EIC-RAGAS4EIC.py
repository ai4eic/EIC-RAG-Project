import os, sys
import streamlit as st
from streamlit_react_flow import react_flow

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Please set the OPENAI_API_KEY secret in your secrets.toml file.")
    st.stop()

os.environ["PROJECT_DIR"] = os.getcwd()
sys.path.append(os.environ["PROJECT_DIR"])

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TRUBRICS_EMAIL"] = st.secrets["TRUBRICS_EMAIL"]
os.environ["TRUBRICS_PASSWORD"] = st.secrets["TRUBRICS_PASSWORD"]
os.environ["STREAMLIT_THEME_BASE"] = "dark"

if st.secrets.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
    os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
    os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]

st.set_page_config(
    page_title="AI4EIC-RAG QA-ChatBot",
    page_icon="https://indico.bnl.gov/event/19560/logo-410523303.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://eic.ai',
        'Report a bug': "https://github.com/wmdataphys/EIC-RAG-Project",
        'About': "# AI4EIC RAG System",
    }
)

st.warning("This project is being continuously developed. Please report any feedback to ai4eic@gmail.com")
col_l, col1, col2, col_r = st.columns([1, 3, 3, 1])

with col1:
    st.image("https://indico.bnl.gov/event/19560/logo-410523303.png")
with col2:
    st.title("""AI4EIC-RAG System""", anchor = "AI4EIC-RAG-QA-Bot", help = "Will Link to arxiv proceeding here.")

if st.session_state.get("user_name"):
    with st.sidebar:
        st.write("# Welcome back!")
        st.write("### " + 
                 st.session_state.get("first_name", "") + 
                 " " + 
                 st.session_state.get("last_name", "")
                 )

MainFrame = st.container()
with MainFrame:
    MainFrame.title("Retrieval Augmented Generation System for EIC (RAGS4EIC)")
    MainFrame.markdown("This is a project currently being developed to build a RAG based system for the upcoming Electron Ion Collider")
    MainFrame.subheader("", divider = "rainbow")
    MainFrame.header("What is RAG and how it can be used for EIC ?")
    Im_col1, Im_col2 = st.columns([1, 1])
    with Im_col1:
        st.image("streamlit_app/Resources/assets/images/What_is_RAG.png")
        st.image("streamlit_app/Resources/assets/images/Ingestion_full.png")
    with Im_col2:
        st.image("streamlit_app/Resources/assets/images/Why_RAG.png")
        st.image("streamlit_app/Resources/assets/images/Pipeline.png")
    MainFrame.subheader("", divider = "rainbow")
    MainFrame.header("Stages of RAG")
    MainFrame.markdown("There are distinctively 3 stages in RAG. Namely, Ingestion, Retrieval and Content Fusion Generation.")
    MainFrame.subheader("Ingestion", divider = "green")
    MainFrame.markdown("""
                       Ingestion in Retrieval-Augmented Generation (RAG) 
                       is a crucial process that involves the preparation and 
                       organization of data to be used by the model.
                       """)
    with st.expander("Expand for more details"):
        st.write(open("streamlit_app/Resources/Markdowns/ingestion.md", "r").read())
        st.image("streamlit_app/Resources/assets/images/ingestion.png")
    
    MainFrame.subheader("Retrieval", divider = "green")
    MainFrame.markdown("""
                       The process of information retrieval 
                       from a vector database given a user query involves several steps:
                       """)
    with st.expander("Expand for more details"):
        st.markdown(open("streamlit_app/Resources/Markdowns/Retrieval.md", "r").read())
        st.image("streamlit_app/Resources/assets/images/mermaid-retrieval.png")
    MainFrame.subheader("Content Fusion Generation", divider = "green")
    MainFrame.markdown("""
                       The process of content fusion and response generation involves several steps:
                       """)
    with st.expander("Expand for more details"):
        st.markdown(open("streamlit_app/Resources/Markdowns/content-fusion.md", "r").read())
        st.image("streamlit_app/Resources/assets/images/mermaid-content.png")
    MainFrame.subheader("Complete Response Generation", divider = "green")
    img_coll, img_colc, img_colr = st.columns([1, 2, 1])
    with img_colc:
        st.image("streamlit_app/Resources/assets/images/mermaid-content-retrieval-full.png", caption = "Coomplete response generation strategy.")
    MainFrame.subheader("", divider = "rainbow")
    MainFrame.header("Project Milestones")
    st.write(open("streamlit_app/Resources/Markdowns/Project_Milestones.md", "r").read())
    MainFrame.subheader("", divider = "rainbow")
    
