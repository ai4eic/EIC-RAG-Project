from openai import OpenAI
import streamlit as st

import os
from trubrics import Trubrics

import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import TrubricsCallbackHandler
import os
import time

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TRUBRICS_EMAIL"] = st.secrets["TRUBRICS_EMAIL"]
os.environ["TRUBRICS_PASSWORD"] = st.secrets["TRUBRICS_PASSWORD"]

db = lancedb.connect("/mnt/d/LLM-Project/my-app/lancedb_meta_data")
table = db.open_table("EIC_archive")
embeddings = OpenAIEmbeddings()
vectorstore = LanceDB(connection = table, embedding = embeddings)

retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs={"k" : 100})

with st.sidebar:
    with st.form("User Name"):
        st.info("By providing you name, you agree that all the prompts and responses will be recorded and will be used to further improve RAG methods")
        name = st.text_input("What's your name?")
        submitted = st.form_submit_button("Submit and start")
        if submitted:
            for key in st.session_state:
                del st.session_state[key]
            st.session_state["user_name"] = name

if "user_name" not in st.session_state:
    st.stop()
llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0, 
                 callbacks=[
                     TrubricsCallbackHandler(
                         project="EIC-RAG-TestRun",
                         tags = ["EIC-RAG-TestRun"],
                         user_id = st.session_state["user_name"],
                                             )
                     ], 
                 max_tokens=4096)

from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


def format_docs(docs):
    return f"\n\n".join(f'{i+1}. ' + doc.page_content.strip("\n") + f"<ARXIV_ID> {doc.metadata['arxiv_id']} <ARXIV_ID/>" for i, doc in enumerate(docs))


from langchain.prompts import PromptTemplate


response = """\
You are an expert in providing up to date information about the Electron Ion Collider (EIC), tasked with answering any question. You greet people when greeted. \
about EIC based only on the provided context. You shall strictly not answer questions anything other than EIC related questions. \
Refrain any other topics by saying you will not answer questions about them and Exit right away here. DO NOT PROCEED. \
You are not allowed to use any other sources other than the provided search results. \

Generate a comprehensive, and informative answer strictly within 200 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. You should use bullet points in your answer for readability. Make sure to break down your answer into bullet points.\
You should not hallicunate nor build up any references, Use only the `context` html block below and do not use any text within <ARXIV_ID> and </ARXIV_ID> except when citing in the end. 
Make sure not to repeat the same context. Be specific to the exact question asked for.\
    
Here is the response template:
---
# Response template 

- Start with a greeting and a summary of the user's query
- Use bullet points to list the main points or facts that answer the query using the information within the tags <context> and <context/>.  
- After answering, analyze the respective source links provided within <ARXIV_ID> and </ARXIV_ID> and keep only the unique links for the next step. Try to minimize the total number of unique links with no more than 10 unique links for the answer.
- You will strictly use no more than 10 most unique links for the answer.
- Use bulleted list of superscript numbers within square brackets to cite the sources for each point or fact. The numbers should correspond to the order of the sources which will be provided in the end of this reponse. Note that for every source, you must provide a URL.
- End with a closing remark and a list of sources with their respective URLs as a bullet list explicitly with full links which are enclosed in the tag <ARXIV_ID> and </ARXIV_ID> respectively.\
---
Here is how an response would look like. Reproduce the same format for your response:
---
# Example response

Hello, thank you for your question about Retrieval Augmented Generation. Here are some key points about RAG:

- Retrieval Augmented Generation is a technique that combines the strengths of pre-trained language models and information retrieval systems to generate responses or content by leveraging external knowledge[^1^] [^2^]
- RAG can be useful when the pre-trained language model alone may not have the necessary information to generate accurate or sufficiently detailed responses, since standard language models like GPT-4 are not capable of accessing real-time or post-training external information directly[^1^] [^3^]
- RAG uses a vector database such as Milvus to index and retrieve relevant documents or text snippets from a knowledge source, and provides them as additional context for the language model[^4^] [^5^]
- RAG can benefit from adding citations to the generated outputs, as it can improve their factual correctness, verifiability, and trustworthiness[^6^] [^7^]

I hope this helps you understand more about RAG.

## Sources

* [^1^][1]: http://arxiv.org/abs/2308.03393v1 

* [^2^][2]: http://arxiv.org/abs/2308.03393v1 

* [^3^][3]: http://arxiv.org/abs/2307.08593v1 

* [^4^][4]: http://arxiv.org/abs/2202.05981v2 

* [^5^][5]: http://arxiv.org/abs/2210.09287v1 

* [^6^][6]: http://arxiv.org/abs/2242.05981v2 

* [^7^][7]: http://arxiv.org/abs/2348.05293v1 

---

Where each of the references are taken from the corresponding <ARXIV_ID> in the context. Strictly do not provide title for the references \
Strictly do not repeat the same links. Use the numbers to cite the sources. \

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." or greet back. Don't try to make up an answer. Write the answer in the form of markdown bullet points.\
Make sure to highlight the most important key words in bold font. Dot repeat any context nor points in the answer.\

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. The context are numbered based on its knowledge retrival and increasing cosine similarity index. \
Make sure to consider the order in which they appear context appear. It is an increasing order of cosine similarity index.\
The contents are formatted in latex, you need to remove any special characters and latex formatting before cohercing the points to build your answer.\
Write your answer in the form of markdown bullet points. You can use latex commands if necessary.
You will strictly cite no more than 10 unqiue citations at maximum from the context below.\
Make sure these citations have to be relavant and strictly do not repeat the context in the answer.

<context>
    {context}
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." or greet back. Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
Question: {question}
"""
rag_prompt_custom = PromptTemplate.from_template(response)

from operator import itemgetter

from langchain.schema.runnable import RunnableMap

rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | rag_prompt_custom
    | llm
    | StrOutputParser()
)
rag_chain_with_source = RunnableMap(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "answer": rag_chain_from_docs,
}

st.warning("This project is being continuously developed. Please report any feedback to ai4eic@gmail.com")

col1, col2 = st.columns(2)
with col1:
    st.image("https://indico.bnl.gov/event/19560/logo-410523303.png")
with col2:
    st.title("AI4EIC Agent")

st.sidebar.title("Data Collection")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = ""
        allchunks = None
        with st.spinner("Gathering info from Knowledge Bank and writing response..."):
            allchunks = rag_chain_with_source.stream(prompt)
            message_placeholder = st.empty()
            for chunk in allchunks:
                full_response += (chunk.get("answer") or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
