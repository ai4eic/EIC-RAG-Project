from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

from langchain.schema.runnable import RunnableMap

# Creating a Runnable Branch
# Query -> Decide -> Generate or RAG chain

def DecideAndRunChain(llm):
    decide_prompt = PromptTemplate.from_template("Templates/Decide_Prompts/decide_prompt_00.template")
    decide_chain = decide_prompt | llm | StrOutputParser()
    
    pass

# Creating a QA Runnable Branch
# From question + contextr generate question
def RunQuestionGeneration(llm):
    """The chain takes in the response template which should have the following template.
    {"prefix" : prefix, "NCLAIMS":n_claims, "CONTEXT": article_content}

    Args:
        llm (_type_): LLM in Langchain definition

    Returns:
        chain : QA chain
    """
    response = open("Templates/QA_Generations/response_01.template").read()
    qa_prompt = PromptTemplate.from_template(response)
    qa_chain = qa_prompt | llm | StrOutputParser()
    return qa_chain

def DecideChain(llm):
    decide_response = open("Templates/Decide_Prompts/decide_prompt_00.template", "r").read()
    decide_prompt = PromptTemplate.from_template(decide_response)
    decide_chain = decide_prompt | llm | StrOutputParser()
    return decide_chain

def CreativeChain(llm):
    creative_chain = (
        PromptTemplate.from_template(
            """
            You are an expert in answering questions about Hadronic physics and the upcoming Electron Ion Collider (EIC).
            But remember you do not have upto date information about the project nor you track its updates.
            You will not answer any question that is not related to the Electron Ion Collider (EIC) or Hadronic physics.
            You will politely decline answering about any other topic. However, to lighten the mood, you will respond with a joke or a quote.
            You are an expert in responding to a question in a professional fashion.
            Starting by greeting and thanking for the question. 
            Answer the question in a very comprehensive way with important numbers if relevant.
            Respond to the question in a fun, calm and professional fashion. 
            Make sure to write a comprehensive answer. 
            End the response with a funny joke or a quote related to the answer. Below is the question you need to respond to.
    Question: {question}
    """
        )
        | llm | {"answer" : StrOutputParser()}
    )
    return creative_chain

def GeneralChain(llm):
    general_chain = (
        PromptTemplate.from_template(
            """
    You are a familiar with the Electron Ion Collider (EIC) and its working. However you are no expert about EIC physics nor have upto date information on it
    Respond to the question by starting with saying, you are not sure of the answer but will try to answer at your best.
    Answer the question in a very comprehensive way.
    <question>
    {question}
    </question>
    """
        ) | llm | {"answer" : StrOutputParser()}
    )
    return general_chain
def RunChatBot(llm, retriever, template_dir = None):
    def format_docs(docs):
        unique_arxiv = list(set(doc.metadata['arxiv_id'] for doc in docs))
        mkdown = """# Retrieved documents \n"""
        for idx, u_ar in enumerate(unique_arxiv):
            mkdown += f"""{idx + 1}. <ARXIV_ID> {u_ar} <ARXIV_ID/> \n
            """
            for i, doc in enumerate(docs):
                if doc.metadata['arxiv_id'] == u_ar:
                    mkdown += """\t*\t""" + doc.page_content.strip("\n") + " \n"
        return mkdown
    
    if template_dir:
        response = open(f"{template_dir}/reponse_01.template", "r").read()
    else:
        response = open("Templates/reponse_01.template", "r").read()
    response_rewrite = """\
    Follow the instructions very very strictly. Do not add anything else in the response. Do not hallucinate nor make up answers.
    - The content below within the tags <MARKDOWN_RESPONSE> and </MARKDOWN_RESPONSE> is presented within a `st.markdown` container in a streamlit chat window. 
    - It may have some syntax errors and improperly arranged citations. 
    - Strictly do no modify the reference URL nor its text.
    - Identify unique reference URL links from the context below and cite them in the form of superscripts as.  
    - The new citations should be numerical and start from one. There has to be atleast one citation in the response.
    - Make sure to only use github flavoured markdown syntax for citations. The superscripts should not be in html tags.
    - Check for GitHub flavoured Markdown syntax and Importantly correct the syntax to be compatible with GitHub flavoured Markdown and specifically the superscripts, and arrange the new citations to be numerical starting from one.
    - The content may have latex commands as well. Edit them to make it compatible within Github flavoured markdown by adding $ before and after the latex command.
    - Make sure the citations are superscripted and has to be displayed properly when presented in a chat window. 
    - Do not include the <MARKDOWN_RESPONSE> and <MARKDOWN_RESPONSE/> tags in your answer.
    - Strictly do no modify the reference URL nor its text. Strictly have only Footnotes with reference links in style of GithubFlavoured markdown.
    <MARKDOWN_RESPONSE>
    {markdown_response}
    <MARKDOWN_RESPONSE/>
    """
    rag_prompt_custom = PromptTemplate.from_template(response)
    rag_prompt_rewrite = PromptTemplate.from_template(response_rewrite)
    
    rag_chain_from_docs = (
        {
            "context": lambda input: format_docs(input["documents"]),
            "question": itemgetter("question"),
        }
        | rag_prompt_custom
        | llm
        | {"markdown_response" : StrOutputParser()} | rag_prompt_rewrite | llm | StrOutputParser()
    )

    rag_chain_with_source = RunnableMap(
        {"documents": retriever, "question": RunnablePassthrough()}
    ) | {
        "answer": rag_chain_from_docs,
    }
    
    return rag_chain_with_source
