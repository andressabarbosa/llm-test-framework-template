import os

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.chains.graph_qa.base import GraphQAChain
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.graphs import NetworkxEntityGraph
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langsmith import traceable

# --- LLM com fallback para gpt-3.5 ---
try:
    llm = ChatOpenAI(model="gpt-4", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))

# --- Prompt estruturado com parser ---
response_schemas = [
    ResponseSchema(name="resumo", description="Resumo do conteúdo abordado"),
    ResponseSchema(
        name="conceitos_chave", description="Lista de conceitos importantes mencionados"
    ),
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente que extrai informações estruturadas de textos."),
        ("user", "{input}"),
    ]
)


@traceable(name="LangChain - Structured Chain")
def run_chain(prompt: str) -> dict:
    """Executa chain estruturada com parser"""
    formatted_prompt = prompt_template.format_messages(input=prompt)
    output = llm(formatted_prompt)
    return parser.parse(output.content)


@traceable(name="LangChain - RAG Chain")
def run_rag_chain(query: str, docs_path: str) -> str:
    """Executa Retrieval-Augmented Generation com um loader local"""
    loader = TextLoader(docs_path)
    documents = loader.load()
    vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history")

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return rag_chain.run(query)


@traceable(name="LangChain - Graph QA Chain")
def run_graph_chain(input_text: str) -> str:
    """Executa uma chain de perguntas e respostas sobre grafos semânticos"""
    graph = NetworkxEntityGraph()
    chain = GraphQAChain.from_llm(llm=llm, graph=graph)
    return chain.run(input_text)


@traceable(name="LangChain - Agent + Tools")
def run_agent_with_tools(question: str) -> str:
    """Executa um agente com ferramenta externa DuckDuckGo"""
    search = DuckDuckGoSearchRun()
    tools = [
        Tool(name="Search", func=search.run, description="Use para pesquisar informações na web")
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent.run(question)
