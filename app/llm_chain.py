from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains.graph_qa.base import GraphQAChain
from langchain.graphs import NetworkxEntityGraph
from langsmith import traceable

llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

response_schemas = [
    ResponseSchema(name="resumo", description="Resumo do conteúdo abordado"),
    ResponseSchema(name="conceitos_chave", description="Lista de conceitos importantes mencionados"),
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente que extrai informações estruturadas de textos."),
    ("user", "{input}")
])

@traceable(name="LangChain - Structured Chain")
def run_chain(prompt: str) -> dict:
    formatted_prompt = prompt_template.format_messages(input=prompt)
    output = llm(formatted_prompt)
    return parser.parse(output.content)

@traceable(name="LangChain - RAG Chain")
def run_rag_chain(query: str, docs_path: str) -> str:
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
        return_source_documents=True
    )
    return rag_chain.run(query)

@traceable(name="LangChain - Graph QA Chain")
def run_graph_chain(input_text: str) -> str:
    graph = NetworkxEntityGraph()
    chain = GraphQAChain.from_llm(llm=llm, graph=graph)
    return chain.run(input_text)

@traceable(name="LangChain - Agent + Tools")
def run_agent_with_tools(question: str) -> str:
    search = DuckDuckGoSearchRun()
    tools = [
        Tool(name="Search", func=search.run, description="Use para pesquisar informações na web")
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent.run(question)