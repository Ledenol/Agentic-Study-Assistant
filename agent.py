# IMPORTS
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS
import os

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# STATE
class AgentState(TypedDict):
    question: str
    messages: List[str]
    retrieved: str
    tool_result: str
    answer: str

# RAG SETUP
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection("study_assistant")


def load_documents(folder_path="data"):
    documents = []

    if not os.path.exists(folder_path):
        return documents

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                documents.extend(loader.load())

            elif file.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
                documents.extend(loader.load())
        except:
            continue

    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


def embed_texts(texts):
    return embedder.encode(texts).tolist()


def setup_rag():
    global collection

    try:
        client.delete_collection("study_assistant")
    except:
        pass

    collection = client.get_or_create_collection("study_assistant")

    docs = load_documents("data")

    if not docs:
        return

    chunks = chunk_documents(docs)
    texts = [c.page_content for c in chunks]
    embeddings = embed_texts(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[f"id_{i}" for i in range(len(texts))]
    )

# RETRIEVAL
def retrieve_docs(query, k=3):
    query_embedding = embed_texts([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    docs = results["documents"][0]
    docs = [d for d in docs if d and len(d.strip()) > 30]

    return "\n".join(docs)

# WEB TOOL
def web_search(query):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            return "\n".join([r["body"] for r in results])
    except:
        return "No results found."

# NODES
def memory_node(state: AgentState):
    msgs = state.get("messages", [])
    msgs.append(state["question"])
    return {"messages": msgs[-6:]}


def retrieval_node(state: AgentState):
    docs = retrieve_docs(state["question"])
    return {"retrieved": docs}


def tool_node(state: AgentState):
    result = web_search(state["question"])
    return {"tool_result": result}


def router(state: AgentState):
    context = state.get("retrieved", "")
    if not context or len(context.strip()) < 20:
        return "tool"
    return "answer"


def answer_node(state: AgentState):
    context = state.get("retrieved", "")
    tool_data = state.get("tool_result", "")

    source = context if context else tool_data

    if not source:
        return {"answer": "I don't know."}

    prompt = f"""
Answer using ONLY the provided information.

{source}

Question: {state['question']}
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}


def save_node(state: AgentState):
    msgs = state["messages"]
    msgs.append(state["answer"])
    return {"messages": msgs}

# GRAPH
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("memory", memory_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("memory")

    graph.add_edge("memory", "retrieve")

    graph.add_conditional_edges(
        "retrieve",
        router,
        {
            "answer": "answer",
            "tool": "tool"
        }
    )

    graph.add_edge("tool", "answer")
    graph.add_edge("answer", "save")
    graph.add_edge("save", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

# INIT
setup_rag()
app = build_agent()

# HELPER
def ask(question, thread_id="1"):
    result = app.invoke(
        {
            "question": question,
            "messages": []
        },
        config={"configurable": {"thread_id": thread_id}}
    )
    return result["answer"]
