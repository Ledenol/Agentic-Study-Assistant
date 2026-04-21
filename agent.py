# =========================
# IMPORTS
# =========================
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os

# LLM 
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key="API_KEY_HERE"
)
# STATE
class AgentState(TypedDict):
    question: str
    messages: List[str]
    retrieved: str
    answer: str
    eval_retries: int

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

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        elif file.endswith(".txt"):
            loader = TextLoader(path)
            documents.extend(loader.load())

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

# RETRIEVAL (STRICT)
def retrieve_docs(query, k=3):
    query_embedding = embed_texts([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    docs = results["documents"][0]

    # Filter weak results
    docs = [d for d in docs if d and len(d.strip()) > 30]

    return "\n".join(docs)

# NODES
def memory_node(state: AgentState):
    msgs = state.get("messages", [])
    msgs.append(state["question"])
    return {"messages": msgs[-6:]}


# 🔥 FORCE RAG ONLY
def retrieval_node(state: AgentState):
    docs = retrieve_docs(state["question"])
    return {"retrieved": docs}


def answer_node(state: AgentState):
    context = state.get("retrieved", "")


    if not context or len(context.strip()) < 20:
        return {
            "answer": "I don't know based on the provided documents."
        }

    prompt = f"""
You are a STRICT study assistant.

RULES:
- Answer ONLY from the context
- DO NOT use prior knowledge
- DO NOT guess
- If not clearly in context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{state['question']}
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}


def eval_node(state: AgentState):
    # simple eval (enough for capstone)
    retries = state.get("eval_retries", 0) + 1

    return {
        "eval_retries": retries
    }


def save_node(state: AgentState):
    msgs = state["messages"]
    msgs.append(state["answer"])
    return {"messages": msgs}

# GRAPH
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("memory", memory_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("memory")

    graph.add_edge("memory", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "eval")
    graph.add_edge("eval", "save")
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
            "messages": [],
            "eval_retries": 0
        },
        config={"configurable": {"thread_id": thread_id}}
    )

    return result["answer"]
