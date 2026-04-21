# Agentic-Study-Assistant
Agentic RAG-based Study Assistant using LangGraph and ChromaDB with memory, routing, and self-evaluation to provide grounded, hallucination-free answers from documents.
# Study Assistant (Agentic RAG System)

## Overview
This project is an AI-powered study assistant that answers questions from uploaded documents using Retrieval-Augmented Generation (RAG) and LangGraph.

## Features
- Document-based Q&A (PDF/TXT)
- ChromaDB vector storage
- LangGraph agent architecture
- Memory with thread_id
- Self-evaluation to reduce hallucination
- Streamlit UI

## Tech Stack
- Python
- LangGraph
- ChromaDB
- SentenceTransformers
- Groq API
- Streamlit

## How to Run
```bash
pip install -r requirements.txt
streamlit run capstone_streamlit.py

This project enforces strict grounding and avoids hallucination by rejecting answers not present in retrieved context.
