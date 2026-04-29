📚 Agentic AI Study Assistant (RAG + LangGraph)

An AI-powered study assistant that answers questions from uploaded documents using Retrieval-Augmented Generation (RAG) and an agentic workflow built with LangGraph.

🚀 Overview

This project enables users to upload PDF/TXT documents and ask questions based strictly on their content.
The system retrieves relevant context using a vector database and generates grounded responses using an LLM.

Unlike typical chatbots, this assistant is designed to avoid hallucination by enforcing strict context-based answering.

✨ Features
📄 Document-based Q&A (PDF/TXT support)
🔍 Semantic search using vector embeddings
🧠 Agentic workflow (LangGraph) with:
Memory
Routing
Evaluation
🚫 Hallucination control (rejects unsupported queries)
🌐 Optional web search tool for out-of-scope queries
🖥️ Interactive UI using Streamlit
🏗️ Tech Stack
Language: Python
Agent Framework: LangGraph, LangChain
LLM: Groq (LLaMA 3)
Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
Vector DB: ChromaDB
Frontend: Streamlit
Evaluation: RAGAS
⚙️ How It Works
Document Ingestion
Upload PDF/TXT files
Split into chunks
Embedding & Storage
Convert text → embeddings
Store in ChromaDB
Query Processing
User asks a question
Retrieve relevant chunks
Agent Workflow
Memory → Retrieval → Routing → Answer generation
Optional tool (web search)
Response Generation
LLM answers using ONLY retrieved context
🧠 Hallucination Control

The system ensures reliability by:

Enforcing strict grounding (answers only from retrieved context)
Rejecting queries when context is insufficient
Using a self-evaluation step to validate responses
📊 Evaluation (RAGAS)
Metric	Score
Faithfulness	0.89
Answer Relevancy	0.87
Context Precision	0.82
🖥️ Running the Project
1️⃣ Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Set environment variables

Create a .env file:

GROQ_API_KEY=your_api_key_here
4️⃣ Run the app
streamlit run capstone_streamlit.py
📂 Project Structure
├── agent.py                # LangGraph agent logic
├── capstone_streamlit.py  # UI (Streamlit)
├── rag/                   # RAG pipeline
│   ├── loader.py
│   ├── embedder.py
│   ├── retriever.py
│   └── vectordb.py
├── data/                  # Documents
├── requirements.txt
└── README.md
🔥 Key Learnings
Building reliable AI systems requires more than just LLM calls
Proper retrieval and grounding are critical to reduce hallucination
Agent frameworks enable structured reasoning and tool usage
Evaluation (RAGAS) is essential for measuring real performance
🚀 Future Improvements
Hybrid search (BM25 + embeddings)
Better chunking strategies
Multi-document reasoning
UI enhancements
Deployment (cloud-based)
🤝 Contributing

Feel free to open issues or submit pull requests!

📌 Author

Siddharth Sinha

LinkedIn: https://linkedin.com/in/siddharth-sinha-3382
GitHub: https://github.com/Ledenol
⭐ If you like this project

Give it a star ⭐ — it helps a lot!

This project enforces strict grounding and avoids hallucination by rejecting answers not present in retrieved context.

## How to Run
```bash
pip install -r requirements.txt
streamlit run capstone_streamlit.py
