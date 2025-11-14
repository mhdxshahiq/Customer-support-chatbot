# 🤖 Customer Support RAG Chatbot

This project is a **Retrieval-Augmented Generation (RAG)** chatbot designed to answer user questions using internal support ticket data. It is built using the modular **LangChain** framework and powered by the **Gemini** LLM.

---

## ✨ Key Components

| Component | Purpose | Package |
| :--- | :--- | :--- |
| **LLM** | Generates answers and maintains conversation flow. | `ChatGoogleGenerativeAI` |
| **Embeddings** | Converts ticket text into numerical vectors. | `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) |
| **Vector Store** | Stores and retrieves relevant ticket data quickly. | `ChromaDB` (Persisted to `./chroma_db_hf`) |
| **Chain** | Orchestrates the retrieval of context and the generation of the final answer. | Custom LCEL Chain |

---

## ⚙️ Setup and Installation

### Prerequisites

* Python 3.9+
* A **Google API Key** for the Gemini model.

### 1. Installation

Create your Python virtual environment and install dependencies using your `requirements.txt` file:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
