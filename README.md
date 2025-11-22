# 🧠 Mental Health RAG Chatbot (Gemini + LangChain + Chroma)

A lightweight mental-health conversational AI assistant built using:

- **Google Gemini 2.0 Flash**
- **LangChain**
- **ChromaDB**
- **Retrieval-Augmented Generation (RAG)**
- **Conversation Memory (last 4 turns = 8 messages)**
- **Safety filters (anti-jailbreak + topic restriction)**

This chatbot ONLY talks about emotional well-being and blocks unsafe or unrelated topics.

---

## ✨ Features

### 🔹 1. Retrieval-Augmented Generation (RAG)
The bot retrieves the most relevant answers from your mental-health dataset stored in **ChromaDB**.

### 🔹 2. Conversation Memory  
Remembers the **last 4 conversation turns** (8 messages total).  
Makes replies more natural and contextual.

### 🔹 3. Safety Guardrails  
Prevents harmful prompts like:

- *ignore previous*
- *jailbreak*
- *switch role*
- *system override*

And refuses off-topic questions politely.

### 🔹 4. Text Summarization  
RAG chunks are summarized before generating the final response.

---

## 📁 Project Structure

│── main.py → Chatbot logic + safety + memory + RAG
│── rag_pipeline.py → ChromaDB retriever
│── ingest.py → CSV → chunks → embeddings → Chroma
│── system_prompt.py → Base system instruction
│── data.csv → Your mental-health FAQ dataset
│── README.md


🧠 How It Works (Workflow)
1. User enters a question

↓

2. Bot checks: Is topic related to mental health?

↓

3. Retrieves relevant chunks from ChromaDB

↓

4. Summarizes chunks using Gemini

↓

5. Builds final prompt with:

summary

memory

user query
↓

6. Gemini generates a safe response

↓

7. Memory updated (max 4 turns)
