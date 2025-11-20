import os
import json
from datetime import datetime
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# LangChain modules
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------------------------------
# ENV + MODELS
# ----------------------------------------------------
load_dotenv()

clf = joblib.load("models/department_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

chat_history = []


# ----------------------------------------------------
# Extract ONLY the solution text
# ----------------------------------------------------
def extract_solution(text: str):
    text = text.replace("\n", " ")

    if "SOLUTION:" in text:
        return text.split("SOLUTION:", 1)[1].strip()

    return ""


# ----------------------------------------------------
# Normalize department name
# ----------------------------------------------------
def normalize_department(dept: str):
    dept = dept.lower().strip()

    mapping = {
        "deliver": "delivery",
        "delivery issue": "delivery",
        "shipping": "delivery",

        "account": "accounts",
        "acc": "accounts",

        "tech": "technical",
        "app": "technical",
        "crash": "technical",
    }

    return mapping.get(dept, dept)




# ----------------------------------------------------
# Load department retriever
# ----------------------------------------------------
def get_department_retriever(department: str):

    base_dir = os.getcwd()
    folder = os.path.join(base_dir, "chroma_db", department)

    if not os.path.exists(folder):
        raise Exception(f"❌ Vector DB folder not found: {folder}")

    vectordb = Chroma(
        persist_directory=folder,
        embedding_function=embeddings
    )

    return vectordb.as_retriever(search_kwargs={"k": 5})


# ----------------------------------------------------
# Predict department
# ----------------------------------------------------
def predict_department(text: str):
    x = vectorizer.transform([text])
    return clf.predict(x)[0]


# ----------------------------------------------------
# Log routing
# ----------------------------------------------------
def save_routing_to_json(query, dept):
    os.makedirs("logs", exist_ok=True)
    file = "logs/routing_log.json"

    if os.path.exists(file):
        logs = json.load(open(file))
    else:
        logs = []

    logs.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "department": dept
    })

    json.dump(logs, open(file, "w"), indent=4)





# ----------------------------------------------------
# Prompt template
# ----------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a strict RAG customer-support bot.

RULES:
1. Use ONLY the provided SOLUTIONS in context.
2. If you don’t find the answer → reply EXACTLY: Not found.
3. NO extra explanation. NO guessing.
4. Answer short and clear."""
     ),
    ("system", "Chat History:\n{history}"),
    ("system", "CONTEXT:\n{context}"),
    ("human", "{question}")
])


# ----------------------------------------------------
# RAG Chain
# ----------------------------------------------------
def rag_chain(question: str, history: str, department: str):

    retriever = get_department_retriever(department)
    documents = retriever.invoke(question)

    if not documents:
        return "Not found."

    solutions = []
    for doc in documents:
        sol = extract_solution(doc.page_content)
        if sol:
            solutions.append(sol)

    if not solutions:
        return "Not found."

    context_text = "\n".join(solutions)

    chain = prompt | llm

    response = chain.invoke({
        "question": question,
        "context": context_text,
        "history": history
    })

    if not response or not response.content.strip():
        return "Not found."

    return response.content.strip()


# ----------------------------------------------------
# FASTAPI
# ----------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    department: str


# ----------------------------------------------------
# MAIN CHAT ENDPOINT
# ----------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    query = request.message.strip().lower()

    # Clear chat history
    if query in ["clear history", "reset", "clear"]:
        chat_history.clear()
        return ChatResponse(reply="History Cleared.", department="none")

    # Predict and normalize department
    predicted = predict_department(query)
    department = normalize_department(predicted)

    save_routing_to_json(query, department)

    # Use only last 3 queries
    limited_history = chat_history[-3:]
    history_text = "\n".join(limited_history)

    reply = rag_chain(query, history_text, department)

    chat_history.append(query)

    return ChatResponse(reply=reply, department=department)



@app.get("/")
def home():
    return {"message": "Backend is running!"}
