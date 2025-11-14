import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

loader = CSVLoader(file_path="data/tickets.csv")
docs = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,              
    persist_directory="./chroma_db_hf" 
)

# print("Embedded and stored in chromaDB")

# print("Document text:")
# print(docs[0].page_content)

# # Generate the embedding (vector)
# vector = embeddings.embed_query(docs[0].page_content)

# print("\n Embedding vector:")
# print(vector)

# print(f"\n Vector length: {len(vector)}")