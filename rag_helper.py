#!/usr/bin/env python
# coding: utf-8
import os
from typing import List, Dict
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Configuration
BUDGET_PDF_PATH = 'Budget_Speech.pdf'
COLLECTION_NAME = 'qa_index'
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from PDF file"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text(text: str) -> List[str]:
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

def setup_vector_store(chunks: List[str]):
    """Initialize and populate vector store"""
    client = QdrantClient("http://localhost:6333")
    
    # Recreate collection
   
    client.delete_collection(COLLECTION_NAME)
        
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    # Create embeddings and upload
    embeddings = EMBEDDING_MODEL.encode(chunks)
    ids = []
    payload = []

    for id, text in enumerate(chunks):
        ids.append(id)
        payload.append({"source": BUDGET_PDF_PATH, "content": text})

    
    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=embeddings,
        payload=payload,
        ids=ids,
        batch_size=256
    )
    return client

def search(client: QdrantClient, text: str, top_k: int = 5) -> List[Dict]:
    """Search for relevant text chunks"""
    query_embedding = EMBEDDING_MODEL.encode(text).tolist()
    return client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        query_filter=None, 
        limit=top_k
    )

def get_completion(client: OpenAI, question: str, context: str) -> str:
    """Get completion from OpenAI"""
    system_prompt = f"You are a helpful assistant who will answer questions about the budget of India. Use the following context to answer the question and do not make up the answer if the context doesn't have it: {context}"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please answer this question: {question}"}
        ]
    )
    return response.choices[0].message.content

