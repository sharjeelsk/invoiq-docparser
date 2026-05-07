from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List

import os
load_dotenv()

# ── Shared setup ──────────────────────────────────────────────────────────────

file_path = "./sample.pdf"
loader = PyPDFLoader(file_path)
loaded_docs = loader.load()  # keep original loader output, don't overwrite later

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ── Pipeline 1: RAG for user queries ─────────────────────────────────────────

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

# Split each page's text into chunks
chunks = []
for doc in loaded_docs:
    chunks.extend(text_splitter.split_text(doc.page_content))

# Wrap chunks as Documents for the vector store
chunk_docs = [Document(page_content=chunk) for chunk in chunks]

# Build vector store and retriever
vector_store = InMemoryVectorStore.from_documents(chunk_docs, embeddings_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# RAG prompt — answers a user question using retrieved context
rag_prompt = PromptTemplate(
    template="""Use the following invoice context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"],
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain: question → retrieve → format → prompt → model → string output
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

# ✅ Pass a plain string to the retriever — NOT a dict
rag_result = rag_chain.invoke("Which products were bought in March?")
print("=== RAG Answer ===")
print(rag_result)


# ── Pipeline 2: Structured JSON extraction from the full document ─────────────

class Item(BaseModel):
    description: str
    price: float

class Invoice(BaseModel):
    invoice_number: str
    date: str
    total_amount: float
    items: List[Item]

output_parser = PydanticOutputParser(pydantic_object=Invoice)

extraction_prompt = PromptTemplate(
    template="""Extract the following information from the invoice:
1. Invoice Number
2. Date
3. Total Amount
4. Items (with description and price)

{format_instructions}

Invoice content:
{invoice_content}
""",
    input_variables=["invoice_content"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

# Extraction chain: prompt → model → Pydantic object
extraction_chain = extraction_prompt | model | output_parser

# Use the raw loaded document text (not the chunked RAG docs)
extraction_result = extraction_chain.invoke({
    "invoice_content": loaded_docs[0].page_content
})

print("\n=== Extracted Invoice (structured) ===")
print(extraction_result)
print(f"\nInvoice #: {extraction_result.invoice_number}")
print(f"Date:      {extraction_result.date}")
print(f"Total:     ${extraction_result.total_amount}")
for item in extraction_result.items:
    print(f"  - {item.description}: ${item.price}")