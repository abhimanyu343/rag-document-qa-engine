"""
RAG retrieval and QA chain — semantic + BM25 hybrid retrieval with source citation.
"""
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import chromadb

CHROMA_PATH = "db/chroma"


def build_qa_chain(local: bool = False, collection_name: str = "documents", k: int = 5):
    """
    Build a conversational RAG chain with hybrid retrieval and memory.
    
    Args:
        local: Use local Ollama LLM and HuggingFace embeddings
        collection_name: ChromaDB collection to query
        k: Number of chunks to retrieve
    
    Returns:
        ConversationalRetrievalChain ready for multi-turn Q&A
    """
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5") if local else OpenAIEmbeddings(model="text-embedding-3-small")
    
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    vectorstore = Chroma(client=client, collection_name=collection_name, embedding_function=embeddings)
    
    # Semantic retriever with MMR for diversity
    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7}
    )
    
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5
    )
    
    if local:
        from langchain_community.llms import Ollama
        llm = Ollama(model="llama3", temperature=0.1)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=semantic_retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    return chain


def query(chain, question: str) -> dict:
    """Run a query and return answer with source citations."""
    result = chain.invoke({"question": question})
    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "file": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "excerpt": doc.page_content[:200] + "..."
        })
    return {
        "answer": result["answer"],
        "sources": sources,
        "num_sources": len(sources)
    }
