import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from sentence_transformers import CrossEncoder

from langchain_core.documents import Document

load_dotenv()

# Cache Model Weights in Memory
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device' : 'cpu'},
        encode_kwargs={'normalize_embeddings' : True}
    )

@st.cache_resource
def load_reranker_model():
   return  CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

EMBEDDINGS = load_embedding_model()
RERANKER = load_reranker_model()

def extract_documents_from_pdfs(uploaded_files):
    raw_documents = []
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        for page_idx, page in enumerate(pdf_reader.pages, start=1):
            content = page.extract_text()
            if content:
                raw_documents.append(
                    Document(
                        page_content=content,
                        metadata={"source" : uploaded_file.name, "page": page_idx}
                    )
                )
    return raw_documents

def split_into_chunked_docs(raw_documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=120, 
        separators=["\n\n", "\n", ".", " "],
        length_function=len,
    )
    chunks = text_splitter.split_documents(raw_documents)
    return chunks

def get_vector_store(chunked_docs):
    vectorstore = FAISS.from_documents(documents=chunked_docs, embedding=EMBEDDINGS)
    return vectorstore

def get_conversation_chain(vector_store):
    primary_llm = ChatGoogleGenerativeAI(
        model="gemini-3.7-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    fallback_llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    local_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
    )

    llm = primary_llm.with_fallbacks(
        [fallback_llm, local_llm],
        exceptions_to_handle=(Exception,)
    )

    initial_retriever = vector_store.as_retriever(search_kwargs={"k":10})

    def rerank_candidates(query):
        candidate_docs = initial_retriever.invoke(query)

        if not candidate_docs:
            return ""

        pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = RERANKER.predict(pairs)

        ranked_docs = [
            doc for _, doc in sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
        ][:3] #Retain top-3 precision-ranked passages

        return "\n\n".join(doc.page_content for doc in ranked_docs)

    prompt = ChatPromptTemplate.from_template("""
    You are a document question-answering assistant.

    Do NOT assume facts or hallucinate numbers not supported by the context.
    - If the user asks for a summary or an overview of the document, provide a high-level synthesis based strictly on whatever information is available in the context.
    - If a specific factual detail or metric is not found in the context, respond with:
    "I could not find that specific detail in the uploaded documents."

    Context:
    {context}
    
    Question:
    {question}
    """)

    chain = (
        {
            "context" : RunnableLambda(rerank_candidates),
            "question" : RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def handle_userinput(user_question):
    response = st.session_state.conversation.invoke(user_question)

    st.session_state.chat_history.append(
        {"role": "user", "content": user_question}
    )

    st.session_state.chat_history.append(
        {"role": "bot", "content": response}
    )

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        else :
            st.write(f"**Bot:** {message['content']}")

    if "vector_store" in st.session_state and st.session_state.vector_store:
        with st.expander("📚 Inspect Cited Sources & Cross-Encoder Scores"):
            candidates = st.session_state.vector_store.as_retriever(search_kwargs={"k": 10}).invoke(user_question)
            
            pairs = [[user_question, doc.page_content] for doc in candidates]
            scores = RERANKER.predict(pairs)
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)[:3]

            for rank, (score, doc) in enumerate(ranked, start=1):
                source_name = doc.metadata.get("source", "Uploaded Document")
                page_num = doc.metadata.get("page", "N/A")
                st.markdown(f"**Rank {rank}** — File: `{source_name}` | **Page: {page_num}** | Score: `{score:.3f}`")
                st.caption(doc.page_content)
                st.divider()
    

def main():
    st.set_page_config(
        page_title="Chat with multiple PDFs", 
        page_icon=":books:"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question :
        if st.session_state.conversation is None:
            st.warning("Please upload and process a document before asking questions.")
        else:
            handle_userinput(user_question)
    
    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader(
            "Upload your PDFs here and press click on 'Process'", accept_multiple_files=True
        )

        if st.button("Process"):
            if not uploaded_files:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Processing and Indexing... "):
                    raw_documents = extract_documents_from_pdfs(uploaded_files)
                    chunked_docs = split_into_chunked_docs(raw_documents)

                    vector_store = get_vector_store(chunked_docs)

                    st.session_state.vector_store = vector_store
                    
                    #conversation chain
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.session_state.chat_history = []
                    st.success("Processing complete! You can now start chatting")
        
    
if __name__ == '__main__':
    main()
