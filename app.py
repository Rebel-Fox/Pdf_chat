import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough




print("Loading lightweight BGE embedding model")
LAPTOP_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device' : 'cpu'},
    encode_kwargs={'normalize_embeddings' : True}
)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    # embeddings = OpenAIEmbeddings()
    # embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=LAPTOP_EMBEDDINGS)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.7-flash",
        temperature=0
    )

    retriever = vector_store.as_retriever(search_kwargs={"k":3})

    prompt = ChatPromptTemplate.from_template("""
    You are a document question-answering assistant.

    Use ONLY the information present in the context below.
    Do NOT use your own knowledge, memory, or external information.
    If the answer is not explicitly stated in the context, respond exactly with:

    "I could not find the answer in the uploaded documents."
    Context:
    {context}
    
    Question:
    {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context" : retriever | format_docs,
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
    

def main():
    load_dotenv()
    st.set_page_config(
        page_title="Chat with multiple PDFs", 
        page_icon=":books:"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question :
        handle_userinput(user_question)
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and press click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                #get the text chunks
                text_chunks = get_text_chunks(raw_text)

                #create vector store(with embeddings)
                vector_store = get_vector_store(text_chunks)
                
                #conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.session_state.chat_history = []
                st.success("Processing complete! You can now start chatting")
        
    
if __name__ == '__main__':
    main()
