# Chat with Multiple PDFs 📚

A Streamlit app that lets you upload PDF documents and ask questions about them.
Documents are split into chunks, embedded locally with a lightweight BGE model,
stored in a FAISS vector store, and answered by Google's Gemini LLM using
retrieval-augmented generation (RAG).

## Features

- Upload multiple PDFs at once via the sidebar
- Local embeddings (BAAI/bge-small-en-v1.5) — no embedding API costs
- FAISS vector search over document chunks (top-3 retrieval)
- Gemini-powered answers restricted strictly to your document content
- Chat-style history within a session

## Project Structure

```
pdf_chat/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Template for environment variables
├── .gitignore          # Files excluded from version control
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- A Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/pdf_chat.git
   cd pdf_chat
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   # .venv\Scripts\activate       # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create your `.env` file from the template and add your API key:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env`:

   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. Upload one or more PDFs in the sidebar
2. Click **Process** to extract text, chunk it, and build the vector store
3. Type a question in the main panel and get answers based on your documents

If the answer isn't in the uploaded documents, the bot will respond with:
*"I could not find the answer in the uploaded documents."*

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| PDF parsing | PyPDF2 |
| Text splitting | LangChain CharacterTextSplitter |
| Embeddings | BAAI/bge-small-en-v1.5 (HuggingFace, CPU) |
| Vector store | FAISS |
| LLM | Google Gemini (ChatGoogleGenerativeAI) |
| Orchestration | LangChain LCEL |

## Notes

- The first run downloads the BGE embedding model (~130 MB), so processing
  may take a moment initially.
- Keep your `.env` file private — it is already listed in `.gitignore`.
