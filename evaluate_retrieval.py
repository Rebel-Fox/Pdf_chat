import json
import time
import numpy as np
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

#embedding model
print("Loading Bi-Encoder ('BAAI/bge-small-en-v1.5')...")
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5",
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings":True}
)

#cross-encoder
print("Loading Cross-Encoder ('cross-encoder/ms-marco-MiniLM-L-6-v2')...")
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            full_text += content + "\n"
    return full_text

def score_hits_and_ranks(retrieved_docs, target_keywords):
    #Checks the top candidate docs against expected ground-truth keywords
    found_rank = None
    for rank_idx, doc in enumerate(retrieved_docs, start=1):
        doc_text = doc.page_content.lower()
        matches = sum(1 for kw in target_keywords if kw in doc_text)
        if matches >= max(1, len(target_keywords) // 2):
            found_rank = rank_idx
            break

    return found_rank

def run_standard_experiment(vectorstore, benchmark_data, top_k=3):
    #Single-stage standard FAISS retrieval
    retriever = vectorstore.as_retriever(search_kwargs={"k":top_k})

    hits =0
    reciprocal_ranks = []
    latencies = []

    for item in benchmark_data:
        query = item["question"]
        target_keywords = [kw.lower() for kw in item["expected_keywords"]]

        start_time = time.time()
        retrieved_docs = retriever.invoke(query)
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)

        found_rank = score_hits_and_ranks(retrieved_docs, target_keywords)

        if found_rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

    hit_rate = (hits / len(benchmark_data)) * 100
    mrr = float(np.mean(reciprocal_ranks))
    avg_latency = float(np.mean(latencies))
    
    return hit_rate, mrr, avg_latency

def run_reranked_experiment(vectorstore, benchmark_data, initial_k=10, final_top_k=3):

    retriever = vectorstore.as_retriever(search_kwargs={"k":initial_k})

    hits =0
    reciprocal_ranks = []
    latencies = []

    for item in benchmark_data:
        query = item["question"]
        target_keywords = [kw.lower() for kw in item["expected_keywords"]]

        start_time = time.time()
        #Stage 1: Candidate fetch
        candidate_docs = retriever.invoke(query)

        #Stage 2: Cross-Encoder joint scoring
        pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = RERANKER.predict(pairs)

        #Sort candidate docs by cross-encoder score descending
        ranked_docs = [
            doc for _, doc in sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
        ][:final_top_k]

        latency = (time.time() - start_time) * 1000
        latencies.append(latency)

        found_rank = score_hits_and_ranks(ranked_docs, target_keywords)
        
        if found_rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

    hit_rate = (hits / len(benchmark_data)) * 100
    mrr = float(np.mean(reciprocal_ranks))
    avg_latency = float(np.mean(latencies))

    return hit_rate, mrr, avg_latency

if __name__ == "__main__":
    PDF_PATH = "Nvidia_report.pdf"
    BENCHMARK_PATH = "benchmark_dataset.json"

    raw_text = extract_pdf_text(PDF_PATH)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)


    #1. Baseline Character Splitter
    char_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks_base = char_splitter.split_text(raw_text)
    vstore_base = FAISS.from_texts(texts=chunks_base, embedding=EMBEDDINGS)

    #2. Optimal Recursive Splitter
    rec_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120, separators=["\n\n", "\n", ".", " "])
    chunks_rec = rec_splitter.split_text(raw_text)
    vstore_rec = FAISS.from_texts(texts=chunks_rec, embedding=EMBEDDINGS)


    print("\n" + "=" * 80)
    print(f"{'STRATEGY':<45} | {'CHUNKS':<7} | {'HIT RATE@3':<11} | {'MRR':<8} | {'LATENCY':<8}")
    print("=" * 80)

    # Experiment 1: Baseline
    hr, mrr, lat = run_standard_experiment(vstore_base, benchmark_data, top_k=3)
    print(f"{'1. Baseline (Character 1000/200)':<48} | {len(chunks_base):<7} | {hr:>9.2f}% | {mrr:>8.4f} | {lat:>6.1f} ms")

    # Experiment 2: Recursive (600/120)
    hr, mrr, lat = run_standard_experiment(vstore_rec, benchmark_data, top_k=3)
    print(f"{'2. Recursive (600/120) - FAISS Only':<48} | {len(chunks_rec):<7} | {hr:>9.2f}% | {mrr:>8.4f} | {lat:>6.1f} ms")

    # Experiment 3: Recursive (600/120) + Cross-Encoder Re-ranker
    hr, mrr, lat = run_reranked_experiment(vstore_rec, benchmark_data, initial_k=10, final_top_k=3)
    print(f"{'3. Recursive (600/120) + Cross-Encoder Re-rank':<48} | {len(chunks_rec):<7} | {hr:>9.2f}% | {mrr:>8.4f} | {lat:>6.1f} ms")
    
    print("=" * 80)