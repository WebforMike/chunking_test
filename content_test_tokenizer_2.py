import hnswlib
import numpy as np
import pandas as pd
from datetime import datetime
import vertexai
from vertexai.language_models import TextEmbeddingModel
from bs4 import BeautifulSoup
import tiktoken

from langchain_text_splitters import TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_vertexai import VertexAIEmbeddings

# -------------------------------
# CONFIGURATION
# -------------------------------
PROJECT_ID = "marketingdata-393009"
LOCATION = "us-central1"
USE_VERTEX_EMBEDDINGS = True
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# -------------------------------
# TOKENIZER SETUP
# -------------------------------
ENCODER = tiktoken.get_encoding("cl100k_base")  # proxy for Gemini

def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))

# -------------------------------
# EMBEDDING SETUP
# -------------------------------
if USE_VERTEX_EMBEDDINGS:
    print("🔑 Initializing Vertex AI...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

def real_embed(texts):
    print(f"🧠 Generating {len(texts)} embeddings from Vertex AI...")
    return np.array([model.get_embeddings([t])[0].values for t in texts], dtype='float32')

embed_function = real_embed

# -------------------------------
# HTML BLOCK SPLITTER
# -------------------------------
def split_html_blocks(html: str):
    """Split HTML into plain-text segments at block tags."""
    soup = BeautifulSoup(html, "html.parser")
    segments = []
    for tag in soup.find_all(['h1','h2','h3','h4','p','li','div']):
        text = tag.get_text(strip=True)
        if text and len(text.split()) >= 3:
            segments.append(text)
    return segments

# -------------------------------
# CHUNKING STRATEGIES
# -------------------------------
def html_chunk(html, max_tokens=100):
    print("🧩 HTML-aware chunking (token-based)...")
    chunks, buffer = [], ""
    for segment in split_html_blocks(html):
        combined = (buffer + ' ' + segment).strip()
        if count_tokens(combined) <= max_tokens:
            buffer = combined
        else:
            if buffer:
                chunks.append(buffer)
            buffer = segment
    if buffer:
        chunks.append(buffer)
    print(f"→ {len(chunks)} HTML-based chunks created.")
    return chunks

def token_chunk(html, chunk_size=100, overlap=20):
    print("🔹 Token-based chunking...")
    chunks = []
    for segment in split_html_blocks(html):
        tokens = ENCODER.encode(segment)
        step = chunk_size - overlap
        for i in range(0, len(tokens), step):
            window = tokens[i:i + chunk_size]
            if len(window) > 5:
                chunks.append(ENCODER.decode(window))
    print(f"→ {len(chunks)} token-based chunks created.")
    return chunks

def recursive_chunk(html, chunk_size=100, overlap=20):
    print("🔁 LangChain TokenTextSplitter chunking...")
    chunks = []
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        encoding_name="cl100k_base"
    )
    for segment in split_html_blocks(html):
        chunks.extend(splitter.split_text(segment))
    print(f"→ {len(chunks)} token-recursive chunks created.")
    return chunks

def semantic_chunk(html, max_tokens=100):
    print("🧠 Hybrid semantic + token chunking...")
    embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )

    chunks = []
    for segment in split_html_blocks(html):
        tokens = ENCODER.encode(segment)
        i = 0
        while i < len(tokens):
            window = tokens[i:i + max_tokens]
            window_text = ENCODER.decode(window)
            window_chunks = chunker.split_text(window_text)
            if len(window_chunks) > 1:
                first = window_chunks[0]
                first_tok_count = len(ENCODER.encode(first))
                chunks.append(first)
                i += first_tok_count
            else:
                chunks.append(window_text)
                i += len(window)
    print(f"→ {len(chunks)} hybrid semantic/token chunks created.")
    return chunks

# -------------------------------
# LOAD DOCUMENTS & CHUNK
# -------------------------------
input_file = "input_documents.csv"
print(f"\n📄 Reading documents from: {input_file}")
df = pd.read_csv(input_file)

indexes = {}
chunk_maps = {}
chunk_records = []

enabled_chunkers = ["html", "tokens", "recursive", "semantic"]
chunk_funcs = {
    "html": html_chunk,
    "tokens": token_chunk,
    "recursive": recursive_chunk,
    "semantic": semantic_chunk
}

for method in enabled_chunkers:
    print(f"\n🧪 Testing with chunking method: {method.upper()}\n")
    chunk_function = chunk_funcs[method]

    for _, row in df.iterrows():
        variant = row["variant"]
        text = row["text"]
        variant_id = f"{variant}__{method}"

        print(f"\n--- Processing {variant_id} ---")
        chunks = chunk_function(text)
        if not chunks:
            print(f"⚠️ Skipping {variant_id}: no valid chunks.")
            continue

        embeddings = embed_function(chunks)
        if embeddings.shape[0] == 0:
            print(f"⚠️ Skipping {variant_id}: no embeddings returned.")
            continue

        dim = embeddings.shape[1]
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
        index.add_items(embeddings)

        indexes[variant_id] = index
        chunk_maps[variant_id] = chunks

        for i, chunk in enumerate(chunks):
            chunk_records.append({
                "variant": variant,
                "chunking_method": method,
                "variant_id": variant_id,
                "chunk_index": i,
                "chunk_text": chunk
            })

        print(f"✅ Finished indexing {variant_id}")

# -------------------------------
# LOAD QUERIES & RETRIEVE
# -------------------------------
query_file = "queries.txt"
print(f"\n📋 Loading queries from: {query_file}")
with open(query_file, "r", encoding="utf-8") as f:
    query_list = [line.strip() for line in f if line.strip()]

results = []
for query in query_list:
    print(f"\n🔍 Query: '{query}'")
    query_emb = embed_function([query])[0].reshape(1, -1)
    buckets = {m: [] for m in enabled_chunkers}

    for vid, index in indexes.items():
        count = index.get_current_count()
        k = min(3, count)
        if k:
            labels, distances = index.knn_query(query_emb, k=k)
            method = vid.split("__")[1]
            for lbl, dist in zip(labels[0], distances[0]):
                buckets[method].append({
                    "query": query,
                    "variant_id": vid,
                    "variant": vid.split("__")[0],
                    "chunking_method": method,
                    "similarity": round(1 - dist, 3),
                    "chunk_index": lbl,
                    "chunk_text": chunk_maps[vid][lbl]
                })
        else:
            print(f"⚠️ Skipping {vid}: no chunks indexed.")

    for method, res_list in buckets.items():
        res_list.sort(key=lambda r: r["similarity"], reverse=True)
        for rank, r in enumerate(res_list, 1):
            r["rank"] = rank
        results.extend(res_list)

# -------------------------------
# SAVE OUTPUTS
# -------------------------------
retrieval_file = f"retrieval_results_{timestamp}.csv"
chunk_file     = f"chunked_documents_{timestamp}.csv"

pd.DataFrame(results).sort_values(
    ["query", "chunking_method", "rank"]
).to_csv(retrieval_file, index=False)

pd.DataFrame(chunk_records).to_csv(chunk_file, index=False)

print(f"\n✅ DONE!")
print(f"🔎 Retrieval results saved to: {retrieval_file}")
print(f"📄 Full chunk map saved to:     {chunk_file}")
