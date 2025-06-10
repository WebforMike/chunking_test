import hnswlib
import numpy as np
import pandas as pd
import re
from datetime import datetime
import vertexai
from vertexai.language_models import TextEmbeddingModel
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_vertexai import VertexAIEmbeddings

# -------------------------------
# CONFIGURATION
# -------------------------------
PROJECT_ID = "project-id-here"
LOCATION = "us-central1"
USE_VERTEX_EMBEDDINGS = True
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# -------------------------------
# EMBEDDING SETUP
# -------------------------------
if USE_VERTEX_EMBEDDINGS:
    print("Initializing Vertex AI...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

def real_embed(texts):
    print(f"Generating {len(texts)} embeddings from Vertex AI...")
    return np.array([model.get_embeddings([t])[0].values for t in texts], dtype='float32')

embed_function = real_embed

# -------------------------------
# CHUNKING STRATEGY DEFINITIONS
# -------------------------------

def html_chunk(text):
    print("HTML-aware chunking...")
    text = text.replace('\n', '').replace('<br>', '').replace('</br>', '')
    soup = BeautifulSoup(text, "html.parser")
    chunks, buffer = [], ""
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'div']):
        segment = tag.get_text(strip=True)
        if not segment or len(segment.split()) < 3:
            continue
        if len(buffer.split()) + len(segment.split()) < 100:
            buffer += ' ' + segment
        else:
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = segment
    if buffer.strip():
        chunks.append(buffer.strip())
    print(f"→ {len(chunks)} HTML-based chunks created.")
    return chunks

def token_chunk(text, chunk_size=500, overlap=100):
    print("Token-based chunking...")
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) > 5:
            chunks.append(chunk)
    print(f"→ {len(chunks)} token-based chunks created.")
    return chunks

def recursive_chunk(text):
    print("LangChain RecursiveCharacterTextSplitter chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    print(f"→ {len(chunks)} recursive chunks created.")
    return chunks

def semantic_chunk(text):
    print("Semantic chunking via LangChain SemanticChunker...")
    embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    chunks = chunker.split_text(text)
    print(f"→ {len(chunks)} semantic chunks created.")
    return chunks

# -------------------------------
# LOAD DOCUMENTS AND RUN MULTI-CHUNKING
# -------------------------------
input_file = "input_documents.csv"
print(f"\n📄 Reading documents from: {input_file}")
df = pd.read_csv(input_file)

indexes = {}
chunk_maps = {}
chunk_records = []

# Enable or disable chunkers here
enabled_chunkers = ["html", "tokens", "recursive", "semantic"]
chunking_methods = [method for method in enabled_chunkers]

for method in chunking_methods:
    print(f"\n🧪 Testing with chunking method: {method.upper()}\n")
    chunk_function = {
        "html": html_chunk,
        "tokens": token_chunk,
        "recursive": recursive_chunk,
        "semantic": semantic_chunk
    }[method]

    for _, row in df.iterrows():
        variant = row["variant"]
        text = row["text"]
        variant_id = f"{variant}__{method}"

        print(f"\n--- Processing Variant {variant} with {method} chunking ---")

        chunks = chunk_function(text)

        if not chunks:
            print(f"Skipping {variant_id}: no valid chunks.")
            continue

        embeddings = embed_function(chunks)

        if embeddings.shape[0] == 0:
            print(f"Skipping {variant_id}: no embeddings returned.")
            continue

        dim = embeddings.shape[1]
        print(f"Indexing {len(embeddings)} embeddings (dim={dim})...")

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

        print(f"Finished indexing Variant {variant_id}")

# -------------------------------
# LOAD QUERIES
# -------------------------------
query_file = "queries.txt"
print(f"\nLoading queries from: {query_file}")
with open(query_file, "r", encoding="utf-8") as f:
    query_list = [line.strip() for line in f if line.strip()]

# -------------------------------
# QUERY & RETRIEVE
# -------------------------------
results = []

for query in query_list:
    print(f"\n🔍 Query: '{query}'")
    query_embedding = embed_function([query])[0].reshape(1, -1)

    method_buckets = {method: [] for method in chunking_methods}

    for variant_id, index in indexes.items():
        num_elements = index.get_current_count()
        k = min(3, num_elements)
        if k > 0:
            labels, distances = index.knn_query(query_embedding, k=k)
            method = variant_id.split("__")[1]
            for label, distance in zip(labels[0], distances[0]):
                method_buckets[method].append({
                    "query": query,
                    "variant_id": variant_id,
                    "variant": variant_id.split("__")[0],
                    "chunking_method": method,
                    "similarity": round(1 - distance, 3),
                    "chunk_index": label,
                    "chunk_text": chunk_maps[variant_id][label]
                })
        else:
            print(f"Skipping variant {variant_id}: no chunks indexed.")

    for method, results_list in method_buckets.items():
        results_list.sort(key=lambda r: r["similarity"], reverse=True)
        for i, r in enumerate(results_list, 1):
            r["rank"] = i
        results.extend(results_list)

# -------------------------------
# SAVE OUTPUTS
# -------------------------------
retrieval_file = f"retrieval_results_{timestamp}.csv"
chunk_file = f"chunked_documents_{timestamp}.csv"

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["query", "chunking_method", "rank"])

results_df.to_csv(retrieval_file, index=False)
pd.DataFrame(chunk_records).to_csv(chunk_file, index=False)

print(f"\nDONE!")
print(f"🔎 Retrieval results saved to: {retrieval_file}")
print(f"📄 Full chunk map saved to:     {chunk_file}")
