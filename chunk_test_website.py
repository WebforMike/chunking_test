import hnswlib
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import vertexai
from vertexai.language_models import TextEmbeddingModel
from bs4 import BeautifulSoup
import tiktoken
import requests # Import the requests library for fetching URLs
from urllib.parse import urlparse # For basic URL validation
from google.auth.exceptions import RefreshError # Import specific auth error
from google.api_core.exceptions import GoogleAPIError # Import general API error for Vertex AI

# Try to import google.colab.auth for Colab-specific authentication
COLAB_ENVIRONMENT = False
try:
    from google.colab import auth
    COLAB_ENVIRONMENT = True
except ImportError:
    pass # Not in Colab environment

# Import LangChain components, handling potential import errors
TokenTextSplitter = None
SemanticChunker = None
VertexAIEmbeddings = None
try:
    from langchain_text_splitters import TokenTextSplitter
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_google_vertexai import VertexAIEmbeddings
except ImportError as e:
    logging.error(f"Failed to import LangChain components. Please ensure you have run "
                  f"'!pip install langchain_experimental langchain_google_vertexai' "
                  f"and restarted your Colab runtime if applicable. Error: {e}")
except Exception as e:
    logging.error(f"An unexpected error occurred during LangChain imports: {e}")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------
# CONFIGURATION
# -------------------------------
class Config:
    """Configuration class for the Website Chunk Analyzer."""
    PROJECT_ID: str = "marketingdata-393009" # Ensure this Project ID has access to Vertex AI APIs
    LOCATION: str = "us-central1"
    USE_VERTEX_EMBEDDINGS: bool = True
    EMBEDDING_MODEL_NAME: str = "gemini-embedding-001" # Consistent model for all embeddings
    SEMANTIC_CHUNKER_MODEL_NAME: str = "text-embedding-004" # LangChain SemanticChunker often uses 004/005
    HTML_MIN_WORDS_PER_SEGMENT: int = 3
    TOKEN_CHUNK_SIZE: int = 100
    TOKEN_OVERLAP: int = 20
    SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT: int = 95 # Percentile for semantic chunking
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_M: int = 16
    KNN_K: int = 3 # Number of nearest neighbors to retrieve
    DEFAULT_EMBEDDING_DIMENSION: int = 768 # Default dimension for gemini-embedding-001 and text-embedding-004/005

    # Input/Output settings (now primarily for queries and output files)
    QUERIES_FILE: str = "queries.txt"
    OUTPUT_RETRIEVAL_PREFIX: str = "retrieval_results"
    OUTPUT_CHUNKS_PREFIX: str = "chunked_documents"

# -------------------------------
# TOKENIZER SETUP
# -------------------------------
# Initialize tiktoken encoder globally as it's static
try:
    ENCODER = tiktoken.get_encoding("cl100k_base")  # Proxy for Gemini tokenization
except Exception as e:
    logger.error(f"Failed to load tiktoken encoder: {e}")
    # Handle this gracefully, perhaps by falling back to a character count or raising a critical error
    raise SystemExit("Cannot proceed without a valid tokenizer.")

def count_tokens(text: str) -> int:
    """Counts the number of tokens in a given text using the global encoder."""
    return len(ENCODER.encode(text))

# -------------------------------
# MAIN ANALYZER CLASS
# -------------------------------
class WebsiteChunkAnalyzer:
    """
    A class to analyze website chunks using various chunking strategies,
    generate embeddings, and perform similarity-based retrieval.
    """
    def __init__(self, config: Config):
        """
        Initializes the WebsiteChunkAnalyzer with the given configuration.

        Args:
            config: An instance of the Config class.
        """
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.embed_model = None
        self.semantic_embeddings_model = None
        self._initialize_vertex_ai()

        self.indexes: dict[str, hnswlib.Index] = {}
        self.chunk_maps: dict[str, list[str]] = {}
        self.chunk_records: list[dict] = []
        self.chunk_funcs = {
            "html": self._html_chunk,
            "tokens": self._token_chunk,
            "recursive": self._recursive_chunk,
            "semantic": self._semantic_chunk
        }

    def _initialize_vertex_ai(self) -> None:
        """
        Initializes Vertex AI and loads embedding models.
        Handles authentication specifically for Google Colab environments.
        """
        if self.config.USE_VERTEX_EMBEDDINGS:
            logger.info("🔑 Initializing Vertex AI...")
            # Check if VertexAIEmbeddings class was successfully imported
            if VertexAIEmbeddings is None:
                logger.error("VertexAIEmbeddings class not available. Skipping Vertex AI model initialization.")
                return

            try:
                # Authenticate for Colab if running in that environment
                if COLAB_ENVIRONMENT:
                    logger.info("Detected Google Colab environment. Authenticating...")
                    auth.authenticate_user()
                    logger.info("Colab authentication successful.")
                else:
                    logger.info("Not in Colab. Ensure 'gcloud auth application-default login' has been run.")

                vertexai.init(project=self.config.PROJECT_ID, location=self.config.LOCATION)
                self.embed_model = TextEmbeddingModel.from_pretrained(self.config.EMBEDDING_MODEL_NAME)
                # LangChain's SemanticChunker uses a different model often,
                # ensuring consistency or allowing for specific model choice here.
                self.semantic_embeddings_model = VertexAIEmbeddings(model_name=self.config.SEMANTIC_CHUNKER_MODEL_NAME)
                logger.info("Vertex AI initialized and embedding models loaded.")
            except (RefreshError, GoogleAPIError) as e:
                logger.error(f"Authentication/API Error: Failed to initialize Vertex AI. Please ensure: "
                             f"1. You are authenticated to Google Cloud (e.g., 'gcloud auth application-default login' or 'google.colab.auth.authenticate_user()').\n"
                             f"2. Project ID '{self.config.PROJECT_ID}' is correct and has the Vertex AI API enabled.\n"
                             f"3. Your authenticated account has the necessary IAM permissions (e.g., 'Vertex AI User').\n"
                             f"Error details: {e}")
                self.embed_model = None
                self.semantic_embeddings_model = None
                # If embedding models cannot be loaded, subsequent operations will be severely impacted.
                # Consider raising a SystemExit or similar if this is a hard requirement.
            except Exception as e:
                logger.error(f"An unexpected error occurred during Vertex AI initialization: {e}")
                self.embed_model = None
                self.semantic_embeddings_model = None
        else:
            logger.info("Skipping Vertex AI initialization as USE_VERTEX_EMBEDDINGS is False.")

    def _real_embed(self, texts: list[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts using the initialized Vertex AI model.
        This function is updated to send texts one by one to avoid batch size issues.

        Args:
            texts: A list of strings to embed.

        Returns:
            A numpy array of embeddings (float32).
        """
        if self.embed_model is None:
            logger.error("Embedding model not initialized. Cannot generate embeddings. Returning an empty array.")
            # Return an empty 2D array with 0 rows and a default dimension
            return np.zeros((0, self.config.DEFAULT_EMBEDDING_DIMENSION), dtype='float32')

        logger.info(f"🧠 Generating {len(texts)} embeddings from Vertex AI using {self.config.EMBEDDING_MODEL_NAME}...")
        embeddings_list = []
        try:
            # Iterate through each text and get its embedding individually
            for i, text in enumerate(texts):
                # model.get_embeddings expects a list, even for a single text, but it's a list of instances, not a batch
                # The error "batchSize must be 1" means the overall request can only contain 1 text.
                # So we send each text in its own request.
                single_embedding = self.embed_model.get_embeddings([text])[0].values
                embeddings_list.append(single_embedding)
                if (i + 1) % 10 == 0: # Log progress every 10 embeddings
                    logger.info(f"  Processed {i + 1}/{len(texts)} embeddings.")

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}. Returning an empty array.")
            # If embedding generation fails, return an empty 2D array with the expected dimension if available,
            # otherwise a default. This keeps the shape consistent for downstream operations.
            if self.embed_model and hasattr(self.embed_model, 'embedding_dimensions'):
                return np.zeros((0, self.embed_model.embedding_dimensions), dtype='float32')
            else:
                return np.zeros((0, self.config.DEFAULT_EMBEDDING_DIMENSION), dtype='float32') # Fallback
        return np.array(embeddings_list, dtype='float32')

    def _split_html_blocks(self, html: str) -> list[str]:
        """
        Splits HTML into plain-text segments at common block tags.

        Args:
            html: The HTML content as a string.

        Returns:
            A list of plain-text segments.
        """
        soup = BeautifulSoup(html, "html.parser")
        segments = []
        # Extended list of common block-level HTML tags for better coverage
        block_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'div', 'blockquote',
                      'article', 'section', 'aside', 'pre', 'code', 'table']
        for tag in soup.find_all(block_tags):
            text = tag.get_text(strip=True)
            # Filter out very short or empty segments
            if text and len(text.split()) >= self.config.HTML_MIN_WORDS_PER_SEGMENT:
                segments.append(text)
        logger.debug(f"HTML split into {len(segments)} block segments.")
        return segments

    # -------------------------------
    # CHUNKING STRATEGIES (Private Methods)
    # -------------------------------
    def _html_chunk(self, html: str) -> list[str]:
        """
        HTML-aware chunking strategy that combines HTML blocks into token-limited chunks.

        Args:
            html: The HTML content as a string.

        Returns:
            A list of text chunks.
        """
        logger.info("🧩 HTML-aware chunking (token-based)...")
        chunks, buffer = [], ""
        for segment in self._split_html_blocks(html):
            combined = (buffer + ' ' + segment).strip() if buffer else segment
            if count_tokens(combined) <= self.config.TOKEN_CHUNK_SIZE:
                buffer = combined
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = segment
        if buffer: # Append any remaining content in the buffer
            chunks.append(buffer)
        logger.info(f"→ {len(chunks)} HTML-based chunks created.")
        return chunks

    def _token_chunk(self, html: str) -> list[str]:
        """
        Token-based chunking strategy applied to HTML segments.

        Args:
            html: The HTML content as a string.

        Returns:
            A list of text chunks.
        """
        logger.info("🔹 Token-based chunking...")
        chunks = []
        for segment in self._split_html_blocks(html):
            tokens = ENCODER.encode(segment)
            step = self.config.TOKEN_CHUNK_SIZE - self.config.TOKEN_OVERLAP
            # Ensure step is at least 1 to avoid infinite loop for chunk_size <= overlap
            step = max(1, step)
            for i in range(0, len(tokens), step):
                window = tokens[i:i + self.config.TOKEN_CHUNK_SIZE]
                if len(window) > 5: # Small heuristic to avoid very tiny chunks
                    chunks.append(ENCODER.decode(window))
        logger.info(f"→ {len(chunks)} token-based chunks created.")
        return chunks

    def _recursive_chunk(self, html: str) -> list[str]:
        """
        LangChain TokenTextSplitter based recursive chunking applied to HTML segments.

        Args:
            html: The HTML content as a string.

        Returns:
            A list of text chunks.
        """
        logger.info("🔁 LangChain TokenTextSplitter chunking...")
        if TokenTextSplitter is None:
            logger.warning("TokenTextSplitter class not available. Falling back to HTML-aware chunking.")
            return self._html_chunk(html)

        chunks = []
        # Ensure TokenTextSplitter is initialized correctly
        splitter = TokenTextSplitter(
            chunk_size=self.config.TOKEN_CHUNK_SIZE,
            chunk_overlap=self.config.TOKEN_OVERLAP,
            encoding_name="cl100k_base"
        )
        for segment in self._split_html_blocks(html):
            # LangChain's splitter handles tokenization internally based on encoding_name
            chunks.extend(splitter.split_text(segment))
        logger.info(f"→ {len(chunks)} token-recursive chunks created.")
        return chunks

    def _semantic_chunk(self, html: str) -> list[str]:
        """
        Hybrid semantic + token chunking strategy. First applies semantic chunking,
        then token-splits any resulting chunks that are too large.

        Args:
            html: The HTML content as a string.

        Returns:
            A list of text chunks.
        """
        logger.info("🧠 Hybrid semantic + token chunking...")
        if SemanticChunker is None or self.semantic_embeddings_model is None:
            logger.error("SemanticChunker or semantic embedding model not initialized. Skipping semantic chunking.")
            # Fallback to a simpler chunking method or raise error
            return self._recursive_chunk(html) # Fallback to recursive chunking

        # Ensure SemanticChunker is initialized correctly
        chunker = SemanticChunker(
            self.semantic_embeddings_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=self.config.SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT
        )
        all_chunks = []
        token_splitter = TokenTextSplitter( # For re-chunking large semantic chunks
            chunk_size=self.config.TOKEN_CHUNK_SIZE,
            chunk_overlap=self.config.TOKEN_OVERLAP,
            encoding_name="cl100k_base"
        )

        for segment in self._split_html_blocks(html):
            try:
                # First, semantically split the segment
                semantically_split_segments = chunker.split_text(segment)
            except Exception as e:
                logger.warning(f"Failed to perform semantic chunking on a segment: {e}. Falling back to token split.")
                semantically_split_segments = [segment] # Treat the whole segment as one, then let token_splitter handle it

            for sem_chunk in semantically_split_segments:
                if not sem_chunk.strip(): # Skip empty semantic chunks
                    continue

                tokens_in_sem_chunk = count_tokens(sem_chunk)
                if tokens_in_sem_chunk > self.config.TOKEN_CHUNK_SIZE:
                    # If a semantically defined chunk is too large, apply token-based splitting
                    logger.debug(f"Semantic chunk too large ({tokens_in_sem_chunk} tokens), applying token splitter.")
                    all_chunks.extend(token_splitter.split_text(sem_chunk))
                else:
                    all_chunks.append(sem_chunk)

        logger.info(f"→ {len(all_chunks)} hybrid semantic/token chunks created.")
        return all_chunks

    def _create_hnsw_index(self, embeddings: np.ndarray) -> hnswlib.Index:
        """
        Creates and initializes an HNSWLib index with the given embeddings.

        Args:
            embeddings: A numpy array of embeddings.

        Returns:
            An initialized hnswlib.Index object.
        """
        # Ensure embeddings are not empty before getting dimension
        if embeddings.shape[0] == 0:
            logger.warning("Attempted to create HNSW index with empty embeddings. Returning None.")
            return None # Or raise an error, depending on desired behavior

        dim = embeddings.shape[1]
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=len(embeddings),
                         ef_construction=self.config.HNSW_EF_CONSTRUCTION,
                         M=self.config.HNSW_M)
        index.add_items(embeddings)
        return index

    def _fetch_html_from_url(self, url: str) -> str | None:
        """
        Fetches HTML content from a given URL.

        Args:
            url: The URL to fetch.

        Returns:
            The HTML content as a string, or None if fetching fails.
        """
        logger.info(f"🌐 Fetching HTML from: {url}")
        try:
            # Basic validation: check if it's a valid URL scheme
            parsed_url = urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                logger.error(f"Invalid URL format: {url}. Please provide a full URL including scheme (e.g., 'https://').")
                return None

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            logger.info(f"Successfully fetched HTML from {url}")
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching {url}: {e}")
            return None


    def load_and_chunk_documents(self, main_url: str, comparison_urls: list[str]) -> None:
        """
        Loads documents from provided URLs, applies various chunking strategies,
        generates embeddings, and builds HNSWLib indexes.

        Args:
            main_url: The URL of the main website to analyze.
            comparison_urls: A list of URLs for comparison websites.
        """
        documents_to_process = []

        # Process main URL
        main_html = self._fetch_html_from_url(main_url)
        if main_html:
            documents_to_process.append({"variant": "main_site", "text": main_html})
        else:
            logger.error(f"Skipping main URL {main_url} due to fetch failure.")

        # Process comparison URLs
        for i, url in enumerate(comparison_urls):
            comp_html = self._fetch_html_from_url(url)
            if comp_html:
                documents_to_process.append({"variant": f"comparison_site_{i+1}", "text": comp_html})
            else:
                logger.error(f"Skipping comparison URL {url} due to fetch failure.")

        if not documents_to_process:
            logger.warning("No valid URLs processed. Exiting document loading.")
            return

        # Adapt the existing processing loop to use the fetched HTML
        enabled_chunkers = ["html", "tokens", "recursive", "semantic"]

        for method in enabled_chunkers:
            logger.info(f"\n🧪 Testing with chunking method: {method.upper()}\n")
            chunk_function = self.chunk_funcs[method]

            for doc in documents_to_process:
                variant = doc["variant"]
                text = doc["text"]
                variant_id = f"{variant}__{method}"

                logger.info(f"\n--- Processing {variant_id} ---")
                chunks = chunk_function(text)
                if not chunks:
                    logger.warning(f"⚠️ Skipping {variant_id}: no valid chunks were created.")
                    continue

                embeddings = self._real_embed(chunks)
                if embeddings.shape[0] == 0:
                    logger.warning(f"⚠️ Skipping {variant_id}: no embeddings returned for {len(chunks)} chunks.")
                    continue

                index = None
                try:
                    index = self._create_hnsw_index(embeddings)
                    if index: # Only proceed if index creation was successful
                        self.indexes[variant_id] = index
                        self.chunk_maps[variant_id] = chunks

                        for i, chunk in enumerate(chunks):
                            self.chunk_records.append({
                                "variant": variant,
                                "chunking_method": method,
                                "variant_id": variant_id,
                                "chunk_index": i,
                                "chunk_text": chunk
                            })
                        logger.info(f"✅ Finished indexing {variant_id}")
                    else:
                        logger.warning(f"❌ Failed to create HNSW index for {variant_id} due to empty embeddings or other issue.")
                except Exception as e:
                    logger.error(f"Error indexing {variant_id} with HNSWLib: {e}")

    def retrieve_queries(self) -> list[dict]:
        """
        Loads queries, performs retrieval against the built indexes,
        and aggregates results.

        Returns:
            A list of dictionaries, each representing a retrieval result.
        """
        logger.info(f"\n📋 Loading queries from: {self.config.QUERIES_FILE}")
        query_list = []
        try:
            with open(self.config.QUERIES_FILE, "r", encoding="utf-8") as f:
                query_list = [line.strip() for line in f if line.strip()]
            if not query_list:
                logger.warning("Queries file is empty or contains no valid queries.")
                return []
        except FileNotFoundError:
            logger.error(f"Error: Queries file not found at {self.config.QUERIES_FILE}")
            return []
        except Exception as e:
            logger.error(f"Error reading queries file: {e}")
            return []

        results = []
        enabled_chunkers = list(self.chunk_funcs.keys()) # Use keys from initialized chunk_funcs

        for query in query_list:
            logger.info(f"\n🔍 Query: '{query}'")
            query_emb = self._real_embed([query])
            if query_emb.shape[0] == 0:
                logger.warning(f"Could not generate embedding for query: '{query}'. Skipping.")
                continue
            query_emb = query_emb[0].reshape(1, -1)

            buckets = {m: [] for m in enabled_chunkers}

            for vid, index in self.indexes.items():
                count = index.get_current_count()
                k = min(self.config.KNN_K, count)
                if k > 0:
                    try:
                        labels, distances = index.knn_query(query_emb, k=k)
                        method = vid.split("__")[1]
                        for lbl, dist in zip(labels[0], distances[0]):
                            buckets[method].append({
                                "query": query,
                                "variant_id": vid,
                                "variant": vid.split("__")[0],
                                "chunking_method": method,
                                "similarity": round(1 - dist, 3), # HNSWLib returns distance, convert to similarity
                                "chunk_index": lbl,
                                "chunk_text": self.chunk_maps[vid][lbl]
                            })
                    except Exception as e:
                        logger.error(f"Error during KNN query for {vid}: {e}")
                else:
                    logger.warning(f"⚠️ Skipping retrieval for {vid}: no chunks indexed.")

            for method, res_list in buckets.items():
                res_list.sort(key=lambda r: r["similarity"], reverse=True)
                for rank, r in enumerate(res_list, 1):
                    r["rank"] = rank
                results.extend(res_list)
        return results

    def save_outputs(self, retrieval_results: list[dict], chunk_records: list[dict]) -> None:
        """
        Saves the retrieval results and chunk records to CSV files.

        Args:
            retrieval_results: List of retrieval result dictionaries.
            chunk_records: List of chunk record dictionaries.
        """
        retrieval_file = f"{self.config.OUTPUT_RETRIEVAL_PREFIX}_{self.timestamp}.csv"
        chunk_file = f"{self.config.OUTPUT_CHUNKS_PREFIX}_{self.timestamp}.csv"

        try:
            if retrieval_results: # Use the passed argument directly
                pd.DataFrame(retrieval_results).sort_values(
                    ["query", "chunking_method", "rank"]
                ).to_csv(retrieval_file, index=False)
                logger.info(f"🔎 Retrieval results saved to: {retrieval_file}")
            else:
                logger.warning("No retrieval results to save.")

            if chunk_records: # Use the passed argument directly
                pd.DataFrame(chunk_records).to_csv(chunk_file, index=False)
                logger.info(f"📄 Full chunk map saved to: {chunk_file}")
            else:
                logger.warning("No chunk records to save.")

        except Exception as e:
            logger.error(f"Error saving outputs: {e}")

# -------------------------------
# EXECUTION
# -------------------------------
if __name__ == "__main__":
    app_config = Config()
    analyzer = WebsiteChunkAnalyzer(app_config)

    # 1. Get user input for URLs
    main_url_input = input("Enter the main URL to analyze (e.g., https://www.example.com): ").strip()
    comparison_urls_input = input("Enter comparison URLs (comma-separated, or leave blank): ").strip()

    comparison_urls = [url.strip() for url in comparison_urls_input.split(',') if url.strip()]

    # 2. Load Documents and Chunk them from URLs
    analyzer.load_and_chunk_documents(main_url_input, comparison_urls)

    # 3. Retrieve Queries (still from queries.txt)
    retrieval_results = analyzer.retrieve_queries()

    # 4. Save Outputs
    analyzer.save_outputs(retrieval_results, analyzer.chunk_records)

    logger.info("\n✅ DONE!")
    logger.info("Please check the generated CSV files for results.")

