Website Chunk Analyzer and Retrieval

This project provides a Python script to analyze website content by applying various text chunking strategies, generating embeddings using Google's Vertex AI, and performing similarity-based retrieval against those chunks. It's designed to help understand how different chunking methods impact content representation and search effectiveness.

Features

HTML Block Splitting: Extracts plain-text segments from HTML based on common block-level tags (e.g., h1, p, div).

Multiple Chunking Strategies:

HTML-aware Chunking: Combines logical HTML blocks into token-limited chunks.

Token-based Chunking: Splits text into fixed-size token chunks with overlap.

Recursive Chunking (LangChain): Uses LangChain's TokenTextSplitter for recursive character splitting.

Hybrid Semantic + Token Chunking (LangChain): Employs LangChain's SemanticChunker to find semantically meaningful breaks, then falls back to token-based splitting if chunks are still too large.

Vertex AI Embeddings: Generates vector embeddings for all chunks using specified Vertex AI embedding models (gemini-embedding-001, text-embedding-004).

HNSWLib Indexing: Creates a highly efficient similarity search index (Hierarchical Navigable Small World) for fast retrieval of relevant chunks.

URL Input: Fetches HTML content directly from a main URL and multiple comparison URLs provided by the user.

Query-based Retrieval: Performs similarity searches using a list of queries from a local file.

CSV Output: Saves detailed chunk information and retrieval results to CSV files for further analysis.

Prerequisites
Before running the script, ensure you have the following:

Python 3.9+

Google Cloud Project: A Google Cloud Project with the Vertex AI API enabled.

Google Cloud SDK: Installed and authenticated on your local machine or Google Colab environment.

For Local Development: Run gcloud auth application-default login in your terminal.

For Google Colab: You will use google.colab.auth.authenticate_user() within the notebook.

IAM Permissions: Your authenticated Google Cloud account must have the Vertex AI User (or a broader Vertex AI Administrator) role on your specified Google Cloud Project (marketingdata-393009).

Installation
If you are running this in a Google Colab environment, execute the following commands in separate cells at the very beginning of your notebook:

!pip install hnswlib
!pip install langchain_experimental
!pip install langchain_google_vertexai
!pip install tiktoken
!pip install beautifulsoup4
!pip install requests

After installing these libraries, it is crucial to restart your Colab runtime (Runtime > Restart session from the menu).

Configuration
The Config class at the beginning of the script holds important parameters:

class Config:
    PROJECT_ID: str = "marketingdata-393009" # Your Google Cloud Project ID
    LOCATION: str = "us-central1" # Region for Vertex AI
    USE_VERTEX_EMBEDDINGS: bool = True
    EMBEDDING_MODEL_NAME: str = "gemini-embedding-001"
    SEMANTIC_CHUNKER_MODEL_NAME: str = "text-embedding-004"
    HTML_MIN_WORDS_PER_SEGMENT: int = 3
    TOKEN_CHUNK_SIZE: int = 100
    TOKEN_OVERLAP: int = 20
    SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT: int = 95
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_M: int = 16
    KNN_K: int = 3
    DEFAULT_EMBEDDING_DIMENSION: int = 768

    QUERIES_FILE: str = "queries.txt" # Path to your queries file
    OUTPUT_RETRIEVAL_PREFIX: str = "retrieval_results"
    OUTPUT_CHUNKS_PREFIX: str = "chunked_documents"

Adjust PROJECT_ID and other parameters as needed.

Usage
Create queries.txt: Create a file named queries.txt in the same directory as your script (or in your Colab environment). Each line in this file should be a query you want to use for retrieval.

Example queries.txt:

What is technical SEO?
How does core web vitals affect SEO?
What are the best practices for website speed?

Authenticate (Colab specific): If you are in Google Colab, run the following in a cell:

from google.colab import auth
auth.authenticate_user()

Follow the prompts to complete the authentication.

Run the script: Execute the entire Python script. The script will prompt you for:

Main URL: The primary website URL you want to analyze.

Comparison URLs: A comma-separated list of additional website URLs for comparison (optional).

Example interaction:

Enter the main URL to analyze (e.g., https://www.example.com): https://webfor.com/blog/an-introduction-to-technical-seo/
Enter comparison URLs (comma-separated, or leave blank): https://ahrefs.com/blog/technical-seo/, https://moz.com/blog/technical-seo-guide

Output
The script will generate two CSV files in the same directory:

retrieval_results_YYYYMMDD_HHMM.csv: Contains the results of the similarity retrieval for each query across all chunking methods and URLs. Columns include query, variant_id, variant, chunking_method, similarity, chunk_index, and chunk_text.

chunked_documents_YYYYMMDD_HHMM.csv: Contains all the generated chunks from each URL and for each chunking method. Columns include variant, chunking_method, variant_id, chunk_index, and chunk_text.

The YYYYMMDD_HHMM timestamp ensures that each run generates unique output files.

Troubleshooting

ModuleNotFoundError: Ensure you have run all !pip install commands and restarted your Colab runtime if applicable.

RefreshError / Vertex AI Initialization Errors:

Verify your PROJECT_ID in the Config class is correct.

Confirm the Vertex AI API is enabled for your Google Cloud Project.

Ensure your authenticated Google account has the necessary IAM permissions (e.g., Vertex AI User role) on that project.

If in Colab, ensure auth.authenticate_user() was run successfully.

"batchSize must be 1" Error: This indicates that the Vertex AI embedding model requires single text inputs per API call. The script's _real_embed function is designed to handle this by sending individual requests. If this error persists, it might indicate an unexpected model behavior or a temporary API issue.

FileNotFoundError: queries.txt: Create a queries.txt file in the same location where your Python script is executed, and populate it with your queries (one per line).

Empty Output Files / "No valid URLs processed": Check your input URLs for correct format (e.g., https://) and ensure they are accessible. Network issues or restrictive website policies (e.g., blocking automated requests) can prevent HTML fetching.
