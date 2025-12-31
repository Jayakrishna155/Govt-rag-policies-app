# Technical Details: RAG Flow & Architecture

## 🔄 RAG Flow Explanation

### Complete Question Answering Pipeline

1. **Document Upload & Processing**
   ```
   PDF Upload → Text Extraction → Hash Generation → Duplicate Check
   → Text Chunking → Embedding Generation → Vector Storage
   ```

2. **Question Answering Flow**
   ```
   User Question → Query Embedding → Vector Search (FAISS)
   → Context Retrieval → LLM Prompting (Groq) → Answer Generation
   ```

## 📊 Detailed Component Flow

### 1. PDF Upload Flow

**Step-by-Step:**

1. **File Reception** (`main.py` → `/upload-pdf`)
   - Receives PDF file via multipart form data
   - Validates file type (must be `.pdf`)

2. **Text Extraction** (`rag/pdf_loader.py`)
   - Uses `pypdf` to extract text from all pages
   - Generates SHA256 hash of PDF bytes for duplicate detection
   - Returns text with page numbers

3. **Duplicate Detection** (`rag/duplicate_handler.py`)
   - Checks if document hash exists in `uploaded_docs.json`
   - If duplicate: returns early, skips processing
   - If new: proceeds to processing

4. **Text Chunking** (`rag/text_splitter.py`)
   - Uses LangChain `RecursiveCharacterTextSplitter`
   - Chunk size: 1000 characters
   - Overlap: 200 characters
   - Preserves metadata: filename, page number

5. **Embedding Generation** (`rag/embeddings.py`)
   - Uses Hugging Face `sentence-transformers/all-MiniLM-L6-v2`
   - Model dimension: 384
   - Generates embeddings for all chunks
   - Returns numpy array of shape `[num_chunks, 384]`

6. **Vector Storage** (`rag/vector_store.py`)
   - Creates/updates FAISS index
   - Stores embeddings with L2 distance metric
   - Saves metadata (text, page, filename) alongside vectors
   - Persists to disk: `data/faiss_index/index.faiss`
   - Metadata saved to: `data/faiss_metadata.json`

### 2. Question Answering Flow

**Step-by-Step:**

1. **Question Reception** (`main.py` → `/ask`)
   - Receives question as JSON
   - Validates question is not empty

2. **Query Embedding** (`rag/embeddings.py`)
   - Generates embedding for user question
   - Same model as document embeddings (384 dimensions)

3. **Vector Search** (`rag/vector_store.py`)
   - Searches FAISS index for top-k similar vectors (k=4)
   - Uses L2 distance (Euclidean)
   - Returns chunks with lowest distance scores

4. **Context Building** (`rag/qa_chain.py`)
   - Combines retrieved chunks into context string
   - Formats with page numbers: `[Page X]: chunk_text`
   - Collects source citations

5. **LLM Prompting** (`rag/qa_chain.py`)
   - Creates prompt with:
     - System message: Instructions to answer only from context
     - Context: Retrieved document chunks
     - Question: User's question
   - Sends to Groq LLM (Llama 3.1 70B)
   - Temperature: 0.1 (low for consistency)

6. **Answer Generation**
   - LLM generates answer from context only
   - Validates answer is not generic
   - Returns answer with source citations

## 🔐 Duplicate Handling Mechanism

### SHA256 Hashing

**Why SHA256?**
- Cryptographic hash function
- Deterministic: same input → same hash
- Collision-resistant: different files → different hashes
- Fast computation

**Implementation:**

```python
# In pdf_loader.py
document_hash = hashlib.sha256(pdf_bytes).hexdigest()
```

**Process:**

1. **Hash Generation**
   - Hash is computed from raw PDF bytes
   - Before any text extraction
   - Ensures identical files have same hash

2. **Hash Storage**
   - Stored in `data/uploaded_docs.json`
   - Format:
     ```json
     {
       "hashes": ["abc123...", "def456..."],
       "count": 2
     }
     ```

3. **Duplicate Check**
   - Before processing, check if hash exists
   - If exists: return duplicate message, skip processing
   - If new: process and add hash to storage

4. **Benefits**
   - Prevents re-processing same documents
   - Saves computation time
   - Prevents duplicate vectors in database
   - Works even if filename changes

## 🗄️ Vector Database Architecture

### FAISS Index Structure

**Index Type:** `IndexFlatL2`
- L2 (Euclidean) distance metric
- Exact search (no approximation)
- Suitable for small to medium datasets

**Storage:**
- **Index file:** `data/faiss_index/index.faiss`
- **Metadata file:** `data/faiss_metadata.json`
- Both persist across server restarts

**Metadata Structure:**
```json
[
  {
    "text": "chunk text here...",
    "page": 3,
    "filename": "policy.pdf"
  },
  ...
]
```

**Vector-Metadata Mapping:**
- Vector at index `i` corresponds to metadata at index `i`
- Maintained during search and retrieval

### Incremental Updates

**How it works:**
1. Load existing index on startup
2. Add new vectors to existing index
3. Append metadata to metadata list
4. Save both index and metadata

**Benefits:**
- No need to rebuild entire index
- Fast incremental additions
- Persistent storage

## 🧠 Embedding Model Details

### Model: `all-MiniLM-L6-v2`

**Specifications:**
- **Provider:** Hugging Face / Sentence Transformers
- **Dimension:** 384
- **Type:** Sentence embeddings
- **Size:** ~80MB (downloads on first use)

**Why this model?**
- Fast inference
- Good quality for semantic search
- Small model size
- Well-suited for document retrieval

**Usage:**
- Same model for both documents and queries
- Ensures embeddings are in same space
- Enables semantic similarity search

## 🔗 LLM Integration (Groq)

### Model: Llama 3.1 70B Versatile

**Configuration:**
- **Provider:** Groq (fast inference API)
- **Temperature:** 0.1 (low for consistency)
- **Context Window:** Large (handles multiple chunks)

**Prompt Engineering:**
- System message enforces context-only answers
- Clear instructions to return "Answer not found" if not in context
- Includes page numbers in context for citation

**Error Handling:**
- Validates API key presence
- Handles API errors gracefully
- Returns user-friendly error messages

## 📈 Performance Considerations

### Optimization Strategies

1. **Lazy Loading**
   - Embedding model loads on first use
   - FAISS index loads on startup
   - Reduces initial startup time

2. **Batch Processing**
   - Embeddings generated in batch for all chunks
   - More efficient than one-by-one

3. **Persistent Storage**
   - Index and metadata saved to disk
   - No need to rebuild on restart
   - Fast startup after initial setup

4. **Duplicate Prevention**
   - Hash check before processing
   - Saves embedding computation
   - Prevents index bloat

## 🧪 Testing the System

### Test Scenarios

1. **Upload Flow**
   - Upload a PDF → Check success message
   - Upload same PDF → Check duplicate detection
   - Upload different PDF → Check processing

2. **Question Answering**
   - Ask question in document → Verify answer with sources
   - Ask question not in document → Should return "Answer not found"
   - Ask vague question → Should handle gracefully

3. **Persistence**
   - Upload document → Restart server → Ask question
   - Should still have document in index

4. **Edge Cases**
   - Empty PDF → Should handle error
   - Corrupted PDF → Should handle error
   - Very large PDF → Should process (may take time)

## 🔧 Troubleshooting

### Common Issues

1. **Embedding Model Download**
   - First run downloads model (~80MB)
   - Requires internet connection
   - Cached for future runs

2. **FAISS Index Errors**
   - Ensure `faiss-cpu` is installed
   - Check disk space for index storage
   - Verify metadata file integrity

3. **Groq API Errors**
   - Verify API key in `.env`
   - Check API rate limits
   - Verify model name is correct

4. **Memory Issues**
   - Large PDFs may use significant memory
   - Consider chunking strategy for very large documents
   - Monitor memory usage during processing

