import uuid
import os
import io
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.config.settings import settings

# Try to import required libraries with fallback handling
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("Warning: Pinecone not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"Warning: SentenceTransformer not available: {e}")
    print("RAG service will use OpenAI embeddings only")

# Try to import OpenAI for embeddings (optional)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from app.core.ocr_service import ocr_service
from app.core.classifications import classification_manager


class RAGService:
    """Service for handling RAG (Retrieval Augmented Generation) operations using Pinecone"""
    
    def __init__(self):
        """Initialize the RAG service with Pinecone vector database and embedding model"""
        # Check if Pinecone is available
        if not PINECONE_AVAILABLE:
            raise Exception("Pinecone library not available. Please install: pip install pinecone")
        
        # Check if Pinecone API key is configured
        if not settings.PINECONE_API_KEY:
            raise Exception("Pinecone API key not configured. Please set PINECONE_API_KEY environment variable.")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        # Initialize embedding model
        # Priority: OpenAI embeddings (like notebook) > SentenceTransformer
        self.use_openai_embeddings = False
        self.embedding_dimension = 384
        self.embedding_model = None
        self.openai_client = None
        
        # Try OpenAI embeddings first (MUST match notebook: text-embedding-3-large with 3072 dimensions)
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                # Use text-embedding-3-large with 3072 dimensions to match the notebooks
                test_response = self.openai_client.embeddings.create(
                    model="text-embedding-3-large",
                    input="test",
                    dimensions=3072
                )
                self.embedding_dimension = len(test_response.data[0].embedding)  # 3072 for text-embedding-3-large
                self.use_openai_embeddings = True
                print(f"Using OpenAI embeddings (text-embedding-3-large) - dimension: {self.embedding_dimension}")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI embeddings: {e}")
                print("Falling back to SentenceTransformer")
        
        # Fallback to SentenceTransformer if OpenAI not available or failed
        if not self.use_openai_embeddings:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    # Try better semantic search model first
                    try:
                        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                        self.embedding_dimension = 384
                        print(f"Using SentenceTransformer (all-MiniLM-L6-v2) - dimension: {self.embedding_dimension}")
                    except Exception:
                        # Fallback to multilingual model
                        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                        self.embedding_dimension = 384
                        print(f"Using SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2) - dimension: {self.embedding_dimension}")
                except Exception as e:
                    print(f"Warning: Could not load embedding model: {e}")
                    self.embedding_model = None
            else:
                print("Warning: SentenceTransformer not available. RAG service requires either OpenAI API key or SentenceTransformer library.")
                self.embedding_model = None
        
        # Get index name from settings (optional - indexes should be specified in UI)
        self.index_name = settings.PINECONE_INDEX_NAME
        
        # Only connect to index if it's explicitly configured and exists
        # We don't auto-create indexes - they should be created via notebooks
        if self.index_name:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if self.index_name in existing_indexes:
                try:
                    index_info = self.pc.describe_index(self.index_name)
                    existing_dimension = index_info.dimension
                    if existing_dimension != self.embedding_dimension:
                        print(f"WARNING: Existing index '{self.index_name}' has dimension {existing_dimension}, but current embedding model uses {self.embedding_dimension}.")
                        print(f"This will cause query failures. Please either:")
                        print(f"1. Delete the existing index and recreate it, or")
                        print(f"2. Use a different index name, or")
                        print(f"3. Switch to the embedding model that matches dimension {existing_dimension}")
                        raise Exception(f"Index dimension mismatch: index has {existing_dimension}, embeddings use {self.embedding_dimension}")
                except Exception as e:
                    if "dimension mismatch" in str(e):
                        raise
                    # If describe_index fails, continue (might be permission issue)
                    print(f"Warning: Could not check index dimension: {e}")
                
                # Connect to index if it exists
                self.index = self.pc.Index(self.index_name)
                print(f"Connected to Pinecone index: {self.index_name}")
            else:
                print(f"Warning: Index '{self.index_name}' not found in Pinecone. Indexes should be created via notebooks.")
                self.index = None
        else:
            # No default index configured - indexes must be selected in UI
            self.index = None
            print("No default Pinecone index configured. Indexes must be selected in the UI (contracts, qadha, etc.)")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.use_openai_embeddings and self.openai_client:
            # Use OpenAI embeddings (MUST match notebook: text-embedding-3-large with 3072 dimensions)
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-large",
                    input=text,
                    dimensions=3072
                )
                return response.data[0].embedding
            except Exception as e:
                raise Exception(f"Error generating OpenAI embedding: {str(e)}")
        elif self.embedding_model:
            # Use SentenceTransformer
            embedding = self.embedding_model.encode(text, convert_to_numpy=True).tolist()
            return embedding
        else:
            raise Exception("Embedding model not initialized")
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch (much faster)"""
        if not texts:
            return []
        
        if self.use_openai_embeddings and self.openai_client:
            # Use OpenAI embeddings - batch API call (up to 2048 texts per request)
            try:
                # OpenAI supports batching, but has limits. Process in batches of 100
                batch_size = 100
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-3-large",
                        input=batch_texts,
                        dimensions=3072
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                
                return all_embeddings
            except Exception as e:
                raise Exception(f"Error generating OpenAI embeddings batch: {str(e)}")
        elif self.embedding_model:
            # Use SentenceTransformer - encode all at once (much faster)
            embeddings = self.embedding_model.encode(
                texts, 
                convert_to_numpy=True, 
                show_progress_bar=False,
                batch_size=32  # Process in batches for memory efficiency
            )
            return embeddings.tolist()
        else:
            raise Exception("Embedding model not initialized")
    
    def _extract_text_from_file(
        self, 
        file_content: bytes, 
        content_type: str, 
        filename: str
    ) -> List[Dict[str, Any]]:
        """
        Extract text content from various file types.
        Returns list of pages/chunks with page numbers for PDFs.
        Only returns pages with non-empty content.
        """
        try:
            # Handle text files
            if content_type.startswith("text/"):
                text = file_content.decode("utf-8", errors="ignore")
                if not text or not text.strip():
                    return []
                return [{
                    "page_number": 1,
                    "content": text.strip()
                }]
            
            # Handle PDF files - keep page information
            if content_type == "application/pdf":
                pages_results = ocr_service.extract_text_from_pdf_bytes(file_content)
                pages = []
                for page_result in pages_results:
                    page_num = page_result.get("page_number", len(pages) + 1)
                    
                    # Try multiple ways to get text content
                    raw_text = page_result.get("raw_text", "")
                    
                    # If raw_text is empty, try to get text from other fields
                    if not raw_text or not raw_text.strip():
                        # Check if there's an error message that might contain text
                        if "error" in page_result:
                            error_msg = page_result.get("error", "")
                            # Sometimes the error contains the actual text
                            if error_msg and len(error_msg) > 50:
                                raw_text = error_msg
                        
                        # Try to extract from response text if available
                        if not raw_text and "response_text" in page_result:
                            raw_text = page_result.get("response_text", "")
                    
                    # Only add pages with actual text content
                    if raw_text and raw_text.strip():
                        pages.append({
                            "page_number": page_num,
                            "content": raw_text.strip()
                        })
                    else:
                        # Log for debugging
                        print(f"Warning: Page {page_num} has no extractable text. Page result keys: {list(page_result.keys())}")
                
                if not pages:
                    # Try to get more info about what was returned
                    print(f"Debug: OCR returned {len(pages_results)} page results")
                    if pages_results:
                        print(f"Debug: First page result keys: {list(pages_results[0].keys())}")
                        print(f"Debug: First page result: {pages_results[0]}")
                
                return pages
            
            # Handle image files
            if content_type.startswith("image/"):
                result = ocr_service.extract_text_from_image_bytes(file_content)
                text = result.get("raw_text", "")
                if not text or not text.strip():
                    return []
                return [{
                    "page_number": 1,
                    "content": text.strip()
                }]
            
            # Try to decode as text for unknown types
            try:
                text = file_content.decode("utf-8", errors="ignore")
                if not text or not text.strip():
                    return []
                return [{
                    "page_number": 1,
                    "content": text.strip()
                }]
            except:
                raise Exception(f"Unsupported file type: {content_type}")
        
        except Exception as e:
            raise Exception(f"Error extracting text from file: {str(e)}")
    
    def _chunk_text_by_tokens(
        self, 
        text: str, 
        max_tokens_per_chunk: int = 4096,  # 4,096 tokens per chunk
        overlap_tokens: int = 512  # 512 tokens overlap (12.5%)
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks based on token count.
        Returns list of dicts with 'content' and 'token_count'.
        Optimized to reduce token counting calls.
        """
        from app.core.token_counter import token_counter
        
        # Use a standard model for token counting (GPT-4 encoding is good for general text)
        model = "gpt-4"
        
        # Count total tokens once
        total_tokens = token_counter.count_tokens_for_text(text, model)
        
        if total_tokens <= max_tokens_per_chunk:
            return [{
                "content": text,
                "token_count": total_tokens,
                "chunk_number": 1
            }]
        
        chunks = []
        start = 0
        chunk_num = 1
        
        # Estimate characters per token (typically 3-4 chars per token)
        # Use a conservative estimate to reduce token counting calls
        estimated_chars_per_token = 3.5
        
        while start < len(text):
            # Estimate end position based on token count (reduce token counting calls)
            estimated_chars = int(max_tokens_per_chunk * estimated_chars_per_token)
            end = start + estimated_chars
            end = min(end, len(text))
            
            # Get initial chunk and count tokens (only once per chunk)
            chunk_text = text[start:end]
            chunk_tokens = token_counter.count_tokens_for_text(chunk_text, model)
            
            # Adjust end position if needed (binary search approach, but limit iterations)
            max_iterations = 3  # Limit to avoid too many token counting calls
            iteration = 0
            while chunk_tokens > max_tokens_per_chunk and iteration < max_iterations and end > start + 1:
                # Too many tokens, reduce end position
                ratio = max_tokens_per_chunk / chunk_tokens
                end = start + int((end - start) * ratio * 0.9)  # 0.9 for safety margin
                chunk_text = text[start:end]
                chunk_tokens = token_counter.count_tokens_for_text(chunk_text, model)
                iteration += 1
            
            # Try to break at sentence/paragraph boundary if we're not at the end
            # Only do this if chunk is reasonably sized to avoid extra token counting
            if end < len(text) and chunk_tokens > max_tokens_per_chunk * 0.7:
                # Look for paragraph breaks first (best break point), then sentences
                search_start = max(start, end - int(max_tokens_per_chunk * estimated_chars_per_token * 2))
                for break_char in ['\n\n', '\n\n\n', '. ', '.\n', '!\n', '?\n']:
                    last_break = text.rfind(break_char, search_start, end)
                    if last_break != -1:
                        # Only recount if we found a good break point
                        new_end = last_break + len(break_char)
                        new_chunk_text = text[start:new_end]
                        new_chunk_tokens = token_counter.count_tokens_for_text(new_chunk_text, model)
                        # If we found a good break and we're within reasonable range, use it
                        if new_chunk_tokens >= max_tokens_per_chunk * 0.5 and new_chunk_tokens <= max_tokens_per_chunk * 1.2:
                            end = new_end
                            chunk_text = new_chunk_text
                            chunk_tokens = new_chunk_tokens
                            break
            
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "token_count": chunk_tokens,
                    "chunk_number": chunk_num
                })
                chunk_num += 1
            
            # Move start with overlap (convert overlap tokens to approximate chars)
            overlap_chars = int(overlap_tokens * estimated_chars_per_token)
            start = max(start + 1, end - overlap_chars)
            
            # If we're at the end, break
            if start >= len(text):
                break
        
        return chunks
    
    def _ensure_index_exists(self, index_name: str) -> None:
        """
        Ensure a Pinecone index exists, create it if it doesn't.
        
        Args:
            index_name: Name of the index to ensure exists
        """
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if index_name not in existing_indexes:
                print(f"Creating Pinecone index '{index_name}' with dimension {self.embedding_dimension}...")
                self.pc.create_index(
                    name=index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"  # Adjust region as needed
                    )
                )
                print(f"Successfully created index '{index_name}'")
            else:
                # Verify dimension matches
                index_info = self.pc.describe_index(index_name)
                existing_dimension = index_info.dimension
                if existing_dimension != self.embedding_dimension:
                    raise Exception(
                        f"Index '{index_name}' exists but has dimension {existing_dimension}, "
                        f"while current embedding model uses {self.embedding_dimension}. "
                        f"Please delete the index or use a different embedding model."
                    )
        except Exception as e:
            raise Exception(f"Error ensuring index '{index_name}' exists: {str(e)}")
    
    def add_document(
        self,
        file_content: bytes,
        content_type: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        target_index_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a document to the vector database.
        
        Args:
            file_content: File content as bytes
            content_type: MIME type of the file
            filename: Original filename
            metadata: Optional metadata to store with the document
            target_index_name: Optional index name to store the document in. 
                             If not provided, uses the default index or creates "user-index-document"
        
        Returns:
            Dictionary with document ID and information
        """
        try:
            # Determine which index to use
            if target_index_name:
                index_name = target_index_name
            else:
                # Default to user-index-document for user uploads
                index_name = "user-index-document"
            
            # Ensure the index exists
            self._ensure_index_exists(index_name)
            
            # Get or create connection to the target index
            target_index = self.pc.Index(index_name)
            # Extract pages from file (each page is a dict with page_number and content)
            pages = self._extract_text_from_file(file_content, content_type, filename)
            
            if not pages:
                raise Exception(f"No text content extracted from file '{filename}'. The file may be empty, corrupted, or contain only images without text.")
            
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Process each page: chunk by tokens
            all_chunks = []
            total_chunks = 0
            global_chunk_number = 1  # Continuous chunk number across all pages
            
            for page_info in pages:
                page_number = page_info.get("page_number", 1)
                page_text = page_info.get("content", "")
                
                # Skip empty pages (shouldn't happen due to filtering, but double-check)
                if not page_text or not page_text.strip():
                    continue
                
                try:
                    # Chunk this page by tokens (4,096 tokens per chunk)
                    page_chunks = self._chunk_text_by_tokens(
                        page_text,
                        max_tokens_per_chunk=4096,  # 4,096 tokens per chunk
                        overlap_tokens=512  # 512 tokens overlap (12.5%)
                    )
                    
                    # Add page number and update chunk_number to be continuous across all pages
                    for chunk_info in page_chunks:
                        chunk_info["page_number"] = page_number
                        chunk_info["chunk_number"] = global_chunk_number  # Use global continuous number
                        all_chunks.append(chunk_info)
                        global_chunk_number += 1
                        total_chunks += 1
                except Exception as e:
                    # Log error but continue with other pages
                    print(f"Error chunking page {page_number}: {str(e)}")
                    continue
            
            if not all_chunks:
                raise Exception(f"No text chunks created from document '{filename}'. The document may contain only images, be corrupted, or have no extractable text.")
            
            # Prepare vectors for Pinecone upsert
            # OPTIMIZATION: Generate embeddings in batch (much faster)
            chunk_contents = [chunk_info["content"] for chunk_info in all_chunks]
            
            print(f"Generating embeddings for {len(chunk_contents)} chunks in batch...")
            embeddings = self._generate_embeddings_batch(chunk_contents)
            print(f"Generated {len(embeddings)} embeddings")
            
            vectors_to_upsert = []
            created_at = datetime.now().isoformat()
            
            for idx, chunk_info in enumerate(all_chunks):
                chunk_content = chunk_info["content"]
                page_number = chunk_info["page_number"]
                chunk_number = chunk_info["chunk_number"]
                token_count = chunk_info["token_count"]
                
                # Create unique chunk ID
                chunk_id = f"{doc_id}_p{page_number}_c{chunk_number}"
                
                # Get pre-generated embedding
                embedding = embeddings[idx]
                
                # Create metadata for chunk (Pinecone metadata must be flat and simple types)
                # Pinecone has a 40KB limit per vector metadata
                max_text_length = 35000  # Leave ~5KB for other metadata
                stored_text = chunk_content[:max_text_length] if len(chunk_content) > max_text_length else chunk_content
                
                chunk_metadata = {
                    "document_id": doc_id,
                    "filename": str(filename)[:100],  # Limit length for Pinecone
                    "content_type": str(content_type)[:50],
                    "page_number": str(page_number),
                    "chunk_number": str(chunk_number),
                    "token_count": str(token_count),
                    "created_at": created_at,
                    "text": stored_text,  # Store text (up to ~35KB)
                    "text_length": str(len(chunk_content))  # Store original length for reference
                }
                
                # Add custom metadata if provided (flatten if needed)
                if metadata:
                    for key, value in metadata.items():
                        key_str = str(key)[:50]
                        value_str = str(value)[:500]
                        chunk_metadata[key_str] = value_str
                
                vectors_to_upsert.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": chunk_metadata
                })
            
            # Upsert to Pinecone in batches (Pinecone recommends batches of 100)
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                target_index.upsert(vectors=batch)
            
            # Calculate total text length
            total_text_length = sum(len(chunk["content"]) for chunk in all_chunks)
            
            return {
                "document_id": doc_id,
                "filename": filename,
                "content_type": content_type,
                "pages_count": len(pages),
                "chunks_count": total_chunks,
                "total_text_length": total_text_length,
                "index_name": index_name,
                "created_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            raise Exception(f"Error adding document to vector database: {str(e)}")
    
    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0  # Minimum similarity threshold - keep at 0.0 to include all matches (even moderate scores can be useful)
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents/chunks across one or more Pinecone indexes.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (Pinecone filter format)
                           - Can include 'index_name' to specify which indexes to search
                           - Can include 'classification' to filter by classification
            min_similarity: Minimum similarity score threshold (0-1), default 0.0 (no filtering)
        
        Returns:
            List of similar document chunks with scores above threshold
        """
        try:
            if not self.use_openai_embeddings and not self.embedding_model:
                raise Exception("Embedding model not initialized")
            
            # Generate query embedding
            print(f"\n{'='*60}")
            print(f" RAG SEARCH IN PINECONE - Query: '{query}'")
            print(f"{'='*60}")
            query_embedding = self._generate_embedding(query)
            print(f"[OK] Generated query embedding (dimension: {len(query_embedding)})")
            
            # Extract index names from filter_metadata if provided
            # This allows searching specific indexes (e.g., "contracts", "qadha") based on user selection
            target_indexes = []
            if filter_metadata and "index_name" in filter_metadata:
                index_names = filter_metadata.get("index_name")
                if isinstance(index_names, list):
                    target_indexes = index_names.copy()  # Use copy to avoid modifying original
                else:
                    target_indexes = [index_names]
                # Remove index_name from filter_dict as it's not a metadata field in Pinecone
                # (it's used to select which indexes to query, not to filter within an index)
                filter_metadata = {k: v for k, v in filter_metadata.items() if k != "index_name"}
                print(f"[OK] User selected {len(target_indexes)} database index(es): {target_indexes}")
            else:
                print(f"Debug: filter_metadata received: {filter_metadata}")
                if filter_metadata:
                    print(f"Debug: Available keys in filter_metadata: {list(filter_metadata.keys())}")
            
            # IMPORTANT: If user_id or filename is in filter, ALSO search user-uploaded documents
            # User-uploaded documents are stored in "user-index-document" index
            # This enables COMBINED searches: user files + database indexes
            if filter_metadata:
                if "user_id" in filter_metadata or "filename" in filter_metadata:
                    if "user-index-document" not in target_indexes:
                        target_indexes.append("user-index-document")
                        print(f"[OK] User files requested - added 'user-index-document' to search list")
                        print(f"    User filters: user_id={filter_metadata.get('user_id')}, filename={filter_metadata.get('filename')}")
            
            # If no indexes specified in filter, use default index
            # This happens when user hasn't selected any specific indexes in the UI
            if not target_indexes:
                print(f"[ERROR] No indexes selected in UI!")
                print(f"   Please check at least one index checkbox (Qadha, Contracts, etc.) in the UI,")
                print(f"   or select a specific user document, or disable RAG to continue without document search.")
                return []
            
            print(f"Debug: Will search across {len(target_indexes)} index(es): {target_indexes}")
            
            # Build filter for Pinecone (must be in Pinecone filter format)
            filter_dict = None
            if filter_metadata:
                # Convert to Pinecone filter format
                filter_dict = {}
                for key, value in filter_metadata.items():
                    # Skip empty values
                    if not value or (isinstance(value, list) and len(value) == 0):
                        continue
                    
                    # Map frontend field names to Pinecone metadata field names
                    # Frontend sends "classification" (singular) but Pinecone stores "classifications" (plural array)
                    if key == "classification":
                        key = "classifications"
                    
                    # IMPORTANT: Normalize classifications to Arabic names for Pinecone
                    # Database indexes store classifications in Arabic, not as IDs
                    # E.g., 'commercial_law' (ID) -> 'النظام التجاري' (Arabic)
                    if key == "classifications" and isinstance(value, list):
                        value = classification_manager.normalize_classifications_list(value)
                        print(f"Debug: Normalized classifications to Arabic: {value}")
                    
                    if isinstance(value, list):
                        # For array fields like "classifications", use $in to check if array contains any of the values
                        # Pinecone will match if the array field contains any element in the $in list
                        if len(value) == 1:
                            # Single item - use $in for array fields (works for both single and multiple)
                            filter_dict[key] = {"$in": value}
                        else:
                            # Multiple items - use $in operator
                            filter_dict[key] = {"$in": value}
                    else:
                        # Single value - use $eq operator
                        filter_dict[key] = {"$eq": value}
                
                # Only use filter if it has at least one condition
                if not filter_dict:
                    filter_dict = None
                else:
                    print(f"Debug: Using metadata filters: {filter_dict}")
                    print(f"Debug: Original filter_metadata keys: {list(filter_metadata.keys())}")
            
            # Query all specified indexes and merge results
            all_results = []
            
            # First, check if indexes exist and have vectors
            available_indexes = []
            try:
                existing_indexes = self.pc.list_indexes()
                index_names_list = [idx.name for idx in existing_indexes]
                print(f" Available indexes in Pinecone: {index_names_list}")
                
                for idx_name in target_indexes:
                    if idx_name not in index_names_list:
                        print(f" WARNING: Index '{idx_name}' does not exist in Pinecone!")
                        print(f"   Available indexes: {index_names_list}")
                        continue
                    
                    # Get or connect to the index
                    if idx_name == self.index_name:
                        idx = self.index
                    else:
                        # Connect to a different index
                        idx = self.pc.Index(idx_name)
                    
                    # Check if index has any vectors
                    try:
                        stats = idx.describe_index_stats()
                        total_vectors = stats.total_vector_count
                        if total_vectors == 0:
                            print(f" WARNING: Index '{idx_name}' is empty (0 vectors). Skipping.")
                            continue
                        print(f"[OK] Index '{idx_name}' has {total_vectors} vectors - will search this index")
                        
                        # Debug: For user-index-document, let's see what metadata is stored
                        if idx_name == "user-index-document":
                            print(f"Debug: Checking metadata structure in user-index-document...")
                            # Query without filter to see what's actually stored
                            test_query = idx.query(
                                vector=query_embedding,
                                top_k=1,
                                include_metadata=True
                            )
                            if test_query.matches:
                                sample_metadata = test_query.matches[0].metadata
                                print(f"Debug: Sample document metadata keys: {list(sample_metadata.keys())}")
                                print(f"Debug: Sample metadata: {sample_metadata}")
                        
                        available_indexes.append((idx_name, idx))
                    except Exception as e:
                        print(f"  ERROR: Could not get stats for index '{idx_name}': {str(e)}")
                        continue
            except Exception as e:
                print(f" ERROR: Could not list indexes: {str(e)}")
                # Fallback: try to use the indexes anyway
                for idx_name in target_indexes:
                    try:
                        idx = self.pc.Index(idx_name)
                        available_indexes.append((idx_name, idx))
                    except:
                        pass
            
            if not available_indexes:
                print(f"\n ERROR: No valid indexes found to search!")
                print(f"   Requested indexes: {target_indexes}")
                print(f"   Please check:")
                print(f"   1. That the indexes exist in Pinecone")
                print(f"   2. That the indexes have vectors (documents uploaded)")
                print(f"   3. That you've selected the correct indexes in the UI")
                return []
            
            # Now query the available indexes
            for idx_name, idx in available_indexes:
                try:
                    
                    # Query Pinecone - get more results to filter by threshold if needed
                    query_k = top_k * 3 if min_similarity > 0 else top_k * 2  # Get more per index when merging
                    query_params = {
                        "vector": query_embedding,
                        "top_k": query_k,
                        "include_metadata": True
                    }
                    
                    # IMPORTANT: Apply different filters based on index type
                    # User-uploaded documents index: use user_id, filename filters
                    # Database indexes (qadha, contracts, etc.): use classifications filter only
                    index_specific_filter = None
                    if filter_dict:
                        if idx_name == "user-index-document":
                            # For user files index, only use user_id and filename filters
                            index_specific_filter = {}
                            if "user_id" in filter_dict:
                                index_specific_filter["user_id"] = filter_dict["user_id"]
                            if "filename" in filter_dict:
                                index_specific_filter["filename"] = filter_dict["filename"]
                            
                            if index_specific_filter:
                                print(f"Debug: Querying user index '{idx_name}' with filters: {index_specific_filter}")
                            else:
                                print(f"Debug: Querying user index '{idx_name}' without filters (all user documents)")
                        else:
                            # For database indexes, only use classifications filter
                            index_specific_filter = {}
                            if "classifications" in filter_dict:
                                index_specific_filter["classifications"] = filter_dict["classifications"]
                            
                            if index_specific_filter:
                                print(f"Debug: Querying database index '{idx_name}' with filters: {index_specific_filter}")
                            else:
                                print(f"Debug: Querying database index '{idx_name}' without filters (all documents)")
                    else:
                        print(f"Debug: Querying index '{idx_name}' without filters (searching all documents)")
                    
                    if index_specific_filter:
                        query_params["filter"] = index_specific_filter
                    
                    query_response = idx.query(**query_params)
                    
                    # Debug: Show what we got back
                    print(f"Debug: Query returned {len(query_response.matches)} matches from '{idx_name}'")
                    if len(query_response.matches) > 0:
                        # Show first match metadata for debugging
                        first_match = query_response.matches[0]
                        print(f"Debug: First match metadata keys: {list(first_match.metadata.keys()) if first_match.metadata else 'No metadata'}")
                        if first_match.metadata:
                            print(f"Debug: First match metadata sample: {dict(list(first_match.metadata.items())[:5])}")
                    
                    # Add index name to each match and collect
                    for match in query_response.matches:
                        match_dict = {
                            'score': match.score,
                            'id': match.id,
                            'metadata': match.metadata if match.metadata else {},
                            'index_name': idx_name
                        }
                        all_results.append(match_dict)
                    
                    print(f"Debug: Found {len(query_response.matches)} results from index '{idx_name}'")
                    
                except Exception as e:
                    print(f"Debug: Error querying index '{idx_name}': {str(e)}")
                    continue
            
            # Sort all results by score (highest first)
            all_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Filter by minimum similarity threshold and format results
            similar_chunks = []
            
            for match_dict in all_results:
                score = match_dict['score']
                
                # Debug: Print scores
                print(f"Debug: Match score: {score:.4f}, ID: {match_dict['id']}, Index: {match_dict['index_name']}")
                
                # Filter by minimum similarity threshold
                if score < min_similarity:
                    continue
                
                metadata = match_dict.get('metadata', {})
                
                # Get text from metadata (try multiple field names)
                # NOTE: The notebooks don't store text in metadata, only file_name, primary_category, classifications
                # So we need to handle the case where text is missing
                chunk_text = (
                    metadata.get("text") or 
                    metadata.get("chunk_text") or 
                    metadata.get("content") or 
                    ""
                )
                
                # If text is missing, create a placeholder message indicating what document was found
                # This allows the system to at least know what documents match, even if full text isn't available
                if not chunk_text or not chunk_text.strip():
                    filename = metadata.get("file_name") or metadata.get("filename") or "Unknown"
                    primary_category = metadata.get("primary_category", "")
                    classifications = metadata.get("classifications", [])
                    classification_str = ", ".join(classifications) if isinstance(classifications, list) else str(classifications)
                    
                    # Create a descriptive placeholder instead of skipping
                    chunk_text = f"[Document found: {filename}"
                    if primary_category:
                        chunk_text += f" | Category: {primary_category}"
                    if classification_str:
                        chunk_text += f" | Classifications: {classification_str}"
                    chunk_text += f" | Similarity: {score:.4f}]"
                    chunk_text += "\n\n NOTE: Full text content not stored in Pinecone metadata. "
                    chunk_text += "The document matches your query, but the complete text needs to be retrieved from the original source."
                    
                    print(f" WARNING: Chunk {match_dict['id']} has no text content in metadata. Using placeholder.")
                    print(f"   File: {filename}, Score: {score:.4f}")
                
                # Extract page and chunk numbers (handle both string and int)
                page_number = metadata.get("page_number") or metadata.get("page", 1)
                chunk_number = metadata.get("chunk_number") or metadata.get("chunk_index", 1)
                
                # Convert to int if needed
                try:
                    page_number = int(page_number) if page_number else 1
                except (ValueError, TypeError):
                    page_number = 1
                
                try:
                    chunk_number = int(chunk_number) if chunk_number else 1
                except (ValueError, TypeError):
                    chunk_number = 1
                
                # Get filename
                filename = (
                    metadata.get("filename") or 
                    metadata.get("file_name") or 
                    metadata.get("document_id") or 
                    "Unknown"
                )
                
                # Get classification if available
                classification = metadata.get("classification") or metadata.get("classifications", [])
                if isinstance(classification, list) and len(classification) > 0:
                    classification = classification[0]  # Use first classification
                
                # Build result dictionary
                result = {
                    "chunk_id": match_dict['id'],
                    "content": chunk_text,  # This is the actual text content - CRITICAL for RAG
                    "similarity_score": score,
                    "filename": filename,
                    "page_number": page_number,
                    "chunk_number": chunk_number,
                    "index_name": match_dict['index_name'],
                    "metadata": metadata
                }
                
                if classification:
                    result["classification"] = classification
                
                similar_chunks.append(result)
            
            # Take top_k results after merging and sorting
            similar_chunks = similar_chunks[:top_k]
            
            print(f"\n[OK] RAG SEARCH COMPLETE:")
            print(f"  - Total matches found: {len(all_results)}")
            print(f"  - Returning top {len(similar_chunks)} results (merged from {len(target_indexes)} index(es))")
            if similar_chunks:
                print(f"  - Similarity score range: {similar_chunks[0]['similarity_score']:.4f} to {similar_chunks[-1]['similarity_score']:.4f}")
                
                # Check if chunks have actual text content
                chunks_with_text = sum(1 for chunk in similar_chunks if chunk.get("content", "").strip() and not chunk.get("content", "").startswith("[Document found:"))
                chunks_without_text = len(similar_chunks) - chunks_with_text
                
                if chunks_without_text > 0:
                    print(f"\n    CRITICAL WARNING: {chunks_without_text} chunk(s) have NO TEXT CONTENT in metadata!")
                    print(f"     The notebooks stored only metadata (file_name, classifications) but NOT the actual text.")
                    print(f"     These chunks will show document info but the LLM won't have the full content to answer questions.")
                    print(f"      SOLUTION: Re-upload data with 'text' or 'chunk_text' field in metadata (see notebook).\n")
                
                print(f"\n RETRIEVED CHUNKS FROM PINECONE:")
                for i, chunk in enumerate(similar_chunks, 1):
                    preview = chunk['content'][:150].replace('\n', ' ').strip()
                    score = chunk['similarity_score']
                    quality = "HIGH" if score > 0.8 else "MEDIUM" if score > 0.6 else "LOW"
                    has_text = chunk.get("content", "").strip() and not chunk.get("content", "").startswith("[Document found:")
                    text_status = "[OK] Has text" if has_text else "[WARNING] No text (metadata only)"
                    print(f"  {i}. [{chunk['index_name']}] {chunk['filename']} (Page {chunk['page_number']}, Chunk {chunk['chunk_number']})")
                    print(f"     Score: {score:.4f} ({quality} relevance) | {text_status}")
                    print(f"     Content: {preview}...")
                    print()
                print(f"{'='*60}\n")
                print(f" IMPORTANT: These {len(similar_chunks)} chunks were RETRIEVED from Pinecone vector database.")
                print(f"   The model should use this content to answer the question, not its training data.\n")
            else:
                print(f"   WARNING: No chunks retrieved! The model will answer from its own knowledge.\n")
            
            return similar_chunks
        
        except Exception as e:
            print(f"Debug: Search error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Error searching similar documents: {str(e)}")
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        try:
            # Query Pinecone with filter for document_id
            # Note: Pinecone doesn't have a direct "get by metadata" - we need to query
            # We'll use fetch with IDs if we can, but we need to know the IDs first
            # For now, we'll use query with a filter (less efficient but works)
            
            # Query with a dummy vector and filter to get all chunks for this document
            # We'll use a zero vector and rely on metadata filter
            dummy_vector = [0.0] * self.embedding_dimension
            
            # Query with filter - get a large number to ensure we get all chunks
            query_response = self.index.query(
                vector=dummy_vector,
                top_k=10000,  # Large number to get all chunks
                include_metadata=True,
                filter={"document_id": {"$eq": document_id}}
            )
            
            if not query_response.matches:
                return None
            
            chunks = []
            for match in query_response.matches:
                metadata = match.metadata or {}
                chunks.append({
                    "chunk_id": match.id,
                    "content": metadata.get("text", ""),
                    "metadata": metadata
                })
            
            # Sort by chunk_index (convert to int for proper sorting)
            chunks.sort(key=lambda x: int(x["metadata"].get("chunk_index", 0)))
            
            # Get document info from first chunk metadata
            if chunks:
                first_metadata = chunks[0]["metadata"]
                return {
                    "document_id": document_id,
                    "filename": first_metadata.get("filename"),
                    "content_type": first_metadata.get("content_type"),
                    "chunks": chunks,
                    "total_chunks": len(chunks)
                }
            
            return None
        
        except Exception as e:
            raise Exception(f"Error getting document: {str(e)}")
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all unique documents in the database"""
        try:
            # Pinecone doesn't have a direct way to list all unique metadata values
            # We'll use stats to get total vectors, but for unique documents we need to query
            # This is a limitation - we'll use a workaround with query
            
            # Get index stats
            stats = self.index.describe_index_stats()
            total_vectors = stats.total_vector_count
            
            if total_vectors == 0:
                return []
            
            # To get unique documents, we'd need to query and group
            # This is inefficient for large datasets, but works for now
            # We'll query with a dummy vector and collect unique document_ids
            dummy_vector = [0.0] * self.embedding_dimension
            
            # Query a sample to get document IDs (limited to 10000)
            query_response = self.index.query(
                vector=dummy_vector,
                top_k=min(10000, total_vectors),
                include_metadata=True
            )
            
            # Group by document_id
            documents_map = {}
            
            for match in query_response.matches:
                metadata = match.metadata or {}
                doc_id = metadata.get("document_id")
                
                if doc_id and doc_id not in documents_map:
                    documents_map[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata.get("filename"),
                        "content_type": metadata.get("content_type"),
                        "created_at": metadata.get("created_at"),
                        "chunks_count": 0
                    }
            
            # Count chunks for each document (we'd need another query for accurate count)
            # For now, we'll estimate or use the query results
            for match in query_response.matches:
                metadata = match.metadata or {}
                doc_id = metadata.get("document_id")
                if doc_id in documents_map:
                    documents_map[doc_id]["chunks_count"] += 1
            
            return list(documents_map.values())
        
        except Exception as e:
            raise Exception(f"Error listing documents: {str(e)}")
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks by document_id"""
        try:
            # Get all chunk IDs for this document
            # Query with filter to find all chunks
            dummy_vector = [0.0] * self.embedding_dimension
            
            query_response = self.index.query(
                vector=dummy_vector,
                top_k=10000,
                include_metadata=True,
                filter={"document_id": {"$eq": document_id}}
            )
            
            if not query_response.matches:
                return False
            
            # Extract chunk IDs
            chunk_ids = [match.id for match in query_response.matches]
            
            # Delete all chunks
            self.index.delete(ids=chunk_ids)
            
            return True
        
        except Exception as e:
            raise Exception(f"Error deleting document: {str(e)}")
    
    def delete_document_by_filename(self, filename: str) -> bool:
        """Delete a document and all its chunks by filename"""
        try:
            # Get all chunk IDs for this filename
            # Query with filter to find all chunks with this filename
            dummy_vector = [0.0] * self.embedding_dimension
            
            query_response = self.index.query(
                vector=dummy_vector,
                top_k=10000,
                include_metadata=True,
                filter={"filename": {"$eq": filename}}
            )
            
            if not query_response.matches:
                return False
            
            # Extract chunk IDs
            chunk_ids = [match.id for match in query_response.matches]
            
            # Delete all chunks
            self.index.delete(ids=chunk_ids)
            
            return True
        
        except Exception as e:
            raise Exception(f"Error deleting document by filename: {str(e)}")


# Create a singleton instance (will raise error if PINECONE_API_KEY not set)
try:
    rag_service = RAGService()
except Exception as e:
    print(f"Warning: RAG service initialization failed: {e}")
    print("RAG service will not be available until PINECONE_API_KEY is configured.")
    rag_service = None
