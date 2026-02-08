from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from app.core.rag_service import rag_service

router = APIRouter(prefix="/rag", tags=["RAG - General"])


@router.get("/inspect/{index_name}", summary="Inspect documents in a Pinecone index")
async def inspect_pinecone_index(
    index_name: str,
    limit: int = Query(10, ge=1, le=100, description="Number of sample documents to retrieve")
):
    """
    Inspect what documents are stored in a specific Pinecone index.
    
    This endpoint retrieves sample documents from the specified index and shows their metadata.
    Useful for debugging and understanding what's stored in Pinecone.
    
    Args:
        index_name: Name of the Pinecone index to inspect (e.g., 'user-index-document', 'qadha', 'contracts')
        limit: Number of sample documents to retrieve (default: 10, max: 100)
    
    Returns:
        Information about the index and sample documents with their metadata
    """
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized. Please set PINECONE_API_KEY environment variable."
        )
    
    try:
        # Connect to the specified index
        from pinecone import Pinecone
        from app.config.settings import settings
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing_indexes:
            raise HTTPException(
                status_code=404,
                detail=f"Index '{index_name}' not found. Available indexes: {existing_indexes}"
            )
        
        # Connect to the index
        index = pc.Index(index_name)
        
        # Get index stats
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        
        if total_vectors == 0:
            return {
                "index_name": index_name,
                "total_vectors": 0,
                "dimension": stats.dimension,
                "message": "Index is empty - no documents uploaded yet",
                "unique_user_ids": [],
                "unique_filenames": [],
                "sample_documents": []
            }
        
        # Query to get sample documents (use a dummy query to get any documents)
        # Get dimension from index stats
        embedding_dimension = stats.dimension
        dummy_vector = [0.0] * embedding_dimension
        sample_results = index.query(
            vector=dummy_vector,
            top_k=min(limit, total_vectors),
            include_metadata=True
        )
        
        # Extract sample documents with metadata
        sample_documents = []
        unique_user_ids = set()
        unique_filenames = set()
        
        for match in sample_results.matches:
            metadata = match.metadata or {}
            
            # Collect unique user_ids and filenames
            if metadata.get("user_id"):
                unique_user_ids.add(metadata.get("user_id"))
            if metadata.get("filename"):
                unique_filenames.add(metadata.get("filename"))
            if metadata.get("file_name"):
                unique_filenames.add(metadata.get("file_name"))
            
            # Build sample document info
            doc_info = {
                "id": match.id,
                "score": match.score,
                "metadata": {
                    "user_id": metadata.get("user_id"),
                    "filename": metadata.get("filename") or metadata.get("file_name"),
                    "document_id": metadata.get("document_id"),
                    "page_number": metadata.get("page_number"),
                    "chunk_number": metadata.get("chunk_number"),
                    "content_type": metadata.get("content_type"),
                    "created_at": metadata.get("created_at"),
                    "s3_bucket": metadata.get("s3_bucket"),
                    "s3_key": metadata.get("s3_key"),
                    "text_preview": (metadata.get("text", "") or "")[:200] + "..." if len(metadata.get("text", "")) > 200 else metadata.get("text", ""),
                    "classifications": metadata.get("classifications"),
                    "primary_category": metadata.get("primary_category"),
                }
            }
            
            # Remove None values for cleaner output
            doc_info["metadata"] = {k: v for k, v in doc_info["metadata"].items() if v is not None}
            
            sample_documents.append(doc_info)
        
        return {
            "index_name": index_name,
            "total_vectors": total_vectors,
            "dimension": stats.dimension,
            "unique_user_ids": sorted(list(unique_user_ids)),
            "unique_filenames": sorted(list(unique_filenames)),
            "sample_size": len(sample_documents),
            "sample_documents": sample_documents
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error inspecting index: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log to console for debugging
        raise HTTPException(status_code=500, detail=f"Error inspecting index: {str(e)}")


@router.get("/inspect", summary="List all Pinecone indexes with basic info")
async def list_pinecone_indexes():
    """
    List all available Pinecone indexes with basic information.
    
    Returns:
        List of all indexes (both database and user uploads) with their vector counts
    """
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized. Please set PINECONE_API_KEY environment variable."
        )
    
    try:
        # List all indexes
        indexes = rag_service.pc.list_indexes()
        
        index_info = []
        for idx in indexes:
            try:
                # Connect to index to get stats
                index = rag_service.pc.Index(idx.name)
                stats = index.describe_index_stats()
                
                # Determine type
                index_type = "user_uploads" if idx.name == "user-index-document" else "database"
                
                index_info.append({
                    "name": idx.name,
                    "dimension": stats.dimension,
                    "total_vectors": stats.total_vector_count,
                    "metric": "cosine",
                    "type": index_type
                })
            except Exception as e:
                index_info.append({
                    "name": idx.name,
                    "error": str(e)
                })
        
        return {
            "total_indexes": len(index_info),
            "indexes": index_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing indexes: {str(e)}")


@router.get("/health", summary="RAG service health check")
async def rag_health_check():
    """
    Check if the RAG service is properly configured and ready to use.
    """
    try:
        # Check if RAG service is initialized
        if rag_service is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "RAG service not initialized",
                    "details": "Please set PINECONE_API_KEY environment variable"
                }
            )
        
        # Check if embedding model is initialized
        if not rag_service.use_openai_embeddings and not rag_service.embedding_model:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "Embedding model not initialized",
                    "details": "The embedding model failed to load"
                }
            )
        
        # Try to get all indexes stats
        try:
            indexes = rag_service.pc.list_indexes()
            total_indexes = len([idx for idx in indexes])
            database_indexes = len([idx for idx in indexes if idx.name != "user-index-document"])
            user_index_exists = any(idx.name == "user-index-document" for idx in indexes)
        except Exception as e:
            total_indexes = 0
            database_indexes = 0
            user_index_exists = False
        
        embedding_info = "OpenAI text-embedding-3-large (3072 dim)" if rag_service.use_openai_embeddings else "SentenceTransformer (384 dim)"
        
        return {
            "status": "healthy",
            "message": "RAG service is ready",
            "total_indexes": total_indexes,
            "database_indexes": database_indexes,
            "user_uploads_index_exists": user_index_exists,
            "embedding_model": embedding_info,
            "embedding_dimension": rag_service.embedding_dimension
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"RAG service error: {str(e)}"
            }
        )
