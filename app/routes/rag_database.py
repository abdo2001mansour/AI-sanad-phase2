from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from app.core.rag_service import rag_service
from app.core.classifications import classification_manager
from app.core.indexes import index_manager
from app.dto.schemas import RAGFilteredQueryRequest

router = APIRouter(prefix="/rag/database", tags=["RAG - Database Indexes"])


@router.post("/query", summary="Search in database indexes (Qadha, Contracts, etc.)")
async def query_database_indexes(request: RAGFilteredQueryRequest):
    """
    Search documents in pre-loaded database Pinecone indexes with filtering.
    
    This endpoint allows you to search in the pre-loaded legal document indexes like:
    - qadha (قضايا)
    - contracts (عقود)
    - bog (البنك الأهلي)
    - new_ncar (نظام المحاماة الجديد)
    - namazig (النماذج)
    
    You can filter by:
    - index_names: Select which indexes to search (e.g., ["qadha", "contracts"])
    - classifications: Filter by legal classification categories
    
    Example request:
    {
        "query": "ماذا تعرف عن نظام ملكية العقارات",
        "top_k": 10,
        "index_names": ["qadha", "contracts"],
        "classifications": ["civil_transactions", "commercial_law"]
    }
    """
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized. Please set PINECONE_API_KEY environment variable."
        )
    
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Build filter_metadata from explicit parameters
        filter_metadata = {}
        
        if request.index_names:
            if not isinstance(request.index_names, list) or len(request.index_names) == 0:
                raise HTTPException(
                    status_code=400, 
                    detail="index_names must be a non-empty list if provided"
                )
            # Convert index identifiers (IDs, Arabic, or English) to technical Pinecone names
            technical_names = index_manager.convert_to_technical_names(request.index_names)
            filter_metadata["index_name"] = technical_names
            
            print(f"Debug: Index names normalized from {request.index_names} to {technical_names}")
        
        if request.classifications:
            if not isinstance(request.classifications, list) or len(request.classifications) == 0:
                raise HTTPException(
                    status_code=400, 
                    detail="classifications must be a non-empty list if provided"
                )
            # Normalize classifications to Arabic names (supports IDs, Arabic, or English)
            # Pinecone stores classifications in Arabic, so we convert IDs/English to Arabic
            normalized_classifications = classification_manager.normalize_classifications_list(request.classifications)
            filter_metadata["classifications"] = normalized_classifications
            
            print(f"Debug: Classifications normalized from {request.classifications} to {normalized_classifications}")
        
        # User documents should NOT be included in database searches
        # If no indexes specified, return error
        if not filter_metadata.get("index_name"):
            raise HTTPException(
                status_code=400,
                detail="Please specify at least one database index to search (qadha, contracts, bog, new_ncar, namazig)"
            )
        
        # Search for similar chunks with filters
        results = rag_service.search_similar(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=filter_metadata,
            min_similarity=request.min_similarity if request.min_similarity is not None else 0.0
        )
        
        # Format response with detailed information
        formatted_results = []
        for result in results:
            # Extract metadata from result
            result_metadata = result.get("metadata", {})
            
            # Extract S3 information from metadata
            s3_bucket = result_metadata.get("s3_bucket")
            s3_key = result_metadata.get("s3_key")
            
            # Build S3 file path/URL if both bucket and key are available
            s3_path = None
            s3_url = None
            if s3_bucket and s3_key:
                s3_path = f"s3://{s3_bucket}/{s3_key}"
                s3_url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_key}" if s3_bucket else None
            
            formatted_results.append({
                "filename": result.get("filename", "Unknown"),
                "index_name": result.get("index_name", "unknown"),
                "page_number": result.get("page_number", 1),
                "chunk_number": result.get("chunk_number", 1),
                "chunk_index": result_metadata.get("chunk_index", 0),
                "similarity_score": result.get("similarity_score", 0),
                "content": result.get("content", ""),
                "content_preview": (result.get("content", "") or "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
                "s3_location": {
                    "bucket": s3_bucket,
                    "key": s3_key,
                    "s3_path": s3_path,
                    "s3_url": s3_url
                },
                "metadata": {
                    "file_name": result.get("filename", "Unknown"),
                    "primary_category": result_metadata.get("primary_category", "unknown"),
                    "classifications": result_metadata.get("classifications", []),
                    "chunk_index": result_metadata.get("chunk_index", 0),
                    "total_chunks": result_metadata.get("total_chunks"),
                    "s3_bucket": s3_bucket,
                    "s3_key": s3_key
                }
            })
        
        return {
            "query": request.query,
            "filters_applied": {
                "index_names": request.index_names,
                "classifications": request.classifications
            },
            "results": formatted_results,
            "total_results": len(formatted_results)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching database indexes: {str(e)}")


@router.get("/indexes", summary="List available database indexes with translations")
async def list_database_indexes():
    """
    List all available pre-loaded database indexes with Arabic and English names.
    
    Returns information about database indexes with:
    - id: Unique identifier (e.g., "regulations", "precedents_diwan")
    - technical_name: Actual Pinecone index name (e.g., "qadha", "contracts")
    - name_ar: Arabic display name (e.g., "سوابق قضائية (ديوان المظالم)")
    - name_en: English display name (e.g., "Judicial Precedents (Board of Grievances)")
    - total_vectors: Number of documents in the index
    - icon: Emoji icon for UI display
    
    Excludes the user-index-document index which is for user uploads.
    """
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized. Please set PINECONE_API_KEY environment variable."
        )
    
    try:
        # Get all configured indexes with translations
        all_indexes = index_manager.get_all_with_translations()
        
        # List actual Pinecone indexes to get stats
        pinecone_indexes = rag_service.pc.list_indexes()
        
        # Build response with stats
        database_indexes = []
        for idx_config in all_indexes:
            technical_name = idx_config["technical_name"]
            
            # Skip user-index-document
            if technical_name == "user-index-document":
                continue
            
            # Check if this index exists in Pinecone
            pinecone_idx = next((idx for idx in pinecone_indexes if idx.name == technical_name), None)
            
            if pinecone_idx:
                try:
                    # Connect to index to get stats
                    index = rag_service.pc.Index(technical_name)
                    stats = index.describe_index_stats()
                    
                    database_indexes.append({
                        "id": idx_config["id"],
                        "technical_name": technical_name,
                        "name_ar": idx_config["name_ar"],
                        "name_en": idx_config["name_en"],
                        "description_ar": idx_config.get("description_ar", ""),
                        "description_en": idx_config.get("description_en", ""),
                        "dimension": stats.dimension,
                        "total_vectors": stats.total_vector_count,
                        "metric": "cosine",
                        "status": "active"
                    })
                except Exception as e:
                    database_indexes.append({
                        "id": idx_config["id"],
                        "technical_name": technical_name,
                        "name_ar": idx_config["name_ar"],
                        "name_en": idx_config["name_en"],
                        "status": "error",
                        "error": str(e)
                    })
            else:
                # Index configured but not in Pinecone
                database_indexes.append({
                    "id": idx_config["id"],
                    "technical_name": technical_name,
                    "name_ar": idx_config["name_ar"],
                    "name_en": idx_config["name_en"],
                    "status": "not_found",
                    "message": "Index configured but not found in Pinecone"
                })
        
        return {
            "total_indexes": len(database_indexes),
            "indexes": database_indexes,
            "description": "Pre-loaded legal document indexes with Arabic and English names"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing database indexes: {str(e)}")


@router.get("/classifications", summary="Get available classifications with translations")
async def get_available_classifications():
    """
    Get a list of available legal classification categories with both Arabic and English names.
    
    This endpoint returns all legal classifications with:
    - id: Unique identifier (e.g., "administrative_law")
    - name_ar: Arabic name (e.g., "النظام الإداري")
    - name_en: English name (e.g., "Administrative Law")
    - description_ar: Arabic description
    - description_en: English description
    
    Use the `id` when making API calls for consistency, and display the appropriate
    language (name_ar or name_en) in your UI.
    """
    classifications = classification_manager.get_all_with_translations()
    
    return {
        "classifications": classifications,
        "total": len(classifications),
        "description": "Saudi legal classification categories with Arabic and English translations"
    }


@router.get("/health", summary="Database RAG service health check")
async def database_health_check():
    """Check if the database RAG service is ready"""
    try:
        if rag_service is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "RAG service not initialized",
                    "details": "Please set PINECONE_API_KEY environment variable"
                }
            )
        
        # Count database indexes (excluding user-index-document)
        try:
            indexes = rag_service.pc.list_indexes()
            database_count = sum(1 for idx in indexes if idx.name != "user-index-document")
        except:
            database_count = 0
        
        return {
            "status": "healthy",
            "message": "Database RAG service is ready",
            "database_indexes_count": database_count,
            "description": "Handles pre-loaded legal document indexes (qadha, contracts, bog, etc.)"
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Database RAG service error: {str(e)}"
            }
        )

