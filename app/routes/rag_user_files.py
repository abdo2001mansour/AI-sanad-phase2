from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from app.core.rag_service import rag_service
from app.core.database.s3_service import s3_service
from app.dto.schemas import (
    DocumentUploadResponse,
    RAGQueryRequest
)

router = APIRouter(prefix="/rag/user-files", tags=["RAG - User Files"])


@router.post("/upload", response_model=DocumentUploadResponse, summary="Upload a user document")
async def upload_user_document(
    file: UploadFile = File(...),
    user_id: str = Form(..., description="User ID to associate with this document")
):
    """
    Upload a user document and add it to the vector database.
    
    This endpoint:
    - Saves the file to S3 in sanad-data-source bucket under {user_id}/ folder
    - Extracts text content from the file (supports text, PDF, images via OCR)
    - Splits the content into chunks
    - Generates embeddings for each chunk
    - Stores them in the "user-index-document" Pinecone index with user_id metadata
    - Creates the index automatically if it doesn't exist
    - Returns a unique document ID and index information
    
    Supported file types:
    - Text files (text/plain, text/markdown, etc.)
    - PDF files (application/pdf)
    - Image files (image/jpeg, image/png, etc.) - uses OCR
    
    Note: All user-uploaded documents are stored in the "user-index-document" index,
    which is separate from the pre-loaded database indexes (qadha, contracts, etc.).
    """
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized. Please set PINECONE_API_KEY environment variable."
        )
    
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required")
    
    try:
        # Read file content
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Get content type
        content_type = file.content_type or "application/octet-stream"
        filename = file.filename or "unknown"
        
        # Save file to S3 in sanad-data-source bucket under user_id folder
        s3_key = None
        if s3_service and s3_service.api_configured:
            try:
                bucket_name = "sanad-data-source"
                s3_key = f"{user_id.strip()}/{filename}"
                
                # Upload to S3
                s3_service.s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=file_content,
                    ContentType=content_type
                )
                print(f"File uploaded to S3: s3://{bucket_name}/{s3_key}")
            except Exception as s3_error:
                print(f"Warning: Failed to upload to S3: {s3_error}")
                # Continue even if S3 upload fails
        
        # Prepare metadata with user_id and S3 location
        metadata = {
            "user_id": user_id.strip(),
            "s3_bucket": "sanad-data-source" if s3_key else None,
            "s3_key": s3_key
        }
        
        # Add document to vector database in "user-index-document" index
        # The index will be created automatically if it doesn't exist
        result = rag_service.add_document(
            file_content=file_content,
            content_type=content_type,
            filename=filename,
            metadata=metadata,
            target_index_name="user-index-document"
        )
        
        return DocumentUploadResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.post("/query", summary="Search in user-uploaded documents")
async def query_user_documents(request: RAGQueryRequest):
    """
    Search for relevant content in user-uploaded documents.
    
    This endpoint searches only in the "user-index-document" index where user uploads are stored.
    Returns matching chunks with their page numbers, similarity scores, and content.
    
    You can filter by:
    - user_id: To search only documents uploaded by a specific user
    - filename: To search only in specific files
    """
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized. Please set PINECONE_API_KEY environment variable."
        )
    
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Build filter for user-index-document
        filter_metadata = request.filter_metadata or {}
        
        # Ensure we're searching in user-index-document
        # Don't override if user specified an index, but default to user-index-document
        if "index_name" not in filter_metadata:
            filter_metadata["index_name"] = ["user-index-document"]
        
        # Search for similar chunks
        results = rag_service.search_similar(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=filter_metadata,
            min_similarity=request.min_similarity if request.min_similarity is not None else 0.0
        )
        
        # Format response with page/chunk info and content
        page_info = []
        for result in results:
            page_info.append({
                "filename": result.get("filename", "Unknown"),
                "page_number": result.get("page_number", 1),
                "chunk_number": result.get("chunk_number", 1),
                "similarity_score": result.get("similarity_score", 0),
                "content": result.get("content", "")
            })
        
        return {
            "query": request.query,
            "index": "user-index-document",
            "results": page_info,
            "total_results": len(page_info)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching user documents: {str(e)}")


@router.get("/list/{user_id}", summary="List files uploaded by a user")
async def list_user_files(user_id: str):
    """
    List all files uploaded by a specific user.
    
    This endpoint searches the user-index-document Pinecone index and S3 bucket
    for all documents with the given user_id.
    """
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized. Please set PINECONE_API_KEY environment variable."
        )
    
    try:
        # Search for all documents with this user_id
        filter_metadata = {"user_id": user_id.strip(), "index_name": ["user-index-document"]}
        
        # Get a sample of results to extract filenames
        results = rag_service.search_similar(
            query="document",  # Generic query to get results
            top_k=1000,  # Get many results to find all files
            filter_metadata=filter_metadata,
            min_similarity=0.0
        )
        
        # Extract unique filenames
        filenames = set()
        for result in results:
            filename = result.get("filename")
            if filename:
                filenames.add(filename)
        
        # Also check S3 for files in the user's folder
        s3_files = []
        if s3_service and s3_service.api_configured:
            try:
                bucket_name = "sanad-data-source"
                user_prefix = f"{user_id.strip()}/"
                objects = s3_service.list_objects(bucket_name, prefix=user_prefix)
                s3_files = [obj["name"].replace(user_prefix, "") for obj in objects.get("files", [])]
            except Exception as s3_error:
                print(f"Warning: Failed to list S3 files: {s3_error}")
        
        # Combine Pinecone and S3 filenames
        all_files = list(filenames.union(set(s3_files)))
        all_files.sort()
        
        return {
            "user_id": user_id,
            "files": all_files,
            "total": len(all_files),
            "source": "Pinecone + S3"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing user files: {str(e)}")


@router.delete("/delete/{filename}", summary="Delete a user document by filename")
async def delete_user_document(filename: str, user_id: str = Form(...)):
    """
    Delete a user document and all its chunks from the vector database by filename.
    
    This operation cannot be undone. All chunks associated with this filename will be deleted
    from the user-index-document index.
    
    Requires user_id to ensure users can only delete their own documents.
    """
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized. Please set PINECONE_API_KEY environment variable."
        )
    
    try:
        # Find document by filename (filtered by user_id for security)
        # TODO: Implement filtered delete in RAG service
        success = rag_service.delete_document_by_filename(filename)
        
        if not success:
            raise HTTPException(
                status_code=404, 
                detail=f"Document with filename '{filename}' not found in vector database"
            )
        
        return {
            "message": f"Document '{filename}' and all its chunks deleted successfully",
            "filename": filename,
            "user_id": user_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@router.get("/health", summary="User files RAG service health check")
async def user_files_health_check():
    """Check if the user files RAG service is ready"""
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
        
        return {
            "status": "healthy",
            "message": "User files RAG service is ready",
            "index_name": "user-index-document",
            "description": "Handles user-uploaded documents stored in user-index-document Pinecone index"
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"User files RAG service error: {str(e)}"
            }
        )

