from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse, StreamingResponse
from typing import Optional
import io
import urllib.parse
import uuid
from datetime import datetime
from app.core.database.s3_service import s3_service
from app.dto.schemas import ErrorResponse

router = APIRouter(prefix="/s3", tags=["S3"])


@router.get("/buckets", summary="List all S3 buckets")
async def list_buckets():
    """
    Get a list of all S3 buckets accessible with the current AWS credentials.
    
    Returns:
        List of buckets with names and creation dates
    """
    if s3_service is None:
        raise HTTPException(
            status_code=503,
            detail="S3 service not initialized. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        )
    
    try:
        buckets = s3_service.list_buckets()
        return {
            "buckets": buckets,
            "total": len(buckets)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing buckets: {str(e)}")


@router.get("/buckets/{bucket_name}/objects", summary="List objects and folders in a bucket")
async def list_objects(
    bucket_name: str,
    prefix: str = Query("", description="Folder path prefix (e.g., 'folder1/subfolder/')"),
    delimiter: str = Query("/", description="Delimiter for grouping (default: '/')")
):
    """
    List objects and folders in an S3 bucket.
    
    Args:
        bucket_name: Name of the S3 bucket
        prefix: Folder path prefix to filter objects
        delimiter: Delimiter to use for grouping folders
    
    Returns:
        Dictionary with folders and files lists
    """
    if s3_service is None:
        raise HTTPException(
            status_code=503,
            detail="S3 service not initialized. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        )
    
    try:
        result = s3_service.list_objects(
            bucket_name=bucket_name,
            prefix=prefix,
            delimiter=delimiter
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/buckets/{bucket_name}/objects/{file_path:path}", summary="Get a file from S3")
async def get_file(
    bucket_name: str,
    file_path: str,
    download: bool = Query(False, description="Force download instead of displaying")
):
    """
    Get a file from S3 bucket.
    
    Args:
        bucket_name: Name of the S3 bucket
        file_path: Path to the file in the bucket (e.g., 'folder/file.pdf')
        download: If True, force download; if False, try to display inline
    
    Returns:
        File content with appropriate content type
    """
    if s3_service is None:
        raise HTTPException(
            status_code=503,
            detail="S3 service not initialized. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        )
    
    try:
        result = s3_service.get_object(bucket_name=bucket_name, key=file_path)
        content = result['content']
        metadata = result['metadata']
        
        # Determine content type
        content_type = metadata.get('content_type', 'application/octet-stream')
        
        # Get filename from path
        filename = file_path.split('/')[-1] if '/' in file_path else file_path
        
        # Encode filename for Content-Disposition header (handles non-ASCII characters)
        # HTTP headers must be latin-1 encodable, so we need to handle non-ASCII filenames carefully
        def prepare_content_disposition(filename, disposition_type):
            """Prepare Content-Disposition header with proper encoding for non-ASCII filenames"""
            try:
                # Check if filename is ASCII-safe
                filename.encode('ascii')
                # Pure ASCII - use directly
                return f'{disposition_type}; filename="{filename}"'
            except UnicodeEncodeError:
                # Contains non-ASCII characters - use RFC 5987 encoding
                # Create ASCII-safe fallback
                safe_ascii_name = filename.encode('ascii', errors='ignore').decode('ascii') or 'file'
                # Encode UTF-8 filename using RFC 5987
                utf8_bytes = filename.encode('utf-8')
                encoded = urllib.parse.quote(utf8_bytes, safe='')
                # Use both filename (ASCII fallback) and filename* (UTF-8 encoded)
                return f'{disposition_type}; filename="{safe_ascii_name}"; filename*=UTF-8\'\'{encoded}'
        
        # Determine disposition type
        disposition_type = "attachment" if download else (
            "inline" if (content_type.startswith('image/') or content_type == 'application/pdf' or content_type.startswith('text/')) 
            else "attachment"
        )
        
        # Build Content-Disposition header (always ASCII-safe)
        content_disposition = prepare_content_disposition(filename, disposition_type)
        
        return Response(
            content=content,
            media_type=content_type,
            headers={
                "Content-Disposition": content_disposition,
                "Content-Length": str(metadata.get('content_length', len(content)))
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/buckets/{bucket_name}/presigned-url", summary="Get presigned URL for a file")
async def get_presigned_url(
    bucket_name: str,
    key: str = Query(..., description="S3 object key (file path)"),
    expiration: int = Query(3600, description="URL expiration time in seconds (default: 3600)")
):
    """
    Generate a presigned URL for an S3 object.
    
    Args:
        bucket_name: Name of the S3 bucket
        key: S3 object key (file path)
        expiration: URL expiration time in seconds
    
    Returns:
        Presigned URL
    """
    if s3_service is None:
        raise HTTPException(
            status_code=503,
            detail="S3 service not initialized. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        )
    
    try:
        url = s3_service.get_presigned_url(
            bucket_name=bucket_name,
            key=key,
            expiration=expiration
        )
        return {
            "url": url,
            "expiration_seconds": expiration,
            "bucket": bucket_name,
            "key": key
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/buckets/{bucket_name}/cdn-url", summary="Get CDN URL for a file")
async def get_cdn_url(
    bucket_name: str,
    key: str = Query(..., description="S3 object key (file path)")
):
    """
    Generate a CDN URL for an S3 object.
    
    Args:
        bucket_name: Name of the S3 bucket
        key: S3 object key (file path)
    
    Returns:
        CDN URL
    """
    if s3_service is None:
        raise HTTPException(
            status_code=503,
            detail="S3 service not initialized. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        )
    
    try:
        url = s3_service.get_cdn_url(key=key, bucket_name=bucket_name)
        return {
            "url": url,
            "bucket": bucket_name,
            "key": key
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/{bucket_name}", summary="Upload a file to S3 bucket")
async def upload_file(
    bucket_name: str,
    file: UploadFile = File(..., description="File to upload"),
    prefix: str = Query("", description="Folder prefix (e.g., 'meeting-videos/')"),
    custom_filename: Optional[str] = Query(None, description="Custom filename (optional, defaults to original name)")
):
    """
    Upload a file directly to an S3 bucket.
    
    Args:
        bucket_name: Name of the S3 bucket
        file: File to upload
        prefix: Folder prefix for the file (e.g., 'meeting-videos/')
        custom_filename: Optional custom filename
    
    Returns:
        Upload result with CDN and presigned URLs
    """
    if s3_service is None:
        raise HTTPException(
            status_code=503,
            detail="S3 service not initialized. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Generate filename
        original_filename = file.filename or "file"
        if custom_filename:
            filename = custom_filename
        else:
            # Add timestamp to avoid collisions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = original_filename.rsplit('.', 1) if '.' in original_filename else (original_filename, '')
            filename = f"{timestamp}_{name}.{ext}" if ext else f"{timestamp}_{name}"
        
        # Construct S3 key
        prefix = prefix.strip('/')
        if prefix:
            key = f"{prefix}/{filename}"
        else:
            key = filename
        
        # Determine content type
        content_type = file.content_type or "application/octet-stream"
        
        # Upload to S3
        result = s3_service.upload_file(
            file_content=file_content,
            bucket_name=bucket_name,
            key=key,
            content_type=content_type
        )
        
        return {
            "success": True,
            "message": "File uploaded successfully",
            "bucket": bucket_name,
            "key": key,
            "filename": filename,
            "original_filename": original_filename,
            "content_type": content_type,
            "size": len(file_content),
            "size_mb": round(len(file_content) / (1024 * 1024), 2),
            "cdn_url": result.get("cdn_url"),
            "presigned_url": result.get("presigned_url")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/presigned-upload/{bucket_name}", summary="Get presigned URL for uploading")
async def get_presigned_upload_url(
    bucket_name: str,
    filename: str = Query(..., description="Filename to upload"),
    prefix: str = Query("", description="Folder prefix (e.g., 'meeting-videos/')"),
    content_type: str = Query("application/octet-stream", description="MIME type of the file"),
    expiration: int = Query(3600, description="URL expiration time in seconds (default: 1 hour)")
):
    """
    Generate a presigned URL for uploading a file directly to S3 from the browser.
    
    Args:
        bucket_name: Name of the S3 bucket
        filename: Name for the uploaded file
        prefix: Folder prefix for the file
        content_type: MIME type of the file
        expiration: URL expiration time in seconds
    
    Returns:
        Presigned POST data for browser upload
    """
    if s3_service is None:
        raise HTTPException(
            status_code=503,
            detail="S3 service not initialized."
        )
    
    try:
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        unique_filename = f"{timestamp}_{name}.{ext}" if ext else f"{timestamp}_{name}"
        
        # Construct S3 key
        prefix = prefix.strip('/')
        if prefix:
            key = f"{prefix}/{unique_filename}"
        else:
            key = unique_filename
        
        result = s3_service.generate_presigned_upload_url(
            bucket_name=bucket_name,
            key=key,
            content_type=content_type,
            expiration=expiration
        )
        
        return {
            "success": True,
            **result,
            "filename": unique_filename,
            "original_filename": filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", summary="S3 service health check")
async def s3_health_check():
    """
    Check if the S3 service is properly configured and ready to use.
    """
    try:
        if s3_service is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "S3 service not initialized",
                    "details": "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
                }
            )
        
        if not s3_service.api_configured:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "AWS credentials not configured",
                    "details": "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
                }
            )
        
        # Try to list buckets to verify credentials work
        try:
            buckets = s3_service.list_buckets()
            return {
                "status": "healthy",
                "message": "S3 service is ready",
                "region": s3_service.aws_region,
                "default_bucket": s3_service.default_bucket,
                "cdn_url": s3_service.cdn_url,
                "buckets_count": len(buckets)
            }
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "S3 service configured but cannot access buckets",
                    "details": str(e)
                }
            )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"S3 service error: {str(e)}"
            }
        )

