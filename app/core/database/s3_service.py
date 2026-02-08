import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import List, Dict, Any, Optional
from app.config.settings import settings
import io


class S3Service:
    """Service for handling AWS S3 operations"""
    
    def __init__(self):
        """Initialize the S3 service with AWS credentials"""
        self.aws_access_key_id = settings.AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY
        self.aws_region = settings.AWS_REGION
        self.default_bucket = settings.AWS_S3_BUCKET_NAME
        self.cdn_url = settings.AWS_CDN_URL
        
        # Check if AWS credentials are configured
        self.api_configured = bool(self.aws_access_key_id and self.aws_secret_access_key)
        
        if self.api_configured:
            try:
                # Initialize S3 client
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.aws_region
                )
                print(f"S3 Service initialized successfully. Region: {self.aws_region}")
            except Exception as e:
                print(f"Warning: Failed to initialize S3 client: {e}")
                self.s3_client = None
        else:
            self.s3_client = None
            print("Warning: AWS credentials not configured. S3 service will not be available.")
    
    def list_buckets(self) -> List[Dict[str, Any]]:
        """
        List all S3 buckets accessible with the current credentials.
        
        Returns:
            List of bucket dictionaries with name and creation date
        """
        if not self.api_configured or not self.s3_client:
            raise Exception("AWS credentials not configured. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        
        try:
            response = self.s3_client.list_buckets()
            buckets = []
            for bucket in response.get('Buckets', []):
                buckets.append({
                    'name': bucket['Name'],
                    'creation_date': bucket['CreationDate'].isoformat() if bucket.get('CreationDate') else None
                })
            return buckets
        except ClientError as e:
            raise Exception(f"Error listing buckets: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")
    
    def list_objects(
        self, 
        bucket_name: Optional[str] = None,
        prefix: str = "",
        delimiter: str = "/"
    ) -> Dict[str, Any]:
        """
        List objects and folders in an S3 bucket.
        
        Args:
            bucket_name: Name of the bucket (defaults to configured bucket)
            prefix: Prefix to filter objects (folder path)
            delimiter: Delimiter to use for grouping (default: "/")
        
        Returns:
            Dictionary with 'folders' and 'files' lists
        """
        if not self.api_configured or not self.s3_client:
            raise Exception("AWS credentials not configured. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        
        bucket = bucket_name or self.default_bucket
        if not bucket:
            raise Exception("No bucket name provided and no default bucket configured.")
        
        try:
            # List objects with delimiter to separate folders and files
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=bucket,
                Prefix=prefix,
                Delimiter=delimiter
            )
            
            folders = []
            files = []
            
            for page in pages:
                # Process common prefixes (folders)
                if 'CommonPrefixes' in page:
                    for prefix_info in page['CommonPrefixes']:
                        folder_path = prefix_info['Prefix']
                        folder_name = folder_path.replace(prefix, '').rstrip('/')
                        if folder_name:  # Only add non-empty folder names
                            folders.append({
                                'name': folder_name,
                                'path': folder_path,
                                'full_path': folder_path
                            })
                
                # Process objects (files)
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        # Skip if it's the prefix itself (folder marker)
                        if key == prefix:
                            continue
                        
                        # Skip if it ends with delimiter (it's a folder marker)
                        if key.endswith(delimiter):
                            continue
                        
                        file_name = key.replace(prefix, '')
                        if file_name:  # Only add non-empty file names
                            files.append({
                                'name': file_name,
                                'key': key,
                                'size': obj['Size'],
                                'last_modified': obj['LastModified'].isoformat() if obj.get('LastModified') else None,
                                'etag': obj.get('ETag', '').strip('"'),
                                'full_path': key
                            })
            
            return {
                'bucket': bucket,
                'prefix': prefix,
                'folders': folders,
                'files': files,
                'total_folders': len(folders),
                'total_files': len(files)
            }
        
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchBucket':
                raise Exception(f"Bucket '{bucket}' does not exist.")
            elif error_code == 'AccessDenied':
                raise Exception(f"Access denied to bucket '{bucket}'. Check your AWS credentials and permissions.")
            else:
                raise Exception(f"Error listing objects: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")
    
    def get_object(
        self,
        bucket_name: Optional[str] = None,
        key: str = ""
    ) -> Dict[str, Any]:
        """
        Get an object (file) from S3.
        
        Args:
            bucket_name: Name of the bucket (defaults to configured bucket)
            key: S3 object key (file path)
        
        Returns:
            Dictionary with file content and metadata
        """
        if not self.api_configured or not self.s3_client:
            raise Exception("AWS credentials not configured. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        
        bucket = bucket_name or self.default_bucket
        if not bucket:
            raise Exception("No bucket name provided and no default bucket configured.")
        
        if not key:
            raise Exception("Object key (file path) is required.")
        
        try:
            # Get object
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            
            # Read content
            content = response['Body'].read()
            
            # Get metadata
            metadata = {
                'key': key,
                'bucket': bucket,
                'content_type': response.get('ContentType', 'application/octet-stream'),
                'content_length': response.get('ContentLength', len(content)),
                'last_modified': response.get('LastModified').isoformat() if response.get('LastModified') else None,
                'etag': response.get('ETag', '').strip('"'),
                'metadata': response.get('Metadata', {})
            }
            
            return {
                'content': content,
                'metadata': metadata
            }
        
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchKey':
                raise Exception(f"File '{key}' not found in bucket '{bucket}'.")
            elif error_code == 'NoSuchBucket':
                raise Exception(f"Bucket '{bucket}' does not exist.")
            elif error_code == 'AccessDenied':
                raise Exception(f"Access denied to file '{key}' in bucket '{bucket}'. Check your AWS credentials and permissions.")
            else:
                raise Exception(f"Error getting object: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")
    
    def get_presigned_url(
        self,
        bucket_name: Optional[str] = None,
        key: str = "",
        expiration: int = 3600
    ) -> str:
        """
        Generate a presigned URL for an S3 object.
        
        Args:
            bucket_name: Name of the bucket (defaults to configured bucket)
            key: S3 object key (file path)
            expiration: URL expiration time in seconds (default: 1 hour)
        
        Returns:
            Presigned URL string
        """
        if not self.api_configured or not self.s3_client:
            raise Exception("AWS credentials not configured. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        
        bucket = bucket_name or self.default_bucket
        if not bucket:
            raise Exception("No bucket name provided and no default bucket configured.")
        
        if not key:
            raise Exception("Object key (file path) is required.")
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            raise Exception(f"Error generating presigned URL: {str(e)}")
    
    def get_cdn_url(self, key: str, bucket_name: Optional[str] = None) -> str:
        """
        Generate a CDN URL for an S3 object.
        
        Args:
            key: S3 object key (file path)
            bucket_name: Name of the bucket (optional, for validation)
        
        Returns:
            CDN URL string
        """
        if not self.cdn_url:
            raise Exception("CDN URL not configured.")
        
        # Remove leading slash from key if present
        key = key.lstrip('/')
        
        # Construct CDN URL
        cdn_url = f"{self.cdn_url.rstrip('/')}/{key}"
        return cdn_url

    def generate_presigned_upload_url(
        self,
        bucket_name: Optional[str] = None,
        key: str = "",
        content_type: str = "application/octet-stream",
        expiration: int = 3600
    ) -> Dict[str, Any]:
        """
        Generate a presigned URL for uploading an object to S3.
        
        Args:
            bucket_name: Name of the bucket (defaults to configured bucket)
            key: S3 object key (file path)
            content_type: MIME type of the file
            expiration: URL expiration time in seconds (default: 1 hour)
        
        Returns:
            Dictionary with presigned URL and fields for upload
        """
        if not self.api_configured or not self.s3_client:
            raise Exception("AWS credentials not configured. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        
        bucket = bucket_name or self.default_bucket
        if not bucket:
            raise Exception("No bucket name provided and no default bucket configured.")
        
        if not key:
            raise Exception("Object key (file path) is required.")
        
        try:
            # Generate presigned POST URL
            presigned_post = self.s3_client.generate_presigned_post(
                Bucket=bucket,
                Key=key,
                Fields={
                    "Content-Type": content_type
                },
                Conditions=[
                    {"Content-Type": content_type},
                    ["content-length-range", 1, 5 * 1024 * 1024 * 1024]  # Max 5GB
                ],
                ExpiresIn=expiration
            )
            
            return {
                "url": presigned_post["url"],
                "fields": presigned_post["fields"],
                "bucket": bucket,
                "key": key,
                "content_type": content_type,
                "expiration_seconds": expiration
            }
        except Exception as e:
            raise Exception(f"Error generating presigned upload URL: {str(e)}")
    
    def upload_file(
        self,
        file_content: bytes,
        bucket_name: Optional[str] = None,
        key: str = "",
        content_type: str = "application/octet-stream"
    ) -> Dict[str, Any]:
        """
        Upload file content to S3.
        
        Args:
            file_content: File content as bytes
            bucket_name: Name of the bucket (defaults to configured bucket)
            key: S3 object key (file path)
            content_type: MIME type of the file
        
        Returns:
            Dictionary with upload result
        """
        if not self.api_configured or not self.s3_client:
            raise Exception("AWS credentials not configured.")
        
        bucket = bucket_name or self.default_bucket
        if not bucket:
            raise Exception("No bucket name provided and no default bucket configured.")
        
        if not key:
            raise Exception("Object key (file path) is required.")
        
        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=file_content,
                ContentType=content_type
            )
            
            # Generate CDN URL if available
            cdn_url = None
            try:
                cdn_url = self.get_cdn_url(key)
            except:
                pass
            
            # Generate presigned URL
            presigned_url = self.get_presigned_url(bucket_name=bucket, key=key, expiration=86400)
            
            return {
                "bucket": bucket,
                "key": key,
                "content_type": content_type,
                "size": len(file_content),
                "cdn_url": cdn_url,
                "presigned_url": presigned_url
            }
        except Exception as e:
            raise Exception(f"Error uploading file: {str(e)}")


# Create a singleton instance
try:
    s3_service = S3Service()
except Exception as e:
    print(f"Warning: S3 service initialization failed: {e}")
    s3_service = None

