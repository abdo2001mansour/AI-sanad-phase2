from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # App Configuration
    APP_NAME: str = "FastAPI Clean Architecture"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database Configuration
    DATABASE_URL: Optional[str] = None
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AI API Configuration
    GOOGLE_API_KEY: Optional[str] = None  # For OCR service and Gemini chat
    OPENAI_API_KEY: Optional[str] = None  # For OpenAI chat service
    GROQ_API_KEY: Optional[str] = None  # For Groq chat service (fast Llama inference)
    PERPLEXITY_API_KEY: Optional[str] = None  # For web search service
    PINECONE_API_KEY: Optional[str] = None  # For Pinecone vector database
    PINECONE_ENVIRONMENT: Optional[str] = None  # Pinecone environment (e.g., "us-east-1-aws")
    PINECONE_INDEX_NAME: Optional[str] = None  # Pinecone index name (optional - indexes should be specified in UI)
    
    # AWS S3 Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None  # AWS Access Key ID
    AWS_SECRET_ACCESS_KEY: Optional[str] = None  # AWS Secret Access Key
    AWS_REGION: str = "eu-north-1"  # AWS Region
    AWS_S3_BUCKET_NAME: Optional[str] = None  # Default S3 bucket name
    AWS_CDN_URL: Optional[str] = None  # CDN URL for S3 objects
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = "ignore"

# Create settings instance
settings = Settings() 