from pydantic import BaseModel, Field, validator
from typing import Optional, List, Any, Literal, Dict
from enum import Enum


class WebSearchMode(str, Enum):
    """Web search mode options"""
    DISABLED = "disabled"
    FAST = "fast"
    DEEP = "deep"

class MessageResponse(BaseModel):
    """Generic message response schema"""
    message: str = Field(..., description="Response message")

class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")

class DocumentContent(BaseModel):
    """Document content structure"""
    headers: List[str] = Field(default_factory=list, description="Headers and titles found")
    main_text: List[str] = Field(default_factory=list, description="Main content paragraphs")
    tables: List[dict] = Field(default_factory=list, description="Tables found in the document")
    lists: List[str] = Field(default_factory=list, description="Lists found in the document")
    legal_references: List[str] = Field(default_factory=list, description="Legal references and article numbers")
    dates: List[str] = Field(default_factory=list, description="Dates found in the document")
    numbers: List[str] = Field(default_factory=list, description="Important numbers found")
    names_places: List[str] = Field(default_factory=list, description="Names and places mentioned")

class OCRResponse(BaseModel):
    """OCR response schema for single image/document"""
    raw_text: str = Field(..., description="Complete raw text extraction from the document")
    page_number: Optional[int] = Field(None, description="Page number (for PDF processing)")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    
    # Legacy fields for backward compatibility (kept empty for API compatibility)
    document_type: Optional[str] = Field(None, description="Type of document (legacy)")
    language: Optional[str] = Field(None, description="Document language (legacy)")
    title: Optional[str] = Field(None, description="Document title (legacy)")
    content: Optional[DocumentContent] = Field(None, description="Structured document content (legacy)")
    columns: List[str] = Field(default_factory=list, description="Column headers (legacy)")
    data: List[List[str]] = Field(default_factory=list, description="Table data (legacy)")

class PDFOCRResponse(BaseModel):
    """OCR response schema for PDF with multiple pages"""
    pages: List[OCRResponse] = Field(..., description="OCR results for each page")
    total_pages: int = Field(..., description="Total number of pages processed")
    success_pages: int = Field(..., description="Number of pages processed successfully")
    failed_pages: int = Field(..., description="Number of pages that failed to process")


# ===== Chat Schemas =====

class ChatMessage(BaseModel):
    """Chat message schema"""
    role: Literal["system", "user", "assistant"] = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")


class ModelPricing(BaseModel):
    """Model pricing information"""
    input: float = Field(..., description="Price per million tokens for input")
    cached_input: Optional[float] = Field(None, description="Price per million tokens for cached input")
    output: float = Field(..., description="Price per million tokens for output")


class ModelContextLength(BaseModel):
    """Model context length limits"""
    input: int = Field(..., description="Maximum input tokens")
    output: int = Field(..., description="Maximum output tokens")


class ModelInfo(BaseModel):
    """Model information schema"""
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Model name")
    pricing: ModelPricing = Field(..., description="Model pricing details")
    context_length: ModelContextLength = Field(..., description="Model context length limits")


class ListModelsResponse(BaseModel):
    """Response schema for list models endpoint"""
    models: List[ModelInfo] = Field(..., description="List of available models")
    total: int = Field(..., description="Total number of models")


class ChatRequest(BaseModel):
    """Chat request schema"""
    model: str = Field(..., description="Model to use for chat completion")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    max_tokens: int = Field(default=1000, ge=1, le=128000, description="Maximum tokens to generate")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty (OpenAI only)")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty (OpenAI only)")
    top_k: Optional[int] = Field(default=None, ge=1, le=100, description="Top-k sampling parameter (Gemini only)")
    stream: bool = Field(default=True, description="Whether to stream the response")
    web_search_mode: WebSearchMode = Field(
        default=WebSearchMode.DISABLED, 
        description="Web search mode: 'disabled' (no search), 'fast' (search when helpful), 'deep' (always search)"
    )
    use_rag: bool = Field(
        default=True,
        description="Whether to use RAG (Retrieval Augmented Generation) - searches uploaded documents for relevant context"
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve from RAG when use_rag is enabled"
    )
    rag_filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional metadata filters for RAG query. Three types of searches:\n\n"
            "1. USER FILES ONLY: {'user_id': 'nosa', 'filename': ['9th_bci_conference_085.pdf']} - Searches user-uploaded documents in 'user-index-document'\n"
            "2. DATABASE ONLY: {'index_name': ['qadha', 'contracts'], 'classifications': ['sharia_procedures', 'commercial_law']} - Searches pre-loaded legal databases\n"
            "3. BOTH USER FILES + DATABASE: {'user_id': 'nosa', 'filename': ['my_contract.pdf'], 'index_name': ['qadha', 'regulations'], 'classifications': ['commercial_law']} - Searches both user files AND database indexes\n\n"
            "For classifications, you can use:\n"
            "- IDs (recommended): 'sharia_procedures', 'commercial_law', 'administrative_law'\n"
            "- Arabic names: 'نظام المرافعات الشرعية', 'النظام التجاري'\n"
            "- English names: 'Sharia Litigation Procedures Law', 'Commercial Law'\n\n"
            "Use GET /api/v1/rag/database/classifications to see all available classifications with IDs and translations."
        ),
        examples=[
            {"user_id": "nosa", "filename": ["9th_bci_conference_085.pdf"]},
            {"index_name": ["precedents_diwan", "precedents_moj"], "classifications": ["sharia_procedures", "commercial_law"]},
            {"user_id": "nosa", "filename": ["my_contract.pdf"], "index_name": ["qadha", "regulations"], "classifications": ["commercial_law"]}
        ]
    )


class ChatCompletionChunk(BaseModel):
    """SSE chunk for streaming chat completion"""
    id: str = Field(..., description="Unique identifier for the chunk")
    model: str = Field(..., description="Model used")
    choices: List[dict] = Field(..., description="List of completion choices")
    created: int = Field(..., description="Unix timestamp of creation")


class ChatCompletionResponse(BaseModel):
    """Non-streaming chat completion response"""
    id: str = Field(..., description="Unique identifier for the completion")
    model: str = Field(..., description="Model used")
    choices: List[dict] = Field(..., description="List of completion choices")
    created: int = Field(..., description="Unix timestamp of creation")
    usage: dict = Field(..., description="Token usage information")


# ===== RAG Schemas =====

class DocumentUploadResponse(BaseModel):
    """Response schema for document upload"""
    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of the file")
    pages_count: int = Field(..., description="Number of pages in the document")
    chunks_count: int = Field(..., description="Number of text chunks created (divided by tokens)")
    total_text_length: int = Field(..., description="Total length of extracted text")
    index_name: str = Field(..., description="Pinecone index name where the document was stored")
    created_at: str = Field(..., description="ISO timestamp of creation")


class SimilarChunk(BaseModel):
    """Schema for a similar document chunk"""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    document_id: Optional[str] = Field(None, description="Document ID this chunk belongs to")
    filename: Optional[str] = Field(None, description="Original filename")
    page_number: int = Field(..., description="Page number in the document")
    chunk_number: int = Field(..., description="Chunk number on this page")
    content: str = Field(..., description="Chunk content text")
    similarity_score: Optional[float] = Field(None, description="Similarity score (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Chunk metadata")


class RAGQueryRequest(BaseModel):
    """Request schema for RAG query"""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    min_similarity: Optional[float] = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0, 
        description="Minimum similarity score threshold (0-1). Results below this will be filtered out. Default 0.0 (no filtering)."
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters for Pinecone query (e.g., {'index_name': 'contracts', 'classification': 'النظام الإداري'})"
    )


class RAGFilteredQueryRequest(BaseModel):
    """Request schema for RAG query with explicit index and classification filters"""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    min_similarity: Optional[float] = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0, 
        description="Minimum similarity score threshold (0-1). Results below this will be filtered out. Default 0.0 (no filtering)."
    )
    index_names: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of indexes to search in. You can use:\n"
            "- IDs (recommended): ['regulations', 'precedents_diwan', 'orders_circulars']\n"
            "- Technical names: ['qadha', 'contracts', 'bog', 'new-ncar', 'namazig']\n"
            "- Arabic names: ['سوابق قضائية (ديوان المظالم)', 'أنظمة / لوائح']\n"
            "- English names: ['Judicial Precedents (Board of Grievances)', 'Regulations / Bylaws']\n"
            "Use GET /api/v1/rag/database/indexes to see all available options."
        ),
        examples=[
            ["regulations", "precedents_diwan"],
            ["qadha", "contracts"],
            ["سوابق قضائية (ديوان المظالم)", "أنظمة / لوائح"]
        ]
    )
    classifications: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of classification categories to filter by. You can use:\n"
            "- IDs (recommended): ['administrative_law', 'civil_transactions', 'commercial_law']\n"
            "- Arabic names: ['النظام الإداري', 'نظام المعاملات المدنية', 'النظام التجاري']\n"
            "- English names: ['Administrative Law', 'Civil Transactions Law', 'Commercial Law']\n"
            "Use GET /api/v1/rag/database/classifications to see all available options."
        ),
        examples=[
            ["sharia_procedures", "commercial_law"],
            ["النظام الإداري", "نظام المعاملات المدنية"]
        ]
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User ID to filter results by. Only returns documents uploaded by this user. If not provided, searches all users' documents."
    )
    filenames: Optional[List[str]] = Field(
        default=None,
        description="List of filenames to filter by. Only returns results from these specific files. Requires user_id to be set."
    )


class RAGQueryResponse(BaseModel):
    """Response schema for RAG query"""
    query: str = Field(..., description="Original query")
    results: List[SimilarChunk] = Field(..., description="List of similar chunks")
    total_results: int = Field(..., description="Total number of results")


class DocumentInfo(BaseModel):
    """Schema for document information"""
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: Optional[str] = Field(None, description="Original filename")
    content_type: Optional[str] = Field(None, description="MIME type")
    created_at: Optional[str] = Field(None, description="ISO timestamp of creation")
    chunks_count: int = Field(..., description="Number of chunks in the document")


class DocumentListResponse(BaseModel):
    """Response schema for listing documents"""
    documents: List[DocumentInfo] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")


class DocumentDetailResponse(BaseModel):
    """Response schema for document details"""
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: Optional[str] = Field(None, description="Original filename")
    content_type: Optional[str] = Field(None, description="MIME type")
    chunks: List[Dict[str, Any]] = Field(..., description="List of document chunks")
    total_chunks: int = Field(..., description="Total number of chunks")


# ===== Speech Schemas =====

class TextToSpeechRequest(BaseModel):
    """Request schema for text-to-speech"""
    text: str = Field(..., description="Text to convert to speech", min_length=1)
    model: str = Field(default="tts-1", description="TTS model to use (tts-1 or tts-1-hd)")
    voice: str = Field(
        default="alloy", 
        description="Voice to use (alloy, echo, fable, onyx, nova, shimmer)"
    )
    response_format: str = Field(
        default="mp3", 
        description="Audio format (mp3, opus, aac, flac)"
    )
    speed: float = Field(
        default=1.0, 
        ge=0.25, 
        le=4.0, 
        description="Speed of the generated audio (0.25 to 4.0)"
    )


class SpeechToTextRequest(BaseModel):
    """Request schema for speech-to-text (form data will be used, this is for documentation)"""
    language: Optional[str] = Field(
        None, 
        description="Optional language code (e.g., 'en', 'ar', 'es')"
    )
    prompt: Optional[str] = Field(
        None, 
        description="Optional text prompt to guide the model"
    )
    response_format: str = Field(
        default="json", 
        description="Response format (json, text, srt, verbose_json, vtt)"
    )
    temperature: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0, 
        description="Sampling temperature (0.0 to 1.0)"
    )


class SpeechToTextResponse(BaseModel):
    """Response schema for speech-to-text"""
    text: str = Field(..., description="Transcribed text")
    language: Optional[str] = Field(None, description="Detected language code")
    duration: Optional[float] = Field(None, description="Duration of the audio in seconds")
    words: Optional[List[Dict[str, Any]]] = Field(None, description="Word-level timestamps (if available)")
    segments: Optional[List[Dict[str, Any]]] = Field(None, description="Segment-level timestamps (if available)")
    format: Optional[str] = Field(None, description="Response format used")


# ===== Chat Title Schemas =====

class ChatTitleRequest(BaseModel):
    """Request schema for generating chat title"""
    messages: List[ChatMessage] = Field(..., min_items=2, max_items=2, description="Exactly 2 messages (typically first user message and first assistant response)")


class ChatTitleResponse(BaseModel):
    """Response schema for chat title generation"""
    title: str = Field(..., description="Generated chat title (2-4 words)")
    model_used: str = Field(..., description="Model used to generate the title")
    
    class Config:
        protected_namespaces = ()


# ===== Magic Tools Schemas =====

class FileRequest(BaseModel):
    """File request schema"""
    list_of_files: List[str] = Field(default_factory=list, description="List of file URLs or file identifiers")


class TranslationType(str, Enum):
    """Translation type options"""
    GENERAL = "general"
    LEGAL = "legal"
    ACADEMIC = "academic"


class TranslationRequest(BaseModel):
    """Request schema for translation"""
    lang: str = Field(..., description="Target language code (e.g., 'en', 'ar', 'fr')")
    text: str = Field(..., description="Text to translate", min_length=1)
    type: TranslationType = Field(..., description="Type of translation: general, legal, or academic")
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files. If provided, returns DOC file URLs instead of streaming text")


class SummarizationType(str, Enum):
    """Summarization type options"""
    BRIEF = "brief"
    DETAILED = "detailed"


class SummarizationRequest(BaseModel):
    """Request schema for summarization"""
    text: str = Field(..., description="Text to summarize", min_length=1)
    type: SummarizationType = Field(..., description="Type of summarization: brief or detailed")
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files. If provided, returns DOC file URLs instead of streaming text")


class AnalysisType(str, Enum):
    """Analysis type options"""
    BRIEF = "brief"
    DETAILED = "detailed"


class AnalysisRequest(BaseModel):
    """Request schema for text analysis"""
    text: str = Field(..., description="Text to analyze", min_length=1)
    type: AnalysisType = Field(..., description="Type of analysis: brief or detailed")
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files. If provided, returns DOC file URLs instead of streaming text")


class RephrasingRequest(BaseModel):
    """Request schema for text rephrasing"""
    text: str = Field(..., description="Text to rephrase", min_length=1)
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files. If provided, returns DOC file URLs instead of streaming text")


class LegalMemoGeneratorRequest(BaseModel):
    """Request schema for legal memo generator - مولد المذكرات القانونية"""
    text: str = Field(..., description="Input text to generate legal memo from", min_length=1)
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files")
    voice: Optional[str] = Field(default=None, description="Voice input URL (if applicable)")
    memo_type: Optional[str] = Field(default=None, description="Type of memo: دفاع, ادعاء, etc.")


class LawsuitPetitionDraftRequest(BaseModel):
    """Request schema for lawsuit petition drafting - صياغة لائحة دعوى"""
    text: str = Field(..., description="Input text to draft lawsuit petition from", min_length=1)
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files")
    voice: Optional[str] = Field(default=None, description="Voice input URL (if applicable)")
    input_method: Optional[str] = Field(default=None, description="Input method: تعبئة نموذج, رفع مستند سابق, كتابة الوقائع يدوياً")
    case_type: Optional[str] = Field(default=None, description="Case type: تجارية, مدنية, إدارية, etc.")
    court: Optional[str] = Field(default=None, description="Court name")
    parties: Optional[str] = Field(default=None, description="Parties description")
    facts: Optional[str] = Field(default=None, description="Case facts")
    requests: Optional[str] = Field(default=None, description="Relief requested")


class JudgmentAnalysisRequest(BaseModel):
    """Request schema for judgment analysis - تحليل الأحكام"""
    text: str = Field(..., description="Court ruling or judgment text to analyze", min_length=1)
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files")
    voice: Optional[str] = Field(default=None, description="Voice input URL (if applicable)")
    analysis_type: Optional[str] = Field(default=None, description="Analysis type: تحليلي مفصل, موجز, etc.")


class LegalArticleExplanationRequest(BaseModel):
    """Request schema for legal article explanation - شرح المواد القانونية"""
    text: str = Field(..., description="Legal article or text to explain", min_length=1)
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files")
    voice: Optional[str] = Field(default=None, description="Voice input URL (if applicable)")
    explanation_level: Optional[str] = Field(default=None, description="Explanation level: مبسط, قانوني, مع أمثلة قضائية")


class JudgmentComparisonRequest(BaseModel):
    """Request schema for judgment comparison - مقارنة الأحكام"""
    text: str = Field(..., description="Text describing judgments to compare (at least 2 judgments required)", min_length=1)
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files")
    voice: Optional[str] = Field(default=None, description="Voice input URL (if applicable)")
    comparison_criteria: Optional[str] = Field(default=None, description="Comparison criteria: الكل, الوقائع, التسبيب, المنطوق, etc.")


class LegalSummaryRequest(BaseModel):
    """Request schema for legal summary - ملخص قانوني"""
    text: str = Field(..., description="Legal document, case, or regulation text to summarize", min_length=1)
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files")
    voice: Optional[str] = Field(default=None, description="Voice input URL (if applicable)")
    summary_type: Optional[str] = Field(default=None, description="Summary type: ملخص تنفيذي, نقاط رئيسية, لاستخدامه في مذكرة")


class PrecedentSearchRequest(BaseModel):
    """Request schema for precedent search - باحث السوابق"""
    text: str = Field(..., description="Search query for laws, regulations, or legal precedents", min_length=1)
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files")
    voice: Optional[str] = Field(default=None, description="Voice input URL (if applicable)")
    court_type: Optional[str] = Field(default=None, description="Court type filter: وزارة العدل, ديوان المظالم, etc.")
    keywords: Optional[List[str]] = Field(default=None, description="Keywords for search")


class CaseEvaluationRequest(BaseModel):
    """Request schema for case evaluation - تقييم القضايا"""
    text: str = Field(..., description="Case details or facts to evaluate", min_length=1)
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files")
    voice: Optional[str] = Field(default=None, description="Voice input URL (if applicable)")
    case_type: Optional[str] = Field(default=None, description="Case type: تجارية, مدنية, إدارية, etc.")
    evaluation_purpose: Optional[str] = Field(default=None, description="Evaluation purpose: قبل رفع الدعوى, أثناء التقاضي, etc.")


class RegulatoryUpdatesRequest(BaseModel):
    """Request schema for regulatory updates - مستجدات الأنظمة"""
    text: str = Field(..., description="Topic or area for regulatory updates", min_length=1)
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files")
    voice: Optional[str] = Field(default=None, description="Voice input URL (if applicable)")
    legal_domain: Optional[str] = Field(default=None, description="Legal domain: تجاري, مدني, إداري, etc.")


class LegalGroundsRequest(BaseModel):
    """Request schema for legal grounds - الإسناد القانوني"""
    text: str = Field(..., description="Legal issue or question to provide legal grounds for", min_length=1)
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files")
    voice: Optional[str] = Field(default=None, description="Voice input URL (if applicable)")


class FileExtractionRequest(BaseModel):
    """Request schema for file content extraction"""
    file_url: str = Field(..., description="URL or link to the file to extract content from")


class LanguageDetectionRequest(BaseModel):
    """Request schema for language detection"""
    text: str = Field(..., description="Text to detect language from", min_length=1)


class LanguageDetectionResponse(BaseModel):
    """Response schema for language detection"""
    success: bool = Field(..., description="Success status")
    detected_language: str = Field(..., description="Detected ISO 639-3 language code or 'none' if not in supported list")
    language_name_en: Optional[str] = Field(None, description="Language name in English (null if none)")
    language_name_ar: Optional[str] = Field(None, description="Language name in Arabic (null if none)")
    confidence: str = Field(..., description="Confidence level (always 'high' for Gemini)")
    note: Optional[str] = Field(None, description="Additional notes if language was detected but not supported")


class LanguageInfo(BaseModel):
    """Language information schema"""
    code: str = Field(..., description="ISO 639-3 three-letter language code")
    name_en: str = Field(..., description="Language name in English")
    name_ar: str = Field(..., description="Language name in Arabic")


class MagicToolsTypesResponse(BaseModel):
    """Response schema for available magic tools types"""
    translation_types: List[str] = Field(..., description="Available translation types")
    summarization_types: List[str] = Field(..., description="Available summarization types")
    analysis_types: List[str] = Field(..., description="Available analysis types")
    supported_file_types: List[str] = Field(..., description="Supported file types for content extraction")
    supported_languages: List[LanguageInfo] = Field(..., description="Supported languages for translation")


class MagicToolsErrorResponse(BaseModel):
    """Error response schema for magic tools"""
    success: bool = Field(False, description="Success status")
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[str] = Field(None, description="Additional error details")


class MagicToolsSuccessResponse(BaseModel):
    """Success response schema for magic tools"""
    success: bool = Field(True, description="Success status")
    message: Optional[str] = Field(None, description="Success message")


class FileResponse(BaseModel):
    """Response schema for file URL returns"""
    success: bool = Field(True, description="Success status")
    file_url: str = Field(..., description="URL of the generated DOC file")
    file_type: str = Field(default="doc", description="File type (always 'doc' for DOC format)")
    example: str = Field(..., description="Example of how the file URL will look") 