from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import json
import asyncio
from datetime import datetime
from app.dto.schemas import (
    LegalMemoGeneratorRequest,
    LawsuitPetitionDraftRequest,
    JudgmentAnalysisRequest,
    LegalArticleExplanationRequest,
    JudgmentComparisonRequest,
    LegalSummaryRequest,
    PrecedentSearchRequest,
    CaseEvaluationRequest,
    RegulatoryUpdatesRequest,
    LegalGroundsRequest,
    MagicToolsErrorResponse
)
from app.core.legal_tools.legal_memo_generator_service import legal_memo_generator_service
from app.core.legal_tools.judgment_analysis_service import judgment_analysis_service
from app.core.legal_tools.lawsuit_petition_service import lawsuit_petition_service
from app.core.legal_tools.article_explanation_service import article_explanation_service
from app.core.legal_tools.judgment_comparison_service import judgment_comparison_service
from app.core.legal_tools.legal_summary_service import legal_summary_service
from app.core.legal_tools.precedent_search_service import precedent_search_service
from app.core.legal_tools.case_evaluation_service import case_evaluation_service
from app.core.legal_tools.regulatory_updates_service import regulatory_updates_service
from app.core.legal_tools.legal_grounds_service import legal_grounds_service

router = APIRouter(prefix="/legal-tools", tags=["Legal Tools"])


@router.get("/types", summary="Get Legal Tools Types and Options")
async def get_legal_tools_types():
    """
    Get available types and options for legal tools
    
    Returns information about memo types, case types, courts, etc.
    """
    return JSONResponse(content={
        "success": True,
        "types": {
            "memo_types": legal_memo_generator_service.get_memo_types(),
            "analysis_types": judgment_analysis_service.get_analysis_types(),
            "case_types": lawsuit_petition_service.get_case_types(),
            "courts": lawsuit_petition_service.get_courts(),
            "explanation_levels": article_explanation_service.get_explanation_levels(),
            "comparison_criteria": judgment_comparison_service.get_comparison_criteria(),
            "summary_types": legal_summary_service.get_summary_types(),
            "precedent_court_types": precedent_search_service.get_court_types(),
            "evaluation_case_types": case_evaluation_service.get_case_types(),
            "evaluation_purposes": case_evaluation_service.get_evaluation_purposes(),
            "regulatory_domains": regulatory_updates_service.get_legal_domains(),
            "input_methods": [
                {"id": "form", "name_en": "Fill Form", "name_ar": "تعبئة نموذج"},
                {"id": "upload", "name_en": "Upload Document", "name_ar": "رفع مستند سابق"},
                {"id": "manual", "name_en": "Manual Entry", "name_ar": "كتابة الوقائع يدوياً"}
            ]
        }
    })


@router.post("/legal-memo-generator", summary="Legal Memo Generator")
async def generate_legal_memo(request: LegalMemoGeneratorRequest, raw_request: Request):
    """
    Generate legal memo from text
    
    Output: SSE streaming with generated memo content
    """
    try:
        if not request.text or not request.text.strip():
            error_response = MagicToolsErrorResponse(success=False, error="Text cannot be empty", error_code="EMPTY_TEXT")
            return JSONResponse(status_code=400, content=error_response.dict())
        
        async def memo_generator():
            try:
                start_data = {
                    "success": True,
                    "type": "legal_memo_generator",
                    "event": "started",
                    "memo_type": request.memo_type or "defense"
                }
                yield f"data: {json.dumps(start_data, ensure_ascii=False)}\n\n"
                
                async for chunk in legal_memo_generator_service.generate_memo_stream(
                    text=request.text,
                    memo_type=request.memo_type,
                    file_content=None
                ):
                    if await raw_request.is_disconnected():
                        return
                    
                    chunk_data = {
                        "success": True,
                        "type": "legal_memo_generator",
                        "content": chunk
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
                complete_data = {
                    "success": True,
                    "type": "legal_memo_generator",
                    "completed": True,
                    "metadata": {
                        "memo_type": request.memo_type or "defense",
                        "generated_at": datetime.now().isoformat()
                    }
                }
                yield f"data: {json.dumps(complete_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                
            except asyncio.CancelledError:
                return
            except Exception as e:
                error_data = {
                    "success": False,
                    "type": "legal_memo_generator",
                    "error": str(e),
                    "error_code": "GENERATION_ERROR"
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            memo_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content=MagicToolsErrorResponse(success=False, error=str(e), error_code="LEGAL_MEMO_ERROR").dict())


@router.post("/judgment-analysis", summary="Judgment Analysis")
async def analyze_judgment(request: JudgmentAnalysisRequest, raw_request: Request):
    """
    Analyze court rulings
    
    Output: SSE (Server-Sent Events)
    """
    try:
        if not request.text or not request.text.strip():
            error_response = MagicToolsErrorResponse(success=False, error="Text cannot be empty", error_code="EMPTY_TEXT")
            return JSONResponse(status_code=400, content=error_response.dict())
        
        async def judgment_analysis_generator():
            try:
                start_data = {
                    "success": True,
                    "type": "judgment_analysis",
                    "event": "started",
                    "analysis_type": request.analysis_type or "detailed"
                }
                yield f"data: {json.dumps(start_data, ensure_ascii=False)}\n\n"
                
                async for chunk in judgment_analysis_service.analyze_judgment_stream(
                    text=request.text,
                    analysis_type=request.analysis_type,
                    file_content=None
                ):
                    if await raw_request.is_disconnected():
                        return
                    
                    chunk_data = {
                        "success": True,
                        "type": "judgment_analysis",
                        "content": chunk
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
                complete_data = {
                    "success": True,
                    "type": "judgment_analysis",
                    "completed": True,
                    "metadata": {
                        "analysis_type": request.analysis_type or "detailed",
                        "analyzed_at": datetime.now().isoformat()
                    }
                }
                yield f"data: {json.dumps(complete_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                
            except asyncio.CancelledError:
                return
            except Exception as e:
                error_data = {
                    "success": False,
                    "type": "judgment_analysis",
                    "error": str(e),
                    "error_code": "ANALYSIS_ERROR"
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            judgment_analysis_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content=MagicToolsErrorResponse(success=False, error=str(e), error_code="JUDGMENT_ANALYSIS_ERROR").dict())


@router.post("/lawsuit-petition-draft", summary="Lawsuit Petition Draft")
async def draft_lawsuit_petition(request: LawsuitPetitionDraftRequest, raw_request: Request):
    """
    Draft lawsuit petition with AI
    
    Output: SSE streaming with generated petition content
    """
    try:
        if not request.text or not request.text.strip():
            error_response = MagicToolsErrorResponse(success=False, error="Text cannot be empty", error_code="EMPTY_TEXT")
            return JSONResponse(status_code=400, content=error_response.dict())
        
        async def petition_generator():
            try:
                start_data = {
                    "success": True,
                    "type": "lawsuit_petition_draft",
                    "event": "started",
                    "case_type": request.case_type or "commercial"
                }
                yield f"data: {json.dumps(start_data, ensure_ascii=False)}\n\n"
                
                async for chunk in lawsuit_petition_service.generate_petition_stream(
                    text=request.text,
                    case_type=request.case_type,
                    court=request.court,
                    parties=request.parties,
                    facts=request.facts,
                    requests=request.requests,
                    file_content=None
                ):
                    if await raw_request.is_disconnected():
                        return
                    
                    chunk_data = {
                        "success": True,
                        "type": "lawsuit_petition_draft",
                        "content": chunk
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
                complete_data = {
                    "success": True,
                    "type": "lawsuit_petition_draft",
                    "completed": True,
                    "metadata": {
                        "case_type": request.case_type or "commercial",
                        "court": request.court,
                        "generated_at": datetime.now().isoformat()
                    }
                }
                yield f"data: {json.dumps(complete_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                
            except asyncio.CancelledError:
                return
            except Exception as e:
                error_data = {
                    "success": False,
                    "type": "lawsuit_petition_draft",
                    "error": str(e),
                    "error_code": "PETITION_ERROR"
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            petition_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content=MagicToolsErrorResponse(success=False, error=str(e), error_code="PETITION_DRAFT_ERROR").dict())


@router.post("/legal-article-explanation", summary="Legal Article Explanation")
async def explain_legal_article(request: LegalArticleExplanationRequest, raw_request: Request):
    """
    Explain legal articles with AI
    
    Output: SSE (Server-Sent Events)
    """
    try:
        if not request.text or not request.text.strip():
            error_response = MagicToolsErrorResponse(success=False, error="Text cannot be empty", error_code="EMPTY_TEXT")
            return JSONResponse(status_code=400, content=error_response.dict())
        
        async def explanation_generator():
            try:
                start_data = {
                    "success": True,
                    "type": "legal_article_explanation",
                    "event": "started",
                    "explanation_level": request.explanation_level or "simple"
                }
                yield f"data: {json.dumps(start_data, ensure_ascii=False)}\n\n"
                
                async for chunk in article_explanation_service.explain_article_stream(
                    text=request.text,
                    explanation_level=request.explanation_level,
                    file_content=None
                ):
                    if await raw_request.is_disconnected():
                        return
                    
                    chunk_data = {
                        "success": True,
                        "type": "legal_article_explanation",
                        "content": chunk
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
                complete_data = {
                    "success": True,
                    "type": "legal_article_explanation",
                    "completed": True,
                    "metadata": {
                        "explanation_level": request.explanation_level or "simple",
                        "explained_at": datetime.now().isoformat()
                    }
                }
                yield f"data: {json.dumps(complete_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                
            except asyncio.CancelledError:
                return
            except Exception as e:
                error_data = {
                    "success": False,
                    "type": "legal_article_explanation",
                    "error": str(e),
                    "error_code": "EXPLANATION_ERROR"
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            explanation_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content=MagicToolsErrorResponse(success=False, error=str(e), error_code="ARTICLE_EXPLANATION_ERROR").dict())


@router.post("/judgment-comparison", summary="Judgment Comparison")
async def compare_judgments(request: JudgmentComparisonRequest, raw_request: Request):
    """
    Compare judgments (2 or more) and return structured comparison table
    
    Output: JSON with comparison_table, legal_conclusion, and usage_recommendation
    """
    try:
        if not request.text or not request.text.strip():
            error_response = MagicToolsErrorResponse(success=False, error="Text cannot be empty", error_code="EMPTY_TEXT")
            return JSONResponse(status_code=400, content=error_response.dict())
        
        # Get comparison result from service
        result = await judgment_comparison_service.compare_judgments(
            text=request.text,
            comparison_criteria=request.comparison_criteria,
            file_content=None
        )
        
        # Check for errors
        if "error" in result and result["error"]:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "type": "judgment_comparison",
                    "error": result["error"],
                    "error_code": "COMPARISON_ERROR"
                }
            )
        
        # Return successful response
        return JSONResponse(content={
            "success": True,
            "type": "judgment_comparison",
            "comparison_table": result.get("comparison_table", []),
            "legal_conclusion": result.get("legal_conclusion", ""),
            "usage_recommendation": result.get("usage_recommendation", ""),
            "metadata": {
                "comparison_criteria": request.comparison_criteria or "all",
                "compared_at": datetime.now().isoformat()
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content=MagicToolsErrorResponse(success=False, error=str(e), error_code="JUDGMENT_COMPARISON_ERROR").dict())


@router.post("/legal-summary", summary="Legal Summary")
async def summarize_legal_document(request: LegalSummaryRequest, raw_request: Request):
    """
    Summarize legal documents with AI
    
    Output: JSON with structured summary (executive, key points, or for memo use)
    """
    try:
        if not request.text or not request.text.strip():
            error_response = MagicToolsErrorResponse(success=False, error="Text cannot be empty", error_code="EMPTY_TEXT")
            return JSONResponse(status_code=400, content=error_response.dict())
        
        # Get summary from service
        result = await legal_summary_service.summarize_document(
            text=request.text,
            summary_type=request.summary_type,
            file_content=None
        )
        
        # Check for errors
        if "error" in result and result["error"]:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "type": "legal_summary",
                    "error": result["error"],
                    "error_code": "SUMMARY_ERROR"
                }
            )
        
        # Return successful response
        return JSONResponse(content={
            "success": True,
            "type": "legal_summary",
            **result,
            "metadata": {
                "summary_type": request.summary_type or "executive",
                "original_length": len(request.text),
                "generated_at": datetime.now().isoformat()
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content=MagicToolsErrorResponse(success=False, error=str(e), error_code="LEGAL_SUMMARY_ERROR").dict())


@router.post("/precedent-search", summary="Precedent Search")
async def search_precedents(request: PrecedentSearchRequest, raw_request: Request):
    """
    Search for judicial precedents relevant to a case
    
    Output: JSON with structured precedent results including similarity percentages
    """
    try:
        if not request.text or not request.text.strip():
            error_response = MagicToolsErrorResponse(success=False, error="Text cannot be empty", error_code="EMPTY_TEXT")
            return JSONResponse(status_code=400, content=error_response.dict())
        
        # Get precedents from service
        result = await precedent_search_service.search_precedents(
            text=request.text,
            court_type=request.court_type,
            keywords=request.keywords,
            file_content=None
        )
        
        # Check for errors
        if "error" in result and result["error"]:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "type": "precedent_search",
                    "error": result["error"],
                    "error_code": "PRECEDENT_ERROR"
                }
            )
        
        # Return successful response
        return JSONResponse(content={
            "success": True,
            "type": "precedent_search",
            **result,
            "metadata": {
                "court_filter": request.court_type or "all",
                "searched_at": datetime.now().isoformat()
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content=MagicToolsErrorResponse(success=False, error=str(e), error_code="PRECEDENT_SEARCH_ERROR").dict())


@router.post("/case-evaluation", summary="Case Evaluation")
async def evaluate_case(request: CaseEvaluationRequest, raw_request: Request):
    """
    Evaluate case strength with AI analysis
    
    Output: JSON with overall assessment, success probability, strengths, weaknesses, risks, and recommendations
    """
    try:
        if not request.text or not request.text.strip():
            error_response = MagicToolsErrorResponse(success=False, error="Text cannot be empty", error_code="EMPTY_TEXT")
            return JSONResponse(status_code=400, content=error_response.dict())
        
        # Get evaluation from service
        result = await case_evaluation_service.evaluate_case(
            text=request.text,
            case_type=request.case_type,
            evaluation_purpose=request.evaluation_purpose,
            file_content=None
        )
        
        # Check for errors
        if "error" in result and result["error"]:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "type": "case_evaluation",
                    "error": result["error"],
                    "error_code": "EVALUATION_ERROR"
                }
            )
        
        # Return successful response
        return JSONResponse(content={
            "success": True,
            "type": "case_evaluation",
            **result,
            "metadata": {
                "case_type": request.case_type or "general",
                "evaluation_purpose": request.evaluation_purpose or "pre_filing",
                "evaluated_at": datetime.now().isoformat()
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content=MagicToolsErrorResponse(success=False, error=str(e), error_code="CASE_EVALUATION_ERROR").dict())


@router.post("/regulatory-updates", summary="Regulatory Updates")
async def get_regulatory_updates(request: RegulatoryUpdatesRequest, raw_request: Request):
    """
    Track regulatory and legislative updates in Saudi Arabia
    
    Output: JSON with structured updates including effective dates and practical impact
    """
    try:
        if not request.text or not request.text.strip():
            error_response = MagicToolsErrorResponse(success=False, error="Text cannot be empty", error_code="EMPTY_TEXT")
            return JSONResponse(status_code=400, content=error_response.dict())
        
        # Get updates from service
        result = await regulatory_updates_service.get_updates(
            text=request.text,
            legal_domain=request.legal_domain,
            file_content=None
        )
        
        # Check for errors
        if "error" in result and result["error"]:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "type": "regulatory_updates",
                    "error": result["error"],
                    "error_code": "UPDATES_ERROR"
                }
            )
        
        # Return successful response
        return JSONResponse(content={
            "success": True,
            "type": "regulatory_updates",
            **result,
            "metadata": {
                "legal_domain": request.legal_domain or "commercial",
                "retrieved_at": datetime.now().isoformat()
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content=MagicToolsErrorResponse(success=False, error=str(e), error_code="REGULATORY_UPDATES_ERROR").dict())


@router.post("/legal-grounds", summary="Legal Grounds")
async def present_legal_grounds(request: LegalGroundsRequest, raw_request: Request):
    """
    Link facts to relevant legal provisions, regulations, and precedents
    
    Output: JSON with structured legal grounds organized by source type (Sharia, regulations, bylaws, precedents, etc.)
    """
    try:
        if not request.text or not request.text.strip():
            error_response = MagicToolsErrorResponse(success=False, error="Text cannot be empty", error_code="EMPTY_TEXT")
            return JSONResponse(status_code=400, content=error_response.dict())
        
        # Get legal grounds from service
        result = await legal_grounds_service.find_legal_grounds(
            text=request.text,
            file_content=None
        )
        
        # Check for errors
        if "error" in result and result["error"]:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "type": "legal_grounds",
                    "error": result["error"],
                    "error_code": "GROUNDS_ERROR"
                }
            )
        
        # Return successful response
        return JSONResponse(content={
            "success": True,
            "type": "legal_grounds",
            **result,
            "metadata": {
                "generated_at": datetime.now().isoformat()
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content=MagicToolsErrorResponse(success=False, error=str(e), error_code="LEGAL_GROUNDS_ERROR").dict())
