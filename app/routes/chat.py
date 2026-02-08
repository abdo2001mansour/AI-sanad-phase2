from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import json
import os
from pathlib import Path
from app.core.chat_service.unified_chat_service import unified_chat_service
from app.dto.schemas import (
    ChatRequest,
    ChatCompletionResponse,
    ListModelsResponse,
    ModelInfo,
    ErrorResponse,
    ChatTitleRequest,
    ChatTitleResponse,
    ChatMessage
)
import asyncio

router = APIRouter(prefix="/chat", tags=["Chat"])

# Load models configuration
MODELS_CONFIG_PATH = Path(__file__).parent.parent / "config" / "models_config.json"


def load_models_config():
    """Load models configuration from JSON file"""
    try:
        with open(MODELS_CONFIG_PATH, 'r') as f:
            config = json.load(f)
            return config.get("models", [])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load models configuration: {str(e)}"
        )


def get_openai_model_priority(model_name: str) -> int:
    """Get priority for OpenAI model selection (lower = higher priority)"""
    model_lower = model_name.lower()
    if "gpt-4o-mini" in model_lower:
        return 1
    elif "gpt-4o" in model_lower and "mini" not in model_lower:
        return 2
    elif "gpt-4-turbo" in model_lower:
        return 3
    elif "gpt-3.5" in model_lower:
        return 4
    elif "gpt-4" in model_lower:
        return 5
    else:
        return 10  # Other GPT models


@router.get("/list_models", response_model=ListModelsResponse, summary="List available chat models")
async def list_models():
    """
    Get a list of all available chat models with their pricing and context limits.
    
    Returns information about each model including:
    - Model ID and name
    - Pricing for input, cached input, and output tokens (per million tokens)
    - Maximum context length for input and output
    """
    try:
        models_data = load_models_config()
        
        # Convert to ModelInfo objects (filter out internal fields like base_model and provider)
        models = []
        for model_data in models_data:
            # Only include fields that are in ModelInfo schema
            model_info = {
                "id": model_data["id"],
                "name": model_data["name"],
                "pricing": model_data["pricing"],
                "context_length": model_data["context_length"]
            }
            models.append(ModelInfo(**model_info))
        
        return ListModelsResponse(
            models=models,
            total=len(models)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing models: {str(e)}"
        )


@router.post("/model_sse", summary="Stream chat completion using SSE")
async def model_sse(request: ChatRequest, raw_request: Request):
    """
    Stream chat completion using Server-Sent Events (SSE).
    
    This endpoint accepts a chat request and streams the response in real-time.
    The stream follows the OpenAI-compatible SSE format.
    
    ## Request Parameters:
    
    **Basic Parameters:**
    - `model`: The model to use (Sanad, Rashed, or Nora)
    - `messages`: List of conversation messages with role and content
    - `temperature`: Sampling temperature (0.0 to 2.0)
    - `top_p`: Nucleus sampling parameter (0.0 to 1.0)
    - `max_tokens`: Maximum tokens to generate
    - `stream`: Should be true for SSE streaming
    
    **Model-Specific Parameters:**
    - `frequency_penalty`: Frequency penalty (-2.0 to 2.0) - OpenAI models only
    - `presence_penalty`: Presence penalty (-2.0 to 2.0) - OpenAI models only
    - `top_k`: Top-k sampling (1 to 100) - Gemini models only
    
    **Web Search:**
    - `web_search_mode`: 'disabled' (no search), 'fast' (quick search), 'deep' (thorough search)
    
    **RAG (Retrieval Augmented Generation):**
    - `use_rag`: Enable/disable RAG document search (default: true)
    - `rag_top_k`: Number of document chunks to retrieve (default: 5, max: 20)
    - `rag_filter_metadata`: Filter options for RAG search (see examples below)
    
    ## RAG Filter Examples:
    
    ### Example 1: Search User-Uploaded Files
    ```json
    {
      "model": "Rashed",
      "messages": [{"role": "user", "content": "what is this doc talking about"}],
      "use_rag": true,
      "rag_top_k": 5,
      "rag_filter_metadata": {
        "user_id": "nosa",
        "filename": ["9th_bci_conference_085.pdf"]
      }
    }
    ```
    This searches in the **user-index-document** index for documents uploaded by user "nosa".
    
    ### Example 2: Search Database Indexes (Qadha, Contracts)
    ```json
    {
      "model": "Rashed",
      "messages": [{"role": "user", "content": "ما هي إجراءات رفع الدعوى؟"}],
      "use_rag": true,
      "rag_top_k": 10,
      "rag_filter_metadata": {
        "index_name": ["qadha", "contracts"],
        "classifications": ["sharia_procedures", "administrative_law"]
      }
    }
    ```
    This searches in the **qadha** and **contracts** database indexes, filtering by classification.
    
    ### Example 3: Search Both User Files AND Database
    ```json
    {
      "model": "Rashed",
      "messages": [{"role": "user", "content": "compare this with legal regulations"}],
      "use_rag": true,
      "rag_top_k": 15,
      "rag_filter_metadata": {
        "index_name": ["qadha"],
        "user_id": "nosa",
        "filename": ["my_contract.pdf"]
      }
    }
    ```
    This searches in BOTH the user's uploaded files AND the qadha database index.
    
    ## Response Format:
    
    The response will be a stream of SSE events:
    ```
    data: {"id": "...", "choices": [...], ...}
    
    data: [DONE]
    ```
    
    Each chunk contains:
    - id: Unique completion ID
    - object: "chat.completion.chunk"
    - created: Unix timestamp
    - model: Model used
    - choices: Array with delta content and finish_reason
    """
    try:
        # Validate model exists
        models_data = load_models_config()
        available_models = [m["id"] for m in models_data]
        
        if request.model not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model: {request.model}. Use /api/v1/chat/list_models to see available models."
            )
        
        # Validate messages
        if not request.messages:
            raise HTTPException(
                status_code=400,
                detail="At least one message is required"
            )
        
        # For streaming requests
        if request.stream:
            async def event_generator():
                token_usage = None
                
                try:
                    async for chunk in unified_chat_service.stream_chat_completion(
                        model=request.model,
                        messages=request.messages,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_tokens=request.max_tokens,
                        frequency_penalty=request.frequency_penalty,
                        presence_penalty=request.presence_penalty,
                        top_k=request.top_k,
                        web_search_mode=request.web_search_mode,
                        use_rag=request.use_rag,
                        rag_top_k=request.rag_top_k,
                        rag_filter_metadata=request.rag_filter_metadata
                    ):
                        # Stop immediately if client disconnected
                        try:
                            if await raw_request.is_disconnected():
                                return
                        except Exception:
                            # Best-effort disconnect check
                            pass
                        
                        # Check if this chunk contains usage information
                        if chunk.startswith("data: "):
                            try:
                                chunk_data = json.loads(chunk[6:])  # Remove "data: " prefix
                                if "usage" in chunk_data:
                                    token_usage = chunk_data["usage"]
                            except (json.JSONDecodeError, KeyError):
                                pass
                        
                        try:
                            yield chunk
                        except GeneratorExit:
                            # Client disconnected/refreshed - stop immediately and silently
                            raise
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                            # Client disconnected - stop immediately
                            return
                except asyncio.CancelledError:
                    # Task cancelled (client gone)
                    return
                except Exception as e:
                    # Log error but don't crash
                    print(f"Error in event generator: {e}")
                    return
                
                # After all chunks are sent, send token usage as a separate event for easy display
                if token_usage:
                    try:
                        token_info_event = {
                            "type": "token_usage",
                            "usage": {
                                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                                "completion_tokens": token_usage.get("completion_tokens", 0),
                                "total_tokens": token_usage.get("total_tokens", 0)
                            }
                        }
                        yield f"data: {json.dumps(token_info_event)}\n\n"
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        # Client disconnected, ignore
                        pass
                    except Exception as e:
                        # Log but don't crash
                        print(f"Error sending token usage: {e}")
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Disable nginx buffering
                }
            )
        else:
            # Non-streaming response
            result = unified_chat_service.get_chat_completion(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                top_k=request.top_k,
                web_search_mode=request.web_search_mode,
                use_rag=request.use_rag,
                rag_top_k=request.rag_top_k,
                rag_filter_metadata=request.rag_filter_metadata
            )
            return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except ValueError as e:
        # Handle validation errors (like web_search_mode)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.post("/generate-title", response_model=ChatTitleResponse, summary="Generate a title for a chat conversation")
async def generate_chat_title(request: ChatTitleRequest):
    """
    Generate a concise title for a chat conversation based on the first 2 messages.
    
    This endpoint:
    - Takes exactly 2 messages (typically the first user message and first assistant response)
    - Uses OpenAI models (prefers gpt-4o-mini) to generate a title
    - Returns a short, descriptive title (2-4 words maximum)
    
    Example request:
    {
        "messages": [
            {"role": "user", "content": "What is the legal framework for environmental pollution in Saudi Arabia?"},
            {"role": "assistant", "content": "The legal framework for environmental pollution..."}
        ]
    }
    
    Example response:
    {
        "title": "Environmental Pollution Framework",
        "model_used": "gpt-4o-mini"
    }
    """
    try:
        # Validate messages
        if len(request.messages) != 2:
            raise HTTPException(
                status_code=400,
                detail="Exactly 2 messages are required"
            )
        
        # Check if first message is user and second is assistant (recommended but not strict)
        if request.messages[0].role != "user":
            raise HTTPException(
                status_code=400,
                detail="First message should be from user"
            )
        
        # Find OpenAI models
        models_data = load_models_config()
        
        # Find OpenAI models by checking provider field or base_model name
        openai_models = []
        for model in models_data:
            provider = model.get("provider", "").lower()
            base_model = model.get("base_model", "").lower()
            model_id = model.get("id", "").lower()
            
            # Check if it's an OpenAI model
            if provider == "openai" or base_model.startswith("gpt-") or model_id.startswith("gpt-"):
                base_model_name = model.get("base_model", model["id"])
                openai_models.append({
                    "id": model["id"],
                    "base_model": base_model_name,
                    "priority": get_openai_model_priority(base_model_name)
                })
        
        if not openai_models:
            raise HTTPException(
                status_code=503,
                detail="No OpenAI models available for title generation. Please ensure OpenAI API key is configured and at least one OpenAI model is available."
            )
        
        # Sort by priority (lower number = higher priority) and select best one
        openai_models.sort(key=lambda x: x["priority"])
        model_to_use = openai_models[0]["id"]
        
        # Create a prompt for title generation
        user_message = request.messages[0].content
        assistant_message = request.messages[1].content[:200] if len(request.messages[1].content) > 200 else request.messages[1].content
        
        title_prompt = f"""Generate a concise title for this conversation. The title must be exactly 2 to 4 words maximum. Return only the title, nothing else.

User: {user_message}
Assistant: {assistant_message}

Title (2-4 words only):"""
        
        # Create messages for title generation
        title_messages = [
            ChatMessage(role="system", content="You are a helpful assistant that generates concise, descriptive titles for conversations. The title must be exactly 2 to 4 words. Return only the title, nothing else. Do not include quotes, punctuation, or any other text."),
            ChatMessage(role="user", content=title_prompt)
        ]
        
        # Generate title using unified chat service (non-streaming, no RAG, no web search)
        result = unified_chat_service.get_chat_completion(
            model=model_to_use,
            messages=title_messages,
            temperature=0.3,  # Lower temperature for more consistent titles
            top_p=0.9,
            max_tokens=30,  # Titles should be very short (2-4 words)
            use_rag=False,  # No RAG for title generation
            web_search_mode="disabled"
        )
        
        # Extract title from response
        title = ""
        if result.get("choices") and len(result["choices"]) > 0:
            title = result["choices"][0].get("message", {}).get("content", "").strip()
        
        # Clean up title (remove quotes, extra whitespace, punctuation, etc.)
        title = title.strip('"\'.,;:!?')
        title = ' '.join(title.split())  # Normalize whitespace
        
        # Enforce 2-4 words limit
        words = title.split()
        if len(words) > 4:
            # Take first 4 words if too many
            title = ' '.join(words[:4])
        elif len(words) < 2:
            # If less than 2 words, use first 2-4 words from user message as fallback
            user_words = user_message.split()
            if len(user_words) >= 2:
                title = ' '.join(user_words[:min(4, len(user_words))])
            else:
                title = user_message[:30]  # Last resort fallback
        
        if not title:
            # Final fallback: use first 2-4 words of user message
            words = user_message.split()
            title = ' '.join(words[:min(4, len(words))]) if len(words) >= 2 else user_message[:30]
        
        return ChatTitleResponse(
            title=title,
            model_used=model_to_use
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating chat title: {str(e)}"
        )


@router.get("/health", summary="Chat service health check")
async def chat_health_check():
    """
    Check if the chat service is properly configured and ready to use.
    """
    try:
        from app.config.settings import settings
        
        # Check API keys
        openai_configured = bool(settings.OPENAI_API_KEY)
        google_configured = bool(settings.GOOGLE_API_KEY)
        groq_configured = bool(settings.GROQ_API_KEY)
        
        if not openai_configured and not google_configured and not groq_configured:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "No AI provider API keys configured",
                    "details": "Please set at least one of: OPENAI_API_KEY, GOOGLE_API_KEY, or GROQ_API_KEY in environment variables"
                }
            )
        
        # Try to load models config
        try:
            models_data = load_models_config()
            total_models = len(models_data)
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "Failed to load models configuration",
                    "details": str(e)
                }
            )
        
        # Check Perplexity API key
        web_search_available = bool(settings.PERPLEXITY_API_KEY)
        
        # Count models by provider
        openai_models = len([m for m in models_data if m.get("provider") == "openai" or m["id"].startswith("gpt")])
        gemini_models = len([m for m in models_data if m.get("provider") == "gemini" or m["id"].startswith("gemini") or m["id"].startswith("learnlm")])
        groq_models = len([m for m in models_data if m.get("provider") == "groq" or ("llama" in m["id"].lower() and m.get("provider") == "groq")])
        
        return {
            "status": "healthy",
            "message": "Chat service is ready - Rashid (راشد) Saudi Legal Assistant",
            "total_models": total_models,
            "providers": {
                "openai": {
                    "configured": openai_configured,
                    "models": openai_models
                },
                "gemini": {
                    "configured": google_configured,
                    "models": gemini_models
                },
                "groq": {
                    "configured": groq_configured,
                    "models": groq_models
                }
            },
            "persona": "Rashid - Saudi lawyer with 20+ years experience",
            "languages": ["Arabic", "English"],
            "web_search_available": web_search_available,
            "web_search_modes": ["disabled", "fast", "deep"]
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Chat service error: {str(e)}"
            }
        )

