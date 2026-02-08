from groq import Groq, AsyncGroq
import json
import time
import uuid
from typing import List, Dict, Any, AsyncGenerator, Optional
from app.config.settings import settings
from app.dto.schemas import ChatMessage, WebSearchMode
from app.core.web_search_service import web_search_service
from app.core.token_counter import token_counter
from app.core.system_prompts import get_system_prompt_for_model


class ChatServiceGroq:
    """Service for handling chat operations using Groq API (fast Llama inference)"""
    
    def __init__(self):
        """Initialize the chat service with Groq API configuration"""
        self.api_key_configured = bool(settings.GROQ_API_KEY)
        
        if self.api_key_configured:
            self.client = Groq(api_key=settings.GROQ_API_KEY)
            self.async_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        else:
            self.client = None
            self.async_client = None
        
        self.last_search_results = None  # Store last web search results
    
    def _convert_messages_to_groq_format(self, messages: List[ChatMessage], persona_name: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Convert chat messages to Groq format (OpenAI-compatible).
        
        Args:
            messages: List of chat messages
            persona_name: Optional persona name (Sanad, Rashed, Nora) to use specific system prompt
        """
        groq_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        # Check if there's already a system message
        has_system_message = any(msg["role"] == "system" for msg in groq_messages)
        
        # Only add persona system prompt if:
        # 1. persona_name is explicitly provided, AND
        # 2. there's no existing system message
        if persona_name and not has_system_message:
            system_instruction = get_system_prompt_for_model(persona_name)
            groq_messages.insert(0, {"role": "system", "content": system_instruction})
        
        # If system message already exists, leave it as-is
        # Don't force persona prompts when a custom system prompt is provided
        
        return groq_messages
    
    def _handle_web_search(
        self, 
        groq_messages: List[Dict[str, str]], 
        web_search_mode: WebSearchMode
    ) -> List[Dict[str, str]]:
        """
        Handle web search based on the search mode.
        
        Args:
            groq_messages: List of messages in Groq format
            web_search_mode: Web search mode (off, on, if_needed)
        
        Returns:
            Updated messages list with search results if applicable
        """
        # Reset search results
        self.last_search_results = None
        
        # If web search is off, return messages as-is
        if web_search_mode == WebSearchMode.DISABLED:
            return groq_messages
        
        # Get the last user message
        last_user_message = None
        for msg in reversed(groq_messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        if not last_user_message:
            return groq_messages
        
        # Determine if we should perform search
        # FAST and DEEP both always search, but use different models
        should_search = web_search_mode in [WebSearchMode.FAST, WebSearchMode.DEEP]
        
        # Perform search if needed
        if should_search:
            search_results = web_search_service.search(last_user_message, search_mode=web_search_mode)
            
            if search_results:
                # Store search results for inclusion in response
                self.last_search_results = search_results
                
                # Enhance the last user message with search results
                for msg in reversed(groq_messages):
                    if msg["role"] == "user":
                        msg["content"] = web_search_service.enhance_prompt_with_search(
                            last_user_message, 
                            search_results
                        )
                        break
        
        return groq_messages
    
    async def stream_chat_completion(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1000,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        web_search_mode: WebSearchMode = WebSearchMode.DISABLED,
        persona_name: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion using Groq API.
        Yields SSE-formatted strings.
        """
        if not self.api_key_configured:
            error_data = {
                "error": {
                    "message": "Groq API API key not configured. Please set GROQ_API_KEY environment variable.",
                    "type": "configuration_error"
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        try:
            # Convert messages to Groq format
            groq_messages = self._convert_messages_to_groq_format(messages, persona_name)
            
            # Handle web search if enabled
            groq_messages = self._handle_web_search(groq_messages, web_search_mode)
            
            # Send web search results as first event if available
            if self.last_search_results:
                search_event = {
                    "web_search": {
                        "used": True,
                        "results": self.last_search_results
                    }
                }
                yield f"data: {json.dumps(search_event, ensure_ascii=False)}\n\n"
            
            # Calculate prompt tokens
            prompt_tokens = token_counter.count_tokens_openai(groq_messages, model)
            
            # Create streaming completion using Groq's async API
            # Based on: https://console.groq.com/docs/text-chat
            stream = await self.async_client.chat.completions.create(
                model=model,
                messages=groq_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=True
            )
            
            # Track completion tokens and content
            completion_tokens = 0
            full_content = ""
            chunk_id = None
            created_time = None
            
            # Stream the response chunks
            # Groq returns an async iterator of completion deltas
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # Store chunk metadata from first chunk
                    if chunk.id and not chunk_id:
                        chunk_id = chunk.id
                    if chunk.created and not created_time:
                        created_time = chunk.created
                    
                    # Build chunk data in OpenAI-compatible format
                    chunk_data = {
                        "id": chunk.id if chunk.id else chunk_id,
                        "object": "chat.completion.chunk",
                        "created": chunk.created if chunk.created else created_time,
                        "model": chunk.model,
                        "choices": [
                            {
                                "index": choice.index,
                                "delta": {},
                                "finish_reason": choice.finish_reason
                            }
                        ]
                    }
                    
                    # Add content delta if present (Groq streams content in delta.content)
                    if hasattr(choice, 'delta') and choice.delta.content is not None:
                        content = choice.delta.content
                        chunk_data["choices"][0]["delta"]["content"] = content
                        full_content += content
                        # Count tokens for this content chunk
                        completion_tokens += token_counter.count_tokens_for_text(content, model)
                    
                    # Add role delta if present (usually in first chunk)
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'role') and choice.delta.role is not None:
                        chunk_data["choices"][0]["delta"]["role"] = choice.delta.role
                    
                    # Use ensure_ascii=False to preserve Arabic characters
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            
            # Send usage information before [DONE]
            if chunk_id and created_time:
                usage_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": None
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
                yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"
            
            # Send [DONE] marker
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    
    def get_chat_completion(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1000,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        web_search_mode: WebSearchMode = WebSearchMode.DISABLED,
        persona_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get non-streaming chat completion using Groq API.
        """
        if not self.api_key_configured:
            raise Exception("Groq API key not configured. Please set GROQ_API_KEY environment variable.")
        
        try:
            # Convert messages to Groq format
            groq_messages = self._convert_messages_to_groq_format(messages, persona_name)
            
            # Handle web search if enabled
            groq_messages = self._handle_web_search(groq_messages, web_search_mode)
            
            # Create completion
            response = self.client.chat.completions.create(
                model=model,
                messages=groq_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            
            # Format response (OpenAI-compatible format)
            result = {
                "id": response.id,
                "object": response.object,
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # Add web search results if available
            if self.last_search_results:
                result["web_search"] = {
                    "used": True,
                    "results": self.last_search_results
                }
            else:
                result["web_search"] = {
                    "used": False,
                    "results": None
                }
            
            return result
            
        except Exception as e:
            raise Exception(f"Error generating chat completion: {str(e)}")


# Create a singleton instance
chat_service_groq = ChatServiceGroq()

