from openai import AsyncOpenAI, OpenAI
import json
import time
import uuid
from typing import List, Dict, Any, AsyncGenerator, Optional
from app.config.settings import settings
from app.dto.schemas import ChatMessage, WebSearchMode
from app.core.web_search_service import web_search_service
from app.core.token_counter import token_counter
from app.core.system_prompts import get_system_prompt_for_model


class ChatService:
    """Service for handling chat operations using OpenAI API"""
    
    def __init__(self):
        """Initialize the chat service with OpenAI API configuration"""
        self.api_key_configured = bool(settings.OPENAI_API_KEY)
        
        if self.api_key_configured:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            self.client = None
            self.async_client = None
        
        self.last_search_results = None  # Store last web search results
    
    def _is_reasoning_model(self, model: str) -> bool:
        """Check if the model is a reasoning/thinking model (o1, o3 series) or GPT-5+ models that use max_completion_tokens"""
        model_lower = model.lower()
        # o1, o3 reasoning models and gpt-5+ models require max_completion_tokens instead of max_tokens
        return model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("gpt-5")
    
    def _convert_messages_to_openai_format(self, messages: List[ChatMessage], persona_name: Optional[str] = None, model: str = "gpt-4o") -> List[Dict[str, str]]:
        """
        Convert chat messages to OpenAI format and prepend system instruction.
        
        Args:
            messages: List of chat messages
            persona_name: Optional persona name (Sanad, Rashed, Nora) to use specific system prompt
            model: Model name (needed to check if it's a reasoning model)
        """
        openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        # Reasoning models (o1, o3) don't support system messages
        # Instead, prepend system instruction as a user message
        if self._is_reasoning_model(model):
            # Get system prompt based on persona
            system_instruction = get_system_prompt_for_model(persona_name) if persona_name else get_system_prompt_for_model("Rashed")
            
            # Remove any existing system messages
            openai_messages = [msg for msg in openai_messages if msg["role"] != "system"]
            
            # Prepend system instruction as part of first user message
            if openai_messages and openai_messages[0]["role"] == "user":
                openai_messages[0]["content"] = f"{system_instruction}\n\n{openai_messages[0]['content']}"
            else:
                # Insert as first user message
                openai_messages.insert(0, {"role": "user", "content": system_instruction})
            
            return openai_messages
        
        # For non-reasoning models, use system message as before
        # Get system prompt based on persona
        system_instruction = get_system_prompt_for_model(persona_name) if persona_name else get_system_prompt_for_model("Rashed")
        
        # Check if there's already a system message
        has_system_message = any(msg["role"] == "system" for msg in openai_messages)
        
        # If no system message exists, prepend the system instruction
        if not has_system_message:
            openai_messages.insert(0, {"role": "system", "content": system_instruction})
        else:
            # If system message exists, enhance it with system instruction
            for msg in openai_messages:
                if msg["role"] == "system":
                    msg["content"] = system_instruction + "\n\n" + msg["content"]
                    break
        
        return openai_messages
    
    def _handle_web_search(
        self, 
        openai_messages: List[Dict[str, str]], 
        web_search_mode: WebSearchMode
    ) -> List[Dict[str, str]]:
        """
        Handle web search based on the search mode.
        
        Args:
            openai_messages: List of messages in OpenAI format
            web_search_mode: Web search mode (off, on, if_needed)
        
        Returns:
            Updated messages list with search results if applicable
        """
        # Reset search results
        self.last_search_results = None
        
        # If web search is off, return messages as-is
        if web_search_mode == WebSearchMode.DISABLED:
            return openai_messages
        
        # Get the last user message
        last_user_message = None
        for msg in reversed(openai_messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        if not last_user_message:
            return openai_messages
        
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
                for msg in reversed(openai_messages):
                    if msg["role"] == "user":
                        msg["content"] = web_search_service.enhance_prompt_with_search(
                            last_user_message, 
                            search_results
                        )
                        break
        
        return openai_messages
    
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
        Stream chat completion using OpenAI API.
        Yields SSE-formatted strings.
        """
        if not self.api_key_configured:
            error_data = {
                "error": {
                    "message": "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.",
                    "type": "configuration_error"
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        try:
            # Convert messages to OpenAI format (pass model to handle reasoning models)
            openai_messages = self._convert_messages_to_openai_format(messages, persona_name, model)
            
            # Handle web search if enabled
            openai_messages = self._handle_web_search(openai_messages, web_search_mode)
            
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
            prompt_tokens = token_counter.count_tokens_openai(openai_messages, model)
            
            # Create streaming completion
            # Reasoning models (o1, o3) have different parameter requirements
            if self._is_reasoning_model(model):
                # o1/o3 models don't support temperature, frequency_penalty, presence_penalty, top_p
                # They only support max_tokens (called max_completion_tokens)
                stream = await self.async_client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
                    max_completion_tokens=max_tokens,  # o1/o3 use max_completion_tokens instead of max_tokens
                    stream=True
                )
            else:
                # Regular models support all parameters
                stream = await self.async_client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stream=True
                )
            
            # Track completion tokens and content
            completion_tokens = 0
            full_content = ""
            chunk_id = None
            created_time = None
            
            # Stream the response chunks
            async for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    
                    # Store chunk metadata
                    if chunk.id:
                        chunk_id = chunk.id
                    if chunk.created:
                        created_time = chunk.created
                    
                    # Build chunk data in OpenAI format
                    chunk_data = {
                        "id": chunk.id,
                        "object": "chat.completion.chunk",
                        "created": chunk.created,
                        "model": chunk.model,
                        "choices": [
                            {
                                "index": choice.index,
                                "delta": {},
                                "finish_reason": choice.finish_reason
                            }
                        ]
                    }
                    
                    # Add content if present
                    if choice.delta.content is not None:
                        content = choice.delta.content
                        chunk_data["choices"][0]["delta"]["content"] = content
                        full_content += content
                        # Count tokens for this content chunk
                        completion_tokens += token_counter.count_tokens_for_text(content, model)
                    
                    # Add role if present (usually in first chunk)
                    if choice.delta.role is not None:
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
        Get non-streaming chat completion using OpenAI API.
        """
        if not self.api_key_configured:
            raise Exception("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
        
        try:
            # Convert messages to OpenAI format (pass model to handle reasoning models)
            openai_messages = self._convert_messages_to_openai_format(messages, persona_name, model)
            
            # Handle web search if enabled
            openai_messages = self._handle_web_search(openai_messages, web_search_mode)
            
            # Create completion
            # Reasoning models (o1, o3) have different parameter requirements
            if self._is_reasoning_model(model):
                # o1/o3 models don't support temperature, frequency_penalty, presence_penalty, top_p
                response = self.client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
                    max_completion_tokens=max_tokens  # o1/o3 use max_completion_tokens
                )
            else:
                # Regular models support all parameters
                response = self.client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )
            
            # Format response (already in OpenAI format)
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
chat_service = ChatService()

