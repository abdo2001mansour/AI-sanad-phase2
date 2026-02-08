from google import genai
from google.genai import types
import json
import time
import uuid
from typing import List, Dict, Any, AsyncGenerator, Optional
from app.config.settings import settings
from app.dto.schemas import ChatMessage, WebSearchMode
from app.core.web_search_service import web_search_service
from app.core.token_counter import token_counter
from app.core.system_prompts import get_system_prompt_for_model


class ChatServiceGemini:
    """Service for handling chat operations using Google Gemini API"""
    
    def __init__(self):
        """Initialize the chat service with Google Gemini API configuration"""
        self.api_key_configured = bool(settings.GOOGLE_API_KEY)
        
        if self.api_key_configured:
            self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        else:
            self.client = None
        
        self.last_search_results = None  # Store last web search results
    
    def _convert_messages_to_gemini_format(self, messages: List[ChatMessage], persona_name: Optional[str] = None) -> str:
        """
        Convert chat messages to Gemini format.
        Returns formatted text content for the Gemini API.
        
        Args:
            messages: List of chat messages
            persona_name: Optional persona name (Sanad, Rashed, Nora) to use specific system prompt
        """
        # Get system prompt based on persona
        system_instruction = get_system_prompt_for_model(persona_name) if persona_name else get_system_prompt_for_model("Nora")
        
        # Add system instruction at the beginning
        conversation_text = f"{system_instruction}\n\n---\n\n"
        
        # Add conversation history
        for msg in messages:
            if msg.role == "system":
                conversation_text += f"[System]: {msg.content}\n\n"
            elif msg.role == "user":
                conversation_text += f"[User]: {msg.content}\n\n"
            elif msg.role == "assistant":
                conversation_text += f"[Assistant]: {msg.content}\n\n"
        
        conversation_text += "[Assistant]: "
        
        return conversation_text
    
    def _handle_web_search(
        self, 
        messages: List[ChatMessage], 
        web_search_mode: WebSearchMode
    ) -> List[ChatMessage]:
        """
        Handle web search based on the search mode.
        
        Args:
            messages: List of chat messages
            web_search_mode: Web search mode (off, on, if_needed)
        
        Returns:
            Updated messages list with search results if applicable
        """
        # Reset search results
        self.last_search_results = None
        
        # If web search is off, return messages as-is
        if web_search_mode == WebSearchMode.DISABLED:
            return messages
        
        # Get the last user message
        last_user_message = None
        last_user_idx = None
        for idx, msg in enumerate(reversed(messages)):
            if msg.role == "user":
                last_user_message = msg.content
                last_user_idx = len(messages) - 1 - idx
                break
        
        if not last_user_message:
            return messages
        
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
                enhanced_content = web_search_service.enhance_prompt_with_search(
                    last_user_message, 
                    search_results
                )
                
                # Create new messages list with enhanced content
                updated_messages = messages.copy()
                updated_messages[last_user_idx] = ChatMessage(
                    role="user",
                    content=enhanced_content
                )
                return updated_messages
        
        return messages
    
    async def stream_chat_completion(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1000,
        top_k: int = None,
        web_search_mode: WebSearchMode = WebSearchMode.DISABLED,
        persona_name: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion using Google Gemini API.
        Yields SSE-formatted strings.
        """
        if not self.api_key_configured:
            error_data = {
                "error": {
                    "message": "Google API key not configured. Please set GOOGLE_API_KEY environment variable.",
                    "type": "configuration_error"
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        try:
            # Handle web search if enabled
            messages = self._handle_web_search(messages, web_search_mode)
            
            # Send web search results as first event if available
            if self.last_search_results:
                search_event = {
                    "web_search": {
                        "used": True,
                        "results": self.last_search_results
                    }
                }
                yield f"data: {json.dumps(search_event, ensure_ascii=False)}\n\n"
            
            # Get system prompt for token counting
            system_instruction = get_system_prompt_for_model(persona_name) if persona_name else get_system_prompt_for_model("Nora")
            
            # Convert messages to Gemini format
            content = self._convert_messages_to_gemini_format(messages, persona_name)
            
            # Calculate prompt tokens
            prompt_tokens_info = token_counter.count_tokens_for_messages(
                messages, 
                model, 
                system_instruction=system_instruction
            )
            prompt_tokens = prompt_tokens_info["prompt_tokens"]
            
            # Build generation config
            config_params = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            if top_k is not None:
                config_params["top_k"] = top_k
            
            generation_config = types.GenerateContentConfig(**config_params)
            
            # Generate unique IDs for streaming
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created_timestamp = int(time.time())
            
            # Send first chunk with role
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"
            
            # WORKAROUND: Use non-streaming and simulate streaming with delays
            # Initialize client_disconnected flag
            client_disconnected = False
            
            try:
                # Get complete response first
                response = self.client.models.generate_content(
                    model=model,
                    contents=content,
                    config=generation_config
                )
                
                # Extract full text
                full_text = ""
                if hasattr(response, "text"):
                    full_text = response.text
                elif hasattr(response, "candidates") and response.candidates:
                    for c in response.candidates:
                        if hasattr(c, "content") and c.content.parts:
                            for part in c.content.parts:
                                if hasattr(part, "text") and part.text:
                                    full_text += part.text
                
                # Simulate streaming by chunking the text
                # Send in word-sized chunks for natural streaming without overwhelming the socket
                if full_text:
                    import asyncio
                    
                    # Split by spaces to send word by word
                    words = full_text.split(' ')
                    
                    for i, word in enumerate(words):
                        if client_disconnected:
                            break
                            
                        # Add space back except for last word
                        chunk_text = word if i == len(words) - 1 else word + ' '
                        
                        chunk_data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_timestamp,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk_text},
                                    "finish_reason": None
                                }
                            ]
                        }
                        
                        try:
                            # Use ensure_ascii=False to preserve Arabic characters
                            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                            # Very small delay to allow network to process
                            await asyncio.sleep(0.001)  # 1ms delay
                        except (BrokenPipeError, ConnectionResetError, Exception):
                            # Client disconnected, stop streaming silently
                            client_disconnected = True
                            break
                        
            except Exception as e:
                print(f"Gemini generation error: {str(e)}")
                pass
            
            # Send final chunk with finish_reason (if client still connected)
            if not client_disconnected:
                try:
                    final_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_timestamp,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }
                        ]
                    }
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                    
                    # Send [DONE] marker
                    yield "data: [DONE]\n\n"
                except (BrokenPipeError, ConnectionResetError, Exception):
                    # Client disconnected, exit silently
                    pass
            
        except Exception as e:
            # Log the error for debugging
            import traceback
            error_trace = traceback.format_exc()
            print(f"Gemini streaming error: {str(e)}")
            print(f"Traceback: {error_trace}")
            
            error_data = {
                "error": {
                    "message": str(e) if str(e) else f"Error type: {type(e).__name__}",
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
        top_k: int = None,
        web_search_mode: WebSearchMode = WebSearchMode.DISABLED,
        persona_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get non-streaming chat completion using Google Gemini API.
        """
        if not self.api_key_configured:
            raise Exception("Google API key not configured. Please set GOOGLE_API_KEY environment variable.")
        
        try:
            # Handle web search if enabled
            messages = self._handle_web_search(messages, web_search_mode)
            
            # Convert messages to Gemini format
            content = self._convert_messages_to_gemini_format(messages, persona_name)
            
            # Build generation config
            config_params = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            if top_k is not None:
                config_params["top_k"] = top_k
            
            generation_config = types.GenerateContentConfig(**config_params)
            
            # Generate completion
            response = self.client.models.generate_content(
                model=model,
                contents=content,
                config=generation_config
            )
            
            # Safely extract text output
            text_output = ""
            if hasattr(response, "text"):
                text_output = response.text
            elif hasattr(response, "candidates") and response.candidates:
                for c in response.candidates:
                    if hasattr(c, "content") and c.content.parts:
                        for part in c.content.parts:
                            if hasattr(part, "text") and part.text:
                                text_output += part.text
            
            # If still no text found, fallback to string cast
            if not text_output:
                text_output = str(response)
            
            # Generate unique ID and timestamp
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created_timestamp = int(time.time())
            
            # Format response in OpenAI-compatible format
            result = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created_timestamp,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text_output
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": getattr(response, "usage_metadata", {}).get("prompt_token_count", 0) if hasattr(response, "usage_metadata") else 0,
                    "completion_tokens": getattr(response, "usage_metadata", {}).get("candidates_token_count", 0) if hasattr(response, "usage_metadata") else 0,
                    "total_tokens": getattr(response, "usage_metadata", {}).get("total_token_count", 0) if hasattr(response, "usage_metadata") else 0
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
            # Log the error for debugging
            import traceback
            error_trace = traceback.format_exc()
            print(f"Gemini completion error: {str(e)}")
            print(f"Traceback: {error_trace}")
            raise Exception(f"Error generating chat completion: {str(e)}")


# Create a singleton instance
chat_service_gemini = ChatServiceGemini()
