from typing import List, Dict, Any, AsyncGenerator, Optional
from pathlib import Path
import json
from app.dto.schemas import ChatMessage, WebSearchMode
from app.core.chat_service.chat_service_openai import chat_service as openai_chat_service
from app.core.chat_service.chat_service_gemini import chat_service_gemini
from app.core.chat_service.chat_service_groq import chat_service_groq


class UnifiedChatService:
    """
    Unified chat service that routes requests to the appropriate provider
    (OpenAI, Google Gemini, or Groq) based on the model name.
    """
    
    def __init__(self):
        """Initialize the unified chat service with all providers"""
        self.openai_service = openai_chat_service
        self.gemini_service = chat_service_gemini
        self.groq_service = chat_service_groq
        
        # Load models config to map custom names to base models
        self.models_config = self._load_models_config()
        self.model_mapping = {}
        for model in self.models_config:
            self.model_mapping[model["id"]] = {
                "base_model": model.get("base_model", model["id"]),
                "provider": model.get("provider", "openai")
            }
        
        # Initialize RAG service (may be None if not configured)
        try:
            from app.core.rag_service import rag_service
            self.rag_service = rag_service
        except Exception:
            self.rag_service = None
    
    def _load_models_config(self) -> List[Dict]:
        """Load models configuration"""
        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "models_config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get("models", [])
        except Exception as e:
            print(f"Warning: Could not load models config: {e}")
            return []
    
    def _resolve_model(self, model: str) -> tuple:
        """
        Resolve custom model name to base model and provider.
        
        Args:
            model: Model identifier (could be custom name like "Sanad" or base model)
        
        Returns:
            tuple: (base_model, provider, persona_name)
        """
        # Check if it's a custom model name
        if model in self.model_mapping:
            mapping = self.model_mapping[model]
            return mapping["base_model"], mapping["provider"], model
        
        # Otherwise, determine provider from model name
        model_lower = model.lower()
        if model_lower.startswith("gemini") or model_lower.startswith("learnlm"):
            return model, "gemini", None
        elif model_lower.startswith("llama") or "groq" in model_lower:
            return model, "groq", None
        else:
            return model, "openai", None
    
    def _is_gemini_model(self, model: str) -> bool:
        """Determine if the model is a Gemini model"""
        _, provider, _ = self._resolve_model(model)
        return provider == "gemini"
    
    def _get_service(self, model: str):
        """
        Get the appropriate service based on the model name.
        
        Args:
            model (str): Model identifier
        
        Returns:
            ChatService: OpenAI, Gemini, or Groq chat service
        """
        _, provider, _ = self._resolve_model(model)
        if provider == "gemini":
            return self.gemini_service
        elif provider == "groq":
            return self.groq_service
        else:
            return self.openai_service
    
    def _handle_rag_retrieval(
        self,
        messages: List[ChatMessage],
        use_rag: bool,
        rag_top_k: int,
        rag_filter_metadata: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None  # Add provider to know if we need to limit content
    ) -> tuple:
        """
        Handle RAG retrieval from vector database.
        
        Args:
            messages: List of conversation messages
            use_rag: Whether to use RAG
            rag_top_k: Number of chunks to retrieve
        
        Returns:
            Tuple of (enhanced_messages, rag_results)
        """
        if not use_rag or not self.rag_service:
            return messages, None
        
        # Get the last user message for RAG query
        last_user_message = None
        last_user_idx = None
        for idx, msg in enumerate(reversed(messages)):
            if msg.role == "user":
                last_user_message = msg.content
                last_user_idx = len(messages) - 1 - idx
                break
        
        if not last_user_message:
            return messages, None
        
        try:
            # Search for similar content in vector database
            rag_results = self.rag_service.search_similar(
                query=last_user_message,
                top_k=rag_top_k,
                filter_metadata=rag_filter_metadata
            )
            
            if not rag_results:
                # Check if it's because no indexes were selected (empty list from RAG service)
                # vs actual search that returned no results
                if rag_filter_metadata is None or not rag_filter_metadata.get("index_name"):
                    # No indexes selected - return special indicator
                    return messages, "NO_RESULTS"
                else:
                    # Indexes were selected but search returned empty - also NO_RESULTS
                    return messages, "NO_RESULTS"
            
            # Build context from RAG results - show which page/chunk is relevant
            # IMPORTANT: This content was retrieved from Pinecone vector database
            context_parts = []
            context_parts.append("=== CONTEXT RETRIEVED FROM PINECONE VECTOR DATABASE ===\n")
            context_parts.append("[CRITICAL] The following information was RETRIEVED from the Pinecone vector database by matching the user's question against stored document embeddings.\n")
            context_parts.append("This is REAL content from uploaded documents stored in Pinecone.\n")
            context_parts.append(f"Total chunks retrieved: {len(rag_results)}\n")
            context_parts.append("IMPORTANT: Use the retrieved context below to answer the question. Even if the similarity scores are moderate, these are the BEST matching documents found in the database.\n")
            context_parts.append("Combine the retrieved information with your knowledge to provide a comprehensive answer. If the retrieved content is relevant (even partially), use it.\n\n")
            
            # Group by document to show page/chunk info clearly
            for idx, result in enumerate(rag_results, 1):
                filename = result.get("filename", "Unknown")
                page_number = result.get("page_number", 1)
                chunk_number = result.get("chunk_number", 1)
                content = result.get("content", "")
                similarity = result.get("similarity_score", 0)
                index_name = result.get("index_name", "unknown")
                
                # Show similarity score interpretation
                # Note: For cosine similarity, scores 0.3-0.5 can still be relevant matches
                if similarity > 0.7:
                    quality = "HIGH relevance"
                elif similarity > 0.5:
                    quality = "MEDIUM relevance"
                elif similarity > 0.3:
                    quality = "MODERATE relevance - use if content is related"
                else:
                    quality = "LOW relevance - review carefully"
                
                # For Groq models, limit content length to prevent token limit issues
                if provider == "groq":
                    # Limit each chunk to ~500 tokens (roughly 2000 chars) to keep total manageable
                    MAX_CHUNK_CHARS = 2000
                    if len(content) > MAX_CHUNK_CHARS:
                        content = content[:MAX_CHUNK_CHARS] + "\n[... content truncated for token limit ...]"
                
                context_parts.append(f"[PINECONE RETRIEVAL {idx}: Index='{index_name}', Document='{filename}', Page {page_number}, Chunk {chunk_number}]")
                context_parts.append(f"Similarity Score: {similarity:.4f} ({quality})")
                context_parts.append(f"Content from Pinecone:\n{content}")
                context_parts.append("\n---\n")
            
            context_parts.append("\n=== END OF PINECONE RETRIEVED CONTEXT ===\n")
            context_parts.append("\n[INSTRUCTIONS]:")
            context_parts.append("1. The above content was RETRIEVED from Pinecone vector database - it is REAL document content")
            context_parts.append("2. Use the retrieved content as the PRIMARY source for your answer")
            context_parts.append("3. Even if similarity scores are moderate (0.3-0.5), these are the BEST matches found - use them if they're relevant to the question")
            context_parts.append("4. Combine retrieved information with your general knowledge to provide a complete answer")
            context_parts.append("5. When using retrieved content, mention the document name (e.g., 'According to نظام ملكية الوحدات العقارية...')")
            context_parts.append("6. If the retrieved content is relevant to the question (even if not a perfect match), use it to answer")
            context_parts.append("7. Only say 'insufficient information' if the retrieved content is completely unrelated to the question\n")
            context_text = "\n".join(context_parts)
            
            # Enhance the last user message with RAG context
            enhanced_content = f"{context_text}\n\nUser Question: {last_user_message}"
            
            # Create new messages list with enhanced content
            enhanced_messages = messages.copy()
            enhanced_messages[last_user_idx] = ChatMessage(
                role="user",
                content=enhanced_content
            )
            
            return enhanced_messages, rag_results
        
        except Exception as e:
            # If RAG fails, log error but continue without RAG
            print(f"RAG retrieval error: {str(e)}")
            return messages, None
    
    async def stream_chat_completion(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1000,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        top_k: int = None,
        web_search_mode: WebSearchMode = WebSearchMode.DISABLED,
        use_rag: bool = True,
        rag_top_k: int = 5,
        rag_filter_metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion using the appropriate provider.
        Yields SSE-formatted strings.
        
        Args:
            model: Model identifier (determines provider)
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            frequency_penalty: Frequency penalty (-2.0 to 2.0) - OpenAI only
            presence_penalty: Presence penalty (-2.0 to 2.0) - OpenAI only
            top_k: Top-k sampling parameter - Gemini only
            web_search_mode: Web search mode (off, on, if_needed)
            use_rag: Whether to use RAG retrieval from uploaded documents
            rag_top_k: Number of document chunks to retrieve
        
        Yields:
            str: SSE-formatted response chunks
        """
        # Resolve model to get provider early (needed for RAG content limiting)
        base_model, provider, persona_name = self._resolve_model(model)
        
        # Handle RAG retrieval if enabled
        messages, rag_results = self._handle_rag_retrieval(messages, use_rag, rag_top_k, rag_filter_metadata, provider=provider)
        
        # If RAG is enabled but no indexes selected or no results found
        if use_rag and rag_results == "NO_RESULTS":
            import json
            # Check if it's because no indexes were selected
            if rag_filter_metadata is None or not rag_filter_metadata.get("index_name"):
                error_message = "⚠️ RAG is enabled but no indexes selected. Please select at least one index (Qadha, Contracts, etc.) in the RAG search section below, or disable RAG to continue without document search."
                error_message_en = error_message
                error_type = "NO_INDEXES_SELECTED"
            else:
                error_message = "لم يتم العثور على أي وثائق ذات صلة في قاعدة البيانات. يرجى التأكد من اختيار الفهارس الصحيحة (contracts أو qadha) أو تعديل السؤال."
                error_message_en = "No relevant documents found in the database. Please ensure you have selected the correct indexes (contracts or qadha) or modify your question."
                error_type = "NO_RAG_RESULTS"
            
            # Send error as SSE event but continue with model response (don't return)
            error_event = {
                "event": "error",
                "data": json.dumps({
                    "error": error_message,
                    "error_en": error_message_en,
                    "type": error_type
                })
            }
            yield f"event: {error_event['event']}\ndata: {error_event['data']}\n\n"
            
            # Continue without RAG - let the model answer without document context
            # Don't return - allow the model to respond
        
        # Send RAG results as first event if available
        if rag_results and rag_results != "NO_RESULTS":
            import json
            # Prepare RAG results for display (include all metadata for debugging/access)
            rag_display_results = []
            for result in rag_results:
                # Get all metadata from Pinecone
                metadata = result.get("metadata", {})
                
                # Extract S3 information from metadata (same as rag_database.py)
                s3_bucket = metadata.get("s3_bucket")
                s3_key = metadata.get("s3_key")
                
                # Build S3 file path/URL if both bucket and key are available
                s3_path = None
                s3_url = None
                if s3_bucket and s3_key:
                    s3_path = f"s3://{s3_bucket}/{s3_key}"
                    s3_url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_key}"
                
                rag_display_results.append({
                    "filename": result.get("filename", "Unknown"),
                    "index_name": result.get("index_name", "unknown"),
                    "page_number": result.get("page_number", 1),
                    "chunk_number": result.get("chunk_number", 1),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "chunk_id": result.get("chunk_id", None),
                    "similarity_score": result.get("similarity_score", 0),
                    "classification": result.get("classification", None),
                    "content": result.get("content", ""),
                    "content_preview": (result.get("content", "") or "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
                    "s3_location": {
                        "bucket": s3_bucket,
                        "key": s3_key,
                        "s3_path": s3_path,
                        "s3_url": s3_url
                    },
                    "metadata": {
                        "file_name": result.get("filename", "Unknown"),
                        "primary_category": metadata.get("primary_category", "unknown"),
                        "classifications": metadata.get("classifications", []),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "total_chunks": metadata.get("total_chunks"),
                        "s3_bucket": s3_bucket,
                        "s3_key": s3_key
                    }
                })
            
            rag_event = {
                "rag": {
                    "used": True,
                    "chunks_retrieved": len(rag_results),
                    "source": "Pinecone Vector Database",
                    "results": rag_display_results
                }
            }
            print(f"RAG: Sending {len(rag_results)} retrieved chunks to client. Source: Pinecone")
            # Use ensure_ascii=False to preserve Arabic characters (not Unicode escape sequences)
            yield f"data: {json.dumps(rag_event, ensure_ascii=False)}\n\n"
        
        service = self._get_service(model)
        
        # For Groq models, limit request size to stay within TPM limits (6000 tokens)
        # Note: messages already include RAG context at this point
        if provider == "groq":
            from app.core.token_counter import token_counter
            # Estimate prompt tokens (messages already include RAG context)
            groq_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            estimated_prompt_tokens = token_counter.count_tokens_openai(groq_messages, base_model)
            
            # Groq free tier limit: 6000 tokens per minute (TPM)
            # Reserve some tokens for prompt, limit max_tokens accordingly
            GROQ_TPM_LIMIT = 6000
            SAFETY_MARGIN = 200  # Reserve 200 tokens for safety (prompt + response overhead)
            max_allowed_tokens = GROQ_TPM_LIMIT - estimated_prompt_tokens - SAFETY_MARGIN
            
            # If max_tokens would exceed limit, reduce it
            if max_tokens > max_allowed_tokens:
                if max_allowed_tokens > 0:
                    original_max_tokens = max_tokens
                    max_tokens = max_allowed_tokens
                    print(f"[Groq] Reduced max_tokens from {original_max_tokens} to {max_tokens} to stay within TPM limit (prompt: {estimated_prompt_tokens} tokens)")
                else:
                    # If prompt itself is too large, truncate messages
                    # Keep system message and last few messages, but preserve RAG context if present
                    if len(messages) > 1:
                        # Find system message
                        system_msg = None
                        user_msgs = []
                        for msg in messages:
                            if msg.role == "system":
                                system_msg = msg
                            elif msg.role == "user":
                                user_msgs.append(msg)
                        
                        # Keep system message and last user message (which may contain RAG context)
                        if system_msg:
                            messages = [system_msg] + user_msgs[-1:] if user_msgs else [system_msg]
                        else:
                            messages = user_msgs[-1:] if user_msgs else messages[-1:]
                        
                        # Re-estimate
                        groq_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
                        estimated_prompt_tokens = token_counter.count_tokens_openai(groq_messages, base_model)
                        max_allowed_tokens = GROQ_TPM_LIMIT - estimated_prompt_tokens - SAFETY_MARGIN
                        max_tokens = max(max_allowed_tokens, 100) if max_allowed_tokens > 0 else 100
                        print(f"[Groq] Truncated messages due to large prompt. New prompt: {estimated_prompt_tokens} tokens, max_tokens: {max_tokens}")
        
        # Route to appropriate service with relevant parameters
        if provider == "gemini":
            # Gemini service
            async for chunk in service.stream_chat_completion(
                model=base_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                top_k=top_k,
                web_search_mode=web_search_mode,
                persona_name=persona_name
            ):
                yield chunk
        elif provider == "groq":
            # Groq service
            async for chunk in service.stream_chat_completion(
                model=base_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                web_search_mode=web_search_mode,
                persona_name=persona_name
            ):
                yield chunk
        else:
            # OpenAI service
            async for chunk in service.stream_chat_completion(
                model=base_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                web_search_mode=web_search_mode,
                persona_name=persona_name
            ):
                yield chunk
    
    def get_chat_completion(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1000,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        top_k: int = None,
        web_search_mode: WebSearchMode = WebSearchMode.DISABLED,
        use_rag: bool = True,
        rag_top_k: int = 5,
        rag_filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get non-streaming chat completion using the appropriate provider.
        
        Args:
            model: Model identifier (determines provider)
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            frequency_penalty: Frequency penalty (-2.0 to 2.0) - OpenAI only
            presence_penalty: Presence penalty (-2.0 to 2.0) - OpenAI only
            top_k: Top-k sampling parameter - Gemini only
            web_search_mode: Web search mode (off, on, if_needed)
            use_rag: Whether to use RAG retrieval from uploaded documents
            rag_top_k: Number of document chunks to retrieve
        
        Returns:
            dict: Chat completion response in OpenAI-compatible format
        """
        # Resolve model to get provider early (needed for RAG content limiting)
        base_model, provider, persona_name = self._resolve_model(model)
        
        # Handle RAG retrieval if enabled
        messages, rag_results = self._handle_rag_retrieval(messages, use_rag, rag_top_k, rag_filter_metadata, provider=provider)
        
        # If RAG is enabled but no indexes selected or no results found
        rag_error = None
        if use_rag and rag_results == "NO_RESULTS":
            # Check if it's because no indexes were selected
            if rag_filter_metadata is None or not rag_filter_metadata.get("index_name"):
                rag_error = "⚠️ RAG is enabled but no indexes selected. Please select at least one index (Qadha, Contracts, etc.) in the RAG search section, or disable RAG to continue without document search."
            else:
                rag_error = "No relevant documents found in the database. Please ensure you have selected the correct indexes (contracts or qadha) or modify your question."
            
            # Continue without RAG - let the model answer without document context
            # Don't return error response - allow the model to respond
        
        service = self._get_service(model)
        
        # For Groq models, limit request size to stay within TPM limits (6000 tokens)
        # Note: messages already include RAG context at this point
        if provider == "groq":
            from app.core.token_counter import token_counter
            # Estimate prompt tokens (messages already include RAG context)
            groq_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            estimated_prompt_tokens = token_counter.count_tokens_openai(groq_messages, base_model)
            
            # Groq free tier limit: 6000 tokens per minute (TPM)
            # Reserve some tokens for prompt, limit max_tokens accordingly
            GROQ_TPM_LIMIT = 6000
            SAFETY_MARGIN = 200  # Reserve 200 tokens for safety (prompt + response overhead)
            max_allowed_tokens = GROQ_TPM_LIMIT - estimated_prompt_tokens - SAFETY_MARGIN
            
            # If max_tokens would exceed limit, reduce it
            if max_tokens > max_allowed_tokens:
                if max_allowed_tokens > 0:
                    original_max_tokens = max_tokens
                    max_tokens = max_allowed_tokens
                    print(f"[Groq] Reduced max_tokens from {original_max_tokens} to {max_tokens} to stay within TPM limit (prompt: {estimated_prompt_tokens} tokens)")
                else:
                    # If prompt itself is too large, truncate messages
                    # Keep system message and last few messages, but preserve RAG context if present
                    if len(messages) > 1:
                        # Find system message
                        system_msg = None
                        user_msgs = []
                        for msg in messages:
                            if msg.role == "system":
                                system_msg = msg
                            elif msg.role == "user":
                                user_msgs.append(msg)
                        
                        # Keep system message and last user message (which may contain RAG context)
                        if system_msg:
                            messages = [system_msg] + user_msgs[-1:] if user_msgs else [system_msg]
                        else:
                            messages = user_msgs[-1:] if user_msgs else messages[-1:]
                        
                        # Re-estimate
                        groq_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
                        estimated_prompt_tokens = token_counter.count_tokens_openai(groq_messages, base_model)
                        max_allowed_tokens = GROQ_TPM_LIMIT - estimated_prompt_tokens - SAFETY_MARGIN
                        max_tokens = max(max_allowed_tokens, 100) if max_allowed_tokens > 0 else 100
                        print(f"[Groq] Truncated messages due to large prompt. New prompt: {estimated_prompt_tokens} tokens, max_tokens: {max_tokens}")
        
        # Route to appropriate service with relevant parameters
        if provider == "gemini":
            # Gemini service
            result = service.get_chat_completion(
                model=base_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                top_k=top_k,
                web_search_mode=web_search_mode,
                persona_name=persona_name
            )
        elif provider == "groq":
            # Groq service
            result = service.get_chat_completion(
                model=base_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                web_search_mode=web_search_mode,
                persona_name=persona_name
            )
        else:
            # OpenAI service
            result = service.get_chat_completion(
                model=base_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                web_search_mode=web_search_mode,
                persona_name=persona_name
            )
        
        # Add RAG information to response if used
        if rag_results and rag_results != "NO_RESULTS":
            # Prepare RAG results for display (include all metadata for debugging/access)
            rag_display_results = []
            for result_item in rag_results:
                # Get all metadata from Pinecone
                metadata = result_item.get("metadata", {})
                
                # Extract S3 information from metadata (same as rag_database.py)
                s3_bucket = metadata.get("s3_bucket")
                s3_key = metadata.get("s3_key")
                
                # Build S3 file path/URL if both bucket and key are available
                s3_path = None
                s3_url = None
                if s3_bucket and s3_key:
                    s3_path = f"s3://{s3_bucket}/{s3_key}"
                    s3_url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_key}"
                
                rag_display_results.append({
                    "filename": result_item.get("filename", "Unknown"),
                    "index_name": result_item.get("index_name", "unknown"),
                    "page_number": result_item.get("page_number", 1),
                    "chunk_number": result_item.get("chunk_number", 1),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "chunk_id": result_item.get("chunk_id", None),
                    "similarity_score": result_item.get("similarity_score", 0),
                    "classification": result_item.get("classification", None),
                    "content": result_item.get("content", ""),
                    "content_preview": (result_item.get("content", "") or "")[:200] + "..." if len(result_item.get("content", "")) > 200 else result_item.get("content", ""),
                    "s3_location": {
                        "bucket": s3_bucket,
                        "key": s3_key,
                        "s3_path": s3_path,
                        "s3_url": s3_url
                    },
                    "metadata": {
                        "file_name": result_item.get("filename", "Unknown"),
                        "primary_category": metadata.get("primary_category", "unknown"),
                        "classifications": metadata.get("classifications", []),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "total_chunks": metadata.get("total_chunks"),
                        "s3_bucket": s3_bucket,
                        "s3_key": s3_key
                    }
                })
            
            result["rag"] = {
                "used": True,
                "chunks_retrieved": len(rag_results),
                "source": "Pinecone Vector Database",
                "results": rag_display_results
            }
            print(f"RAG: Returning {len(rag_results)} retrieved chunks in response. Source: Pinecone")
        else:
            # RAG not used or disabled
            result["rag"] = {
                "used": use_rag,
                "chunks_retrieved": 0,
                "results": None
            }
            if use_rag:
                if rag_error:
                    result["rag"]["error"] = rag_error
                else:
                    result["rag"]["error"] = "No documents retrieved from Pinecone"
        
        return result
    
    def is_api_configured(self, model: str) -> bool:
        """
        Check if the API for the given model is configured.
        
        Args:
            model: Model identifier
        
        Returns:
            bool: True if API is configured, False otherwise
        """
        service = self._get_service(model)
        return service.api_key_configured


# Create a singleton instance
unified_chat_service = UnifiedChatService()

