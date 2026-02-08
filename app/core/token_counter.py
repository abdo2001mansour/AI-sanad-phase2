"""
Token counting utility for different model providers.
Supports OpenAI (tiktoken), Gemini (estimation), and Llama models (transformers).
"""
from typing import List, Dict, Any, Optional
import tiktoken
from app.dto.schemas import ChatMessage

# Try to import transformers for Llama models (optional)
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TokenCounter:
    """Utility class for counting tokens across different model providers"""
    
    # Model to tiktoken encoding mapping
    TIKTOKEN_ENCODINGS = {
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4-turbo-preview": "cl100k_base",
        "gpt-4-0125-preview": "cl100k_base",
        "gpt-4-1106-preview": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        "o1": "o200k_base",
        "o1-preview": "o200k_base",
        "o1-mini": "o200k_base",
        "o1-mini-preview": "o200k_base",
    }
    
    # Llama model tokenizer mapping
    LLAMA_MODELS = {
        "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3-8B-Instruct": "meta-llama/Llama-3-8B-Instruct",
        "meta-llama/Llama-3-70B-Instruct": "meta-llama/Llama-3-70B-Instruct",
    }
    
    def __init__(self):
        """Initialize token counter with encodings"""
        self._tiktoken_encoders = {}
        self._llama_tokenizers = {}
    
    def _get_tiktoken_encoding(self, model: str) -> Optional[tiktoken.Encoding]:
        """Get tiktoken encoding for OpenAI models"""
        # Try to find matching encoding
        encoding_name = None
        for model_prefix, enc in self.TIKTOKEN_ENCODINGS.items():
            if model.startswith(model_prefix):
                encoding_name = enc
                break
        
        # Default to cl100k_base for OpenAI models
        if not encoding_name and (model.startswith("gpt") or model.startswith("o1")):
            encoding_name = "cl100k_base"
        
        if not encoding_name:
            return None
        
        # Cache encodings
        if encoding_name not in self._tiktoken_encoders:
            try:
                self._tiktoken_encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
            except Exception:
                return None
        
        return self._tiktoken_encoders[encoding_name]
    
    def _get_llama_tokenizer(self, model: str):
        """Get tokenizer for Llama models"""
        if not TRANSFORMERS_AVAILABLE:
            return None
        
        # Find matching Llama model
        tokenizer_name = None
        for model_key, tokenizer_key in self.LLAMA_MODELS.items():
            if model.startswith(model_key):
                tokenizer_name = tokenizer_key
                break
        
        # Default to Llama-3.1-8B if it's a meta-llama model
        if not tokenizer_name and model.startswith("meta-llama/"):
            tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"
        
        if not tokenizer_name:
            return None
        
        # Cache tokenizers
        if tokenizer_name not in self._llama_tokenizers:
            try:
                self._llama_tokenizers[tokenizer_name] = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    trust_remote_code=True
                )
            except Exception:
                return None
        
        return self._llama_tokenizers[tokenizer_name]
    
    def count_tokens_openai(self, messages: List[Dict[str, str]], model: str) -> int:
        """
        Count tokens for OpenAI models using tiktoken.
        
        Args:
            messages: List of messages in OpenAI format
            model: Model name
            
        Returns:
            Number of tokens
        """
        encoding = self._get_tiktoken_encoding(model)
        if not encoding:
            # Fallback: estimate ~4 characters per token
            total_chars = sum(len(msg.get("content", "")) for msg in messages)
            return total_chars // 4
        
        # Count tokens according to OpenAI's format
        # Format: <im_start>role\ncontent<im_end>\n
        tokens_per_message = 3  # <im_start>role\n
        tokens_per_name = 1  # name\n
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key == "content":
                    num_tokens += len(encoding.encode(str(value)))
                elif key == "name":
                    num_tokens += tokens_per_name
                    num_tokens += len(encoding.encode(str(value)))
        
        num_tokens += 3  # <im_start>assistant\n
        return num_tokens
    
    def count_tokens_gemini(self, text: str) -> int:
        """
        Estimate tokens for Gemini models.
        Gemini uses SentencePiece tokenizer, roughly 1 token per 2-3 characters.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated number of tokens
        """
        # Gemini tokenizer is roughly 1 token per 2.5 characters
        # This is an approximation
        return len(text) // 2.5 if text else 0
    
    def count_tokens_llama(self, text: str, model: str) -> int:
        """
        Count tokens for Llama models using transformers.
        
        Args:
            text: Text to count tokens for
            model: Model name
            
        Returns:
            Number of tokens
        """
        tokenizer = self._get_llama_tokenizer(model)
        if not tokenizer:
            # Fallback: estimate ~4 characters per token
            return len(text) // 4 if text else 0
        
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception:
            # Fallback: estimate
            return len(text) // 4 if text else 0
    
    def count_tokens_for_messages(
        self, 
        messages: List[ChatMessage], 
        model: str,
        system_instruction: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Count tokens for a list of messages based on the model type.
        
        Args:
            messages: List of ChatMessage objects
            model: Model name
            system_instruction: Optional system instruction text
            
        Returns:
            Dictionary with prompt_tokens count
        """
        # Convert ChatMessage to OpenAI format for counting
        openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        # Add system instruction if provided
        if system_instruction:
            openai_messages.insert(0, {"role": "system", "content": system_instruction})
        
        # Determine model type and count tokens
        if model.startswith("gpt") or model.startswith("o1"):
            # OpenAI models
            prompt_tokens = self.count_tokens_openai(openai_messages, model)
        elif model.startswith("gemini") or model.startswith("learnlm"):
            # Gemini models - convert to text format
            text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
            if system_instruction:
                text = f"{system_instruction}\n\n{text}"
            prompt_tokens = int(self.count_tokens_gemini(text))
        elif model.startswith("meta-llama/"):
            # Llama models - convert to text format
            text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
            if system_instruction:
                text = f"{system_instruction}\n\n{text}"
            prompt_tokens = self.count_tokens_llama(text, model)
        else:
            # Default: estimate
            total_chars = sum(len(msg.content) for msg in messages)
            if system_instruction:
                total_chars += len(system_instruction)
            prompt_tokens = total_chars // 4
        
        return {"prompt_tokens": prompt_tokens}
    
    def count_tokens_for_text(self, text: str, model: str) -> int:
        """
        Count tokens for a single text string based on the model type.
        
        Args:
            text: Text to count
            model: Model name
            
        Returns:
            Number of tokens
        """
        if model.startswith("gpt") or model.startswith("o1"):
            encoding = self._get_tiktoken_encoding(model)
            if encoding:
                return len(encoding.encode(text))
            return len(text) // 4
        elif model.startswith("gemini") or model.startswith("learnlm"):
            return int(self.count_tokens_gemini(text))
        elif model.startswith("meta-llama/"):
            return self.count_tokens_llama(text, model)
        else:
            return len(text) // 4


# Create singleton instance
token_counter = TokenCounter()

