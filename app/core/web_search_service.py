from openai import OpenAI
from typing import List, Dict, Any, Optional, Literal
from app.config.settings import settings
from app.dto.schemas import WebSearchMode


class WebSearchService:
    """Service for handling web search operations using Perplexity API"""
    
    def __init__(self):
        """Initialize the web search service with Perplexity API configuration"""
        self.api_key_configured = bool(settings.PERPLEXITY_API_KEY)
        
        if self.api_key_configured:
            # Perplexity uses OpenAI-compatible API
            self.client = OpenAI(
                api_key=settings.PERPLEXITY_API_KEY,
                base_url="https://api.perplexity.ai"
            )
        else:
            self.client = None
        
        self.last_error = None  # Track last error for debugging
    
    def search(self, query: str, search_mode: WebSearchMode = WebSearchMode.FAST) -> Optional[str]:
        """
        Perform web search using Perplexity API.
        
        Args:
            query (str): The search query
            search_mode (WebSearchMode): Search mode - FAST uses faster model, DEEP uses more thorough model
                       Default: WebSearchMode.FAST
        
        Returns:
            str: Search results summary or None if search fails
        """
        if not self.api_key_configured:
            self.last_error = "Perplexity API key not configured"
            return None
        
        # Choose model and system prompt based on search mode
        if search_mode == WebSearchMode.DEEP:
            # Deep search: use more thorough model and detailed prompt
            model = "sonar-deep-research"  # More thorough, comprehensive search
            system_prompt = "You are a comprehensive research assistant. Perform thorough web searches and provide detailed, well-researched information with multiple sources and perspectives. Include relevant context, statistics, and recent developments."
        else:
            # Fast search: use faster, more efficient model
            model = "sonar"  # Faster, more efficient
            system_prompt = "You are a helpful search assistant. Provide concise, accurate information based on web search results. Focus on the most relevant and current information."
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            )
            
            self.last_error = None
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            self.last_error = error_msg
            
            # Log detailed error for debugging
            print(f"Web search error: {error_msg}")
            
            # Check for common errors
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                print("ERROR: Invalid Perplexity API key. Please check your PERPLEXITY_API_KEY in .env file")
                print("Get your API key from: https://www.perplexity.ai/settings/api")
            elif "400" in error_msg or "invalid_model" in error_msg.lower():
                print("ERROR: Invalid Perplexity model name")
                print(f"Tried to use model '{model}'. Available models: 'sonar' (fast) or 'sonar-deep-research' (deep)")
                print("Check: https://docs.perplexity.ai/api-reference/chat-completions")
            elif "404" in error_msg:
                print("ERROR: Invalid Perplexity API endpoint")
            elif "429" in error_msg:
                print("ERROR: Perplexity API rate limit exceeded")
            
            return None
    
    def should_search(self, user_message: str) -> bool:
        """
        Determine if a web search is needed based on the user's message.
        Uses heuristics to detect if the query requires current information.
        
        Args:
            user_message (str): The user's message
        
        Returns:
            bool: True if web search is recommended
        """
        # Keywords that typically indicate need for current information
        search_indicators = [
            # English
            "current", "latest", "recent", "today", "now", "news", 
            "update", "2024", "2025", "this year", "this month",
            "what's happening", "breaking", "price", "stock",
            # Arabic
            "الحالي", "الأخير", "الأخيرة", "اليوم", "الآن", "أخبار",
            "تحديث", "حديث", "جديد", "ما يحدث", "السعر"
        ]
        
        message_lower = user_message.lower()
        
        # Check for search indicators
        for indicator in search_indicators:
            if indicator in message_lower:
                return True
        
        # Check for questions about events, facts, or data
        question_words = ["what", "when", "where", "who", "how much", "ما", "متى", "أين", "من", "كم"]
        if any(word in message_lower for word in question_words):
            # Check if it's asking about specific named entities or current events
            if any(keyword in message_lower for keyword in ["news", "أخبار", "event", "حدث", "price", "سعر"]):
                return True
        
        return False
    
    def enhance_prompt_with_search(self, user_message: str, search_results: str) -> str:
        """
        Enhance the user's prompt with web search results.
        
        Args:
            user_message (str): Original user message
            search_results (str): Web search results
        
        Returns:
            str: Enhanced prompt with search context
        """
        enhanced_prompt = f"""[Web Search Results]:
{search_results}

[User Question]:
{user_message}

Please answer the user's question using the web search results above as context. Provide accurate information based on the search results."""
        
        return enhanced_prompt


# Create a singleton instance
web_search_service = WebSearchService()

