"""
Meeting Transcript Summarization Service using Google Gemini.
Provides brief and detailed meeting summaries with key points extraction.
"""

from google import genai
from google.genai import types
import asyncio
import json
from typing import AsyncGenerator, Dict, Any, Optional, List
from app.config.settings import settings


class MeetingSummaryService:
    """Service for summarizing meeting transcripts using Google Gemini"""
    
    def __init__(self):
        """Initialize the meeting summary service with Google Gemini API configuration"""
        self.api_key_configured = bool(settings.GOOGLE_API_KEY)
        
        if self.api_key_configured:
            self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        else:
            self.client = None
            print("Warning: GOOGLE_API_KEY not configured. Meeting summary will not work.")
    
    def _build_brief_summary_prompt(self, transcript: str) -> str:
        """Build prompt for brief meeting summary"""
        return f"""You are an expert meeting analyst. Summarize the following meeting transcript in a brief, concise manner.

INSTRUCTIONS:
- Keep the summary short (3-5 sentences maximum)
- Focus on the main topics discussed and key outcomes
- Highlight any important decisions made
- Mention critical action items if any
- Use clear and professional language
- Do not add information not present in the transcript

MEETING TRANSCRIPT:
{transcript}

BRIEF SUMMARY:"""

    def _build_detailed_summary_prompt(self, transcript: str) -> str:
        """Build prompt for detailed meeting summary"""
        return f"""You are an expert meeting analyst. Provide a comprehensive summary of the following meeting transcript.

INSTRUCTIONS:
- Create a thorough summary organized into clear sections
- Include these sections:
  1. **Meeting Overview**: Brief description of what the meeting was about
  2. **Topics Discussed**: Main subjects covered during the meeting
  3. **Key Points**: Important points and highlights from the discussion
  4. **Decisions Made**: Any decisions or agreements reached
  5. **Action Items**: Tasks assigned with responsible parties (if mentioned)
  6. **Next Steps**: Follow-up actions or future meeting plans
- Use bullet points for clarity
- Maintain professional language
- Do not add information not present in the transcript

MEETING TRANSCRIPT:
{transcript}

DETAILED SUMMARY:"""

    def _build_key_points_prompt(self, transcript: str) -> str:
        """Build prompt for extracting key points from meeting"""
        return f"""Extract the key points from this meeting transcript. Return them as a JSON array of strings.

INSTRUCTIONS:
- Extract 5-10 most important points
- Each point should be a concise sentence
- Focus on decisions, action items, and important discussions
- Return ONLY a valid JSON array, no other text

MEETING TRANSCRIPT:
{transcript}

KEY POINTS (JSON array):"""

    async def summarize_transcript_stream(
        self,
        transcript: str,
        summary_type: str = "brief",
        model: str = "gemini-2.5-flash"
    ) -> AsyncGenerator[str, None]:
        """
        Summarize meeting transcript with streaming.
        
        Args:
            transcript: The meeting transcript text
            summary_type: Type of summary (brief, detailed)
            model: Gemini model to use
        
        Yields:
            SSE-formatted strings with summary chunks
        """
        if not self.api_key_configured:
            error_data = {
                "success": False,
                "error": "Google API key not configured. Please set GOOGLE_API_KEY environment variable.",
                "error_code": "CONFIGURATION_ERROR"
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        if not transcript or not transcript.strip():
            error_data = {
                "success": False,
                "error": "Transcript cannot be empty",
                "error_code": "EMPTY_TRANSCRIPT"
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        try:
            # Build the appropriate prompt
            if summary_type == "brief":
                prompt = self._build_brief_summary_prompt(transcript)
            else:
                prompt = self._build_detailed_summary_prompt(transcript)
            
            # Configure generation settings
            generation_config = types.GenerateContentConfig(
                max_output_tokens=4096,  # More tokens for detailed summaries
                temperature=0.3,  # Lower temperature for more focused summaries
                top_p=0.95,
                top_k=40
            )
            
            # Generate summary
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=generation_config
            )
            
            # Extract summary text
            summary_text = ""
            if hasattr(response, "text"):
                summary_text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                summary_text += part.text
            
            if not summary_text:
                error_data = {
                    "success": False,
                    "error": "No summary generated",
                    "error_code": "SUMMARIZATION_FAILED"
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Stream word by word for a nicer experience
            words = summary_text.split(' ')
            
            for i, word in enumerate(words):
                chunk_text = word if i == len(words) - 1 else word + ' '
                
                chunk_data = {
                    "success": True,
                    "type": "meeting_summary",
                    "content": chunk_text,
                    "summary_type": summary_type
                }
                
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)  # Small delay for streaming effect
            
            # Send completion marker
            final_data = {
                "success": True,
                "type": "meeting_summary",
                "completed": True,
                "summary_type": summary_type
            }
            yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Meeting summary error: {str(e)}")
            print(f"Traceback: {error_trace}")
            
            error_data = {
                "success": False,
                "error": str(e),
                "error_code": "SUMMARIZATION_ERROR"
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    
    async def summarize_transcript(
        self,
        transcript: str,
        summary_type: str = "brief",
        extract_key_points: bool = True,
        model: str = "gemini-2.5-flash"
    ) -> Dict[str, Any]:
        """
        Summarize meeting transcript (non-streaming).
        
        Args:
            transcript: The meeting transcript text
            summary_type: Type of summary (brief, detailed)
            extract_key_points: Whether to also extract key points
            model: Gemini model to use
        
        Returns:
            Dictionary with summary result
        """
        if not self.api_key_configured:
            raise Exception("Google API key not configured. Please set GOOGLE_API_KEY environment variable.")
        
        if not transcript or not transcript.strip():
            raise Exception("Transcript cannot be empty")
        
        try:
            # Build the appropriate prompt
            if summary_type == "brief":
                prompt = self._build_brief_summary_prompt(transcript)
            else:
                prompt = self._build_detailed_summary_prompt(transcript)
            
            # Configure generation settings
            generation_config = types.GenerateContentConfig(
                max_output_tokens=4096,
                temperature=0.3,
                top_p=0.95,
                top_k=40
            )
            
            # Generate summary
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=generation_config
            )
            
            # Extract summary text
            summary_text = ""
            if hasattr(response, "text"):
                summary_text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                summary_text += part.text
            
            if not summary_text:
                raise Exception("No summary generated")
            
            result = {
                "success": True,
                "type": "meeting_summary",
                "summary": summary_text,
                "summary_type": summary_type
            }
            
            # Optionally extract key points
            if extract_key_points:
                key_points = await self._extract_key_points(transcript, model)
                result["key_points"] = key_points
            
            return result
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Meeting summary error: {str(e)}")
            print(f"Traceback: {error_trace}")
            raise Exception(f"Meeting summary failed: {str(e)}")
    
    async def _extract_key_points(
        self,
        transcript: str,
        model: str = "gemini-2.5-flash"
    ) -> List[str]:
        """Extract key points from meeting transcript"""
        try:
            prompt = self._build_key_points_prompt(transcript)
            
            generation_config = types.GenerateContentConfig(
                max_output_tokens=1024,
                temperature=0.2,  # Lower for more deterministic output
                top_p=0.95,
                top_k=40
            )
            
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=generation_config
            )
            
            response_text = ""
            if hasattr(response, "text"):
                response_text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                response_text += part.text
            
            # Try to parse as JSON
            if response_text:
                # Clean up the response (remove markdown code blocks if present)
                response_text = response_text.strip()
                if response_text.startswith("```"):
                    response_text = response_text.split("\n", 1)[1]
                if response_text.endswith("```"):
                    response_text = response_text.rsplit("```", 1)[0]
                response_text = response_text.strip()
                
                try:
                    key_points = json.loads(response_text)
                    if isinstance(key_points, list):
                        return key_points
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract points manually
                    lines = response_text.split("\n")
                    key_points = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith("[") and not line.startswith("]"):
                            # Remove list markers
                            line = line.lstrip("- •*").strip()
                            line = line.lstrip("0123456789.").strip()
                            if line:
                                key_points.append(line)
                    return key_points[:10]  # Limit to 10 points
            
            return []
            
        except Exception as e:
            print(f"Error extracting key points: {str(e)}")
            return []
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get the current status of the meeting summary service"""
        return {
            "gemini_available": self.api_key_configured,
            "summary_types": ["brief", "detailed"],
            "model": "gemini-2.5-flash"
        }


# Create singleton instance
meeting_summary_service = MeetingSummaryService()

