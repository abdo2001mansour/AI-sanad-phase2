from google import genai
from google.genai import types
from PIL import Image
import json
import os
import tempfile
import io
from typing import List, Dict, Any
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.config.settings import settings


class OCRService:
    """Service for handling OCR operations using Google Gemini AI"""
    
    def __init__(self):
        """Initialize the OCR service with Gemini AI configuration"""
        self.api_key_configured = bool(settings.GOOGLE_API_KEY)
        
        if self.api_key_configured:
            self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
            # Use gemini-2.0-flash (gemini-1.5-flash has been deprecated)
            self.model_name = 'gemini-2.0-flash'
        else:
            self.client = None
            self.model_name = None
        
        # OCR prompt for Saudi Arabian legal documents - raw text only
        self.ocr_prompt = """This is a document related to laws in Saudi Arabia. Please extract ALL text content from this image exactly as it appears.

        Extract everything you can see including:
        - Headers and titles
        - Article numbers and legal references
        - Main content and legal text
        - Tables (preserve table structure with spacing)
        - Lists and numbered items
        - Dates, numbers, and any other information
        - Names, places, and legal terms
        
        Preserve the original formatting and structure as much as possible.
        If it's a table, maintain the column alignment.
        If it's regular text, keep paragraph breaks.
        
        Return the data as a simple JSON object:
        {
            "raw_text": "complete raw text extraction exactly as it appears in the document"
        }

        Extract everything visible in the document exactly as written. Don't miss any text, numbers, or content.
        Return only the JSON object with raw_text field, no additional text or formatting."""

    def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Extract structured data from a single image using Gemini AI.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            dict: Extracted structured data from the image
        """
        if not self.api_key_configured:
            return {
                "raw_text": "",
                "error": "Google API key not configured. Please set GOOGLE_API_KEY environment variable."
            }
        
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Open and load the image
            image = Image.open(image_path)
            
            # Generate content using Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[self.ocr_prompt, image]
            )
            
            return self._parse_gemini_response(response, image_path)
        
        except Exception as e:
            raise Exception(f"Error processing image {image_path}: {str(e)}")

    def extract_text_from_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract structured data from image bytes using Gemini AI.
        
        Args:
            image_bytes (bytes): Image data as bytes
        
        Returns:
            dict: Extracted structured data from the image
        """
        if not self.api_key_configured:
            return {
                "raw_text": "",
                "error": "Google API key not configured. Please set GOOGLE_API_KEY environment variable."
            }
        
        try:
            # Create image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Generate content using Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[self.ocr_prompt, image]
            )
            
            return self._parse_gemini_response(response, "image_bytes")
        
        except Exception as e:
            raise Exception(f"Error processing image bytes: {str(e)}")

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract structured data from all pages of a PDF using Gemini AI.
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            list: List of extracted structured data from each page
        """
        try:
            # Check if file exists
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(pdf_path)
            results = []
            
            for page_num in range(len(pdf_document)):
                try:
                    # Get the page
                    page = pdf_document[page_num]
                    
                    # Convert page to image (PNG format)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                    img_data = pix.tobytes("png")
                    
                    # Create PIL Image from bytes
                    page_image = Image.open(io.BytesIO(img_data))
                    
                    # Generate content using Gemini for each page
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[self.ocr_prompt, page_image]
                    )
                    page_result = self._parse_gemini_response(response, f"{pdf_path}_page_{page_num + 1}")
                    page_result["page_number"] = page_num + 1
                    results.append(page_result)
                    
                except Exception as e:
                    # If a page fails, add error info but continue with other pages
                    results.append({
                        "page_number": page_num + 1,
                        "error": f"Failed to process page {page_num + 1}: {str(e)}",
                        "document_type": "legal_document",
                        "language": "arabic",
                        "content": None,
                        "columns": [],
                        "data": []
                    })
            
            # Close the PDF document
            pdf_document.close()
            return results
        
        except Exception as e:
            raise Exception(f"Error processing PDF {pdf_path}: {str(e)}")

    def _process_single_page_image(self, img_data: bytes, page_num: int) -> Dict[str, Any]:
        """Process a single PDF page image (helper for parallel processing)"""
        try:
            # Create PIL Image from bytes
            page_image = Image.open(io.BytesIO(img_data))
            
            # Generate content using Gemini for this page
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[self.ocr_prompt, page_image]
            )
            page_result = self._parse_gemini_response(response, f"pdf_bytes_page_{page_num + 1}")
            page_result["page_number"] = page_num + 1
            return page_result
            
        except Exception as e:
            # If a page fails, return error info
            return {
                "page_number": page_num + 1,
                "error": f"Failed to process page {page_num + 1}: {str(e)}",
                "document_type": "legal_document",
                "language": "arabic",
                "content": None,
                "columns": [],
                "data": []
            }
    
    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extract structured data from PDF bytes.
        First tries direct text extraction with PyMuPDF, then falls back to OCR for scanned pages.
        
        Args:
            pdf_bytes (bytes): PDF data as bytes
        
        Returns:
            list: List of extracted structured data from each page
        """
        if not self.api_key_configured:
            return [{
                "page_number": 1,
                "raw_text": "",
                "error": "Google API key not configured. Please set GOOGLE_API_KEY environment variable."
            }]
        
        try:
            # Open PDF directly from bytes using PyMuPDF
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(pdf_document)
            
            print(f"Processing PDF with {total_pages} pages...")
            
            # First pass: Try to extract text directly from each page
            pages_needing_ocr = []
            results = []
            
            for page_num in range(total_pages):
                try:
                    page = pdf_document[page_num]
                    # Try to extract text directly
                    text = page.get_text().strip()
                    
                    if text and len(text) > 10:  # Page has extractable text
                        print(f"Page {page_num + 1}: Extracted {len(text)} characters directly")
                        results.append({
                            "page_number": page_num + 1,
                            "raw_text": text,
                            "columns": [],
                            "data": [],
                            "extraction_method": "direct"
                        })
                    else:
                        # No text or very little - needs OCR
                        print(f"Page {page_num + 1}: No extractable text, will use OCR")
                        pages_needing_ocr.append(page_num)
                        results.append(None)  # Placeholder
                except Exception as e:
                    print(f"Error extracting text from page {page_num + 1}: {e}")
                    pages_needing_ocr.append(page_num)
                    results.append(None)
            
            # If no pages need OCR, return early
            if not pages_needing_ocr:
                pdf_document.close()
                print(f"Successfully extracted text from all {total_pages} pages directly")
                return results
            
            # Second pass: Use OCR for pages without extractable text
            print(f"Using OCR for {len(pages_needing_ocr)} pages...")
            total_pages = len(pdf_document)
            
            # Extract images only for pages needing OCR
            page_images = []
            for page_num in pages_needing_ocr:
                try:
                    page = pdf_document[page_num]
                    # Convert page to image (PNG format)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                    img_data = pix.tobytes("png")
                    page_images.append((img_data, page_num))
                except Exception as e:
                    # Store error for this page
                    page_images.append((None, page_num))
                    print(f"Warning: Failed to extract image from page {page_num + 1}: {str(e)}")
            
            # Close the PDF document early (no longer needed)
            pdf_document.close()
            
            print(f"Processing {len(pages_needing_ocr)} pages with OCR...")
            
            # Process pages in parallel (limit to 5 concurrent requests to avoid rate limits)
            max_workers = min(5, len(pages_needing_ocr))
            
            if len(pages_needing_ocr) > 0:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all page processing tasks
                    future_to_page = {}
                    for img_data, page_num in page_images:
                        if img_data is not None:
                            future = executor.submit(self._process_single_page_image, img_data, page_num)
                            future_to_page[future] = page_num
                        else:
                            # Handle pages that failed image extraction
                            results[page_num] = {
                                "page_number": page_num + 1,
                                "error": f"Failed to extract image from page {page_num + 1}",
                                "raw_text": "",
                                "columns": [],
                                "data": [],
                                "extraction_method": "ocr_failed"
                            }
                    
                    # Collect results as they complete
                    completed = 0
                    for future in as_completed(future_to_page):
                        page_num = future_to_page[future]
                        try:
                            page_result = future.result()
                            page_result["extraction_method"] = "ocr"
                            results[page_num] = page_result
                            completed += 1
                            if completed % 5 == 0 or completed == len(pages_needing_ocr):
                                print(f"OCR processed {completed}/{len(pages_needing_ocr)} pages...")
                        except Exception as e:
                            # Fallback error handling
                            results[page_num] = {
                                "page_number": page_num + 1,
                                "error": f"Failed to process page {page_num + 1}: {str(e)}",
                                "raw_text": "",
                                "columns": [],
                                "data": [],
                                "extraction_method": "ocr_failed"
                            }
            
            # Remove None placeholders and return results in order
            final_results = [r for r in results if r is not None]
            print(f"Completed processing PDF: {len(final_results)} pages total ({len(pages_needing_ocr)} via OCR, {total_pages - len(pages_needing_ocr)} direct extraction)")
            return final_results
        
        except Exception as e:
            raise Exception(f"Error processing PDF bytes: {str(e)}")

    def _parse_gemini_response(self, response, source_info: str) -> Dict[str, Any]:
        """
        Parse Gemini AI response and extract JSON data.
        
        Args:
            response: Gemini AI response object
            source_info (str): Information about the source for error reporting
        
        Returns:
            dict: Parsed structured data with raw_text field
        """
        try:
            # Get the raw response text
            response_text = response.text.strip()
            
            # Remove any markdown code block markers if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            # Try to parse as JSON
            try:
                extracted_data = json.loads(response_text.strip())
                
                # Ensure the response has the expected structure
                if not isinstance(extracted_data, dict):
                    raise ValueError("Response is not a dictionary")
                
                # Ensure raw_text exists - if not, use the response text or try to extract it
                if "raw_text" not in extracted_data or not extracted_data.get("raw_text"):
                    # If no raw_text in JSON, use the full response text
                    extracted_data["raw_text"] = response_text.strip()
                
                # Legacy fields for backward compatibility
                if "columns" not in extracted_data:
                    extracted_data["columns"] = []
                
                if "data" not in extracted_data:
                    extracted_data["data"] = []
                
                return extracted_data
                
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw text
                return {
                    "columns": [],
                    "data": [],
                    "raw_text": response_text.strip(),
                    "error": f"Could not parse JSON response for {source_info}"
                }
            
        except Exception as e:
            # If response.text fails, try to get any available text
            try:
                fallback_text = response.text if hasattr(response, 'text') else str(response)
            except:
                fallback_text = ""
            
            return {
                "columns": [],
                "data": [],
                "raw_text": fallback_text,
                "error": f"Error parsing response for {source_info}: {str(e)}"
            }


# Create a singleton instance
ocr_service = OCRService()
