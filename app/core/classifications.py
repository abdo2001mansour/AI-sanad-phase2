"""
Classification management for Saudi legal documents
"""
import json
from pathlib import Path
from typing import List, Dict, Optional

# Load classifications configuration
CLASSIFICATIONS_CONFIG_PATH = Path(__file__).parent.parent / "config" / "classifications_config.json"


def load_classifications() -> List[Dict]:
    """Load classifications from JSON configuration file"""
    try:
        with open(CLASSIFICATIONS_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get("classifications", [])
    except Exception as e:
        print(f"Warning: Could not load classifications config: {e}")
        return []


class ClassificationManager:
    """Manager for legal document classifications"""
    
    def __init__(self):
        """Initialize the classification manager"""
        self.classifications = load_classifications()
        
        # Build lookup dictionaries for fast access
        self.by_id = {c["id"]: c for c in self.classifications}
        self.by_arabic = {c["name_ar"]: c for c in self.classifications}
        self.by_english = {c["name_en"]: c for c in self.classifications}
    
    def get_by_id(self, classification_id: str) -> Optional[Dict]:
        """Get classification by ID"""
        return self.by_id.get(classification_id)
    
    def get_by_arabic(self, arabic_name: str) -> Optional[Dict]:
        """Get classification by Arabic name"""
        return self.by_arabic.get(arabic_name)
    
    def get_by_english(self, english_name: str) -> Optional[Dict]:
        """Get classification by English name"""
        return self.by_english.get(english_name)
    
    def get_arabic_name(self, classification_id: str) -> str:
        """Get Arabic name from ID"""
        classification = self.by_id.get(classification_id)
        return classification["name_ar"] if classification else classification_id
    
    def get_english_name(self, classification_id: str) -> str:
        """Get English name from ID"""
        classification = self.by_id.get(classification_id)
        return classification["name_en"] if classification else classification_id
    
    def convert_ids_to_arabic(self, classification_ids: List[str]) -> List[str]:
        """Convert list of IDs to Arabic names"""
        return [self.get_arabic_name(id) for id in classification_ids]
    
    def convert_ids_to_english(self, classification_ids: List[str]) -> List[str]:
        """Convert list of IDs to English names"""
        return [self.get_english_name(id) for id in classification_ids]
    
    def convert_arabic_to_ids(self, arabic_names: List[str]) -> List[str]:
        """Convert list of Arabic names to IDs"""
        ids = []
        for name in arabic_names:
            classification = self.get_by_arabic(name)
            if classification:
                ids.append(classification["id"])
            else:
                # If not found, keep original (might be legacy data)
                ids.append(name)
        return ids
    
    def convert_english_to_ids(self, english_names: List[str]) -> List[str]:
        """Convert list of English names to IDs"""
        ids = []
        for name in english_names:
            classification = self.get_by_english(name)
            if classification:
                ids.append(classification["id"])
            else:
                # If not found, keep original (might be legacy data)
                ids.append(name)
        return ids
    
    def get_all_with_translations(self) -> List[Dict]:
        """Get all classifications with both Arabic and English names"""
        return [
            {
                "id": c["id"],
                "name_ar": c["name_ar"],
                "name_en": c["name_en"],
                "description_ar": c.get("description_ar", ""),
                "description_en": c.get("description_en", "")
            }
            for c in self.classifications
        ]
    
    def normalize_classification_value(self, value: str) -> str:
        """
        Normalize a classification value (could be ID, Arabic, or English) to Arabic name.
        This ensures compatibility with existing Pinecone data stored in Arabic.
        
        Args:
            value: Classification value (ID, Arabic name, or English name)
        
        Returns:
            Arabic name for Pinecone query
        """
        # Try ID first
        if value in self.by_id:
            return self.by_id[value]["name_ar"]
        
        # Try English
        if value in self.by_english:
            return self.by_english[value]["name_ar"]
        
        # Try Arabic (already correct)
        if value in self.by_arabic:
            return value
        
        # Not found - return as-is (might be legacy data)
        return value
    
    def normalize_classifications_list(self, values: List[str]) -> List[str]:
        """
        Normalize a list of classification values to Arabic names for Pinecone.
        
        Args:
            values: List of classification values (IDs, Arabic, or English names)
        
        Returns:
            List of Arabic names for Pinecone query
        """
        return [self.normalize_classification_value(v) for v in values]


# Create singleton instance
classification_manager = ClassificationManager()

