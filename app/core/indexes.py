"""
Index management for Pinecone vector databases
"""
import json
from pathlib import Path
from typing import List, Dict, Optional

# Load indexes configuration
INDEXES_CONFIG_PATH = Path(__file__).parent.parent / "config" / "indexes_config.json"


def load_indexes() -> List[Dict]:
    """Load indexes from JSON configuration file"""
    try:
        with open(INDEXES_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get("indexes", [])
    except Exception as e:
        print(f"Warning: Could not load indexes config: {e}")
        return []


class IndexManager:
    """Manager for Pinecone index names and translations"""
    
    def __init__(self):
        """Initialize the index manager"""
        self.indexes = load_indexes()
        
        # Build lookup dictionaries for fast access
        self.by_id = {idx["id"]: idx for idx in self.indexes}
        self.by_technical_name = {idx["technical_name"]: idx for idx in self.indexes}
        self.by_arabic = {idx["name_ar"]: idx for idx in self.indexes}
        self.by_english = {idx["name_en"]: idx for idx in self.indexes}
    
    def get_by_id(self, index_id: str) -> Optional[Dict]:
        """Get index by ID"""
        return self.by_id.get(index_id)
    
    def get_by_technical_name(self, technical_name: str) -> Optional[Dict]:
        """Get index by technical name (actual Pinecone index name)"""
        return self.by_technical_name.get(technical_name)
    
    def get_by_arabic(self, arabic_name: str) -> Optional[Dict]:
        """Get index by Arabic name"""
        return self.by_arabic.get(arabic_name)
    
    def get_by_english(self, english_name: str) -> Optional[Dict]:
        """Get index by English name"""
        return self.by_english.get(english_name)
    
    def get_technical_name(self, index_identifier: str) -> str:
        """
        Get technical Pinecone index name from any identifier (ID, Arabic, or English).
        
        Args:
            index_identifier: ID, Arabic name, or English name
        
        Returns:
            Technical Pinecone index name (e.g., "qadha", "contracts")
        """
        # Try ID first
        if index_identifier in self.by_id:
            return self.by_id[index_identifier]["technical_name"]
        
        # Try technical name (already correct)
        if index_identifier in self.by_technical_name:
            return index_identifier
        
        # Try Arabic
        if index_identifier in self.by_arabic:
            return self.by_arabic[index_identifier]["technical_name"]
        
        # Try English
        if index_identifier in self.by_english:
            return self.by_english[index_identifier]["technical_name"]
        
        # Not found - return as-is (might be a custom index)
        return index_identifier
    
    def get_display_names(self, technical_name: str) -> Dict[str, str]:
        """
        Get display names (Arabic and English) for a technical index name.
        
        Args:
            technical_name: Technical Pinecone index name
        
        Returns:
            Dictionary with name_ar and name_en
        """
        index = self.get_by_technical_name(technical_name)
        if index:
            return {
                "name_ar": index["name_ar"],
                "name_en": index["name_en"]
            }
        else:
            return {
                "name_ar": technical_name,
                "name_en": technical_name
            }
    
    def convert_to_technical_names(self, identifiers: List[str]) -> List[str]:
        """
        Convert list of identifiers (IDs, Arabic, or English) to technical Pinecone names.
        
        Args:
            identifiers: List of index identifiers
        
        Returns:
            List of technical Pinecone index names
        """
        return [self.get_technical_name(identifier) for identifier in identifiers]
    
    def get_all_with_translations(self) -> List[Dict]:
        """Get all indexes with both Arabic and English names"""
        return [
            {
                "id": idx["id"],
                "technical_name": idx["technical_name"],
                "name_ar": idx["name_ar"],
                "name_en": idx["name_en"],
                "description_ar": idx.get("description_ar", ""),
                "description_en": idx.get("description_en", "")
            }
            for idx in self.indexes
        ]
    
    def get_all_technical_names(self) -> List[str]:
        """Get list of all technical Pinecone index names"""
        return [idx["technical_name"] for idx in self.indexes]


# Create singleton instance
index_manager = IndexManager()

