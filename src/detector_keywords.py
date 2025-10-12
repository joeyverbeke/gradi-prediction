import ahocorasick
from typing import List


class KeywordDetector:
    """Keyword detector using Aho-Corasick automaton"""
    
    def __init__(self, stems: List[str]):
        """
        Initialize keyword detector
        
        Args:
            stems: List of keyword stems to detect
        """
        self.stems = [stem.lower() for stem in stems]
        
        # Build Aho-Corasick automaton
        self.automaton = ahocorasick.Automaton()
        
        for stem in self.stems:
            self.automaton.add_word(stem, stem)
        
        self.automaton.make_automaton()
        print(f"Built keyword detector with {len(stems)} stems")
    
    def scan(self, text: str) -> bool:
        """
        Scan text for keyword matches
        
        Args:
            text: Text to scan
            
        Returns:
            True if any keyword stem is found, False otherwise
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Search for matches
        for end_index, stem in self.automaton.iter(text_lower):
            return True  # Found at least one match
        
        return False
    
    def scan_with_details(self, text: str) -> List[str]:
        """
        Scan text and return matching stems
        
        Args:
            text: Text to scan
            
        Returns:
            List of matching stems
        """
        if not text:
            return []
        
        text_lower = text.lower()
        matches = []
        
        # Search for matches
        for end_index, stem in self.automaton.iter(text_lower):
            matches.append(stem)
        
        return matches
    
    def get_stems(self) -> List[str]:
        """Get list of stems"""
        return self.stems.copy()
    
    @classmethod
    def from_config(cls, config_stems: List[str]):
        """
        Create detector from configuration stems
        
        Args:
            config_stems: Stems from config file
            
        Returns:
            KeywordDetector instance
        """
        return cls(config_stems)
