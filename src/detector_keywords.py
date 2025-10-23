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
    
    def find_matches(self, text: str) -> List[tuple]:
        """Return list of (stem, end_index) matches found in text."""
        matches = []
        if not text:
            return matches

        text_lower = text.lower()
        for end_index, stem in self.automaton.iter(text_lower):
            matches.append((stem, end_index))

        return matches

    def scan(self, text: str) -> bool:
        """
        Scan text for keyword matches

        Args:
            text: Text to scan
            
        Returns:
            True if any keyword stem is found, False otherwise
        """
        return bool(self.find_matches(text))
    
    def scan_with_details(self, text: str) -> List[str]:
        """
        Scan text and return matching stems
        
        Args:
            text: Text to scan
            
        Returns:
            List of matching stems
        """
        return [stem for stem, _ in self.find_matches(text)]
    
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
