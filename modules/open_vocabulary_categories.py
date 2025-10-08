#!/usr/bin/env python3

"""
Open Vocabulary Category Manager for Voxeland

This module provides dynamic category management for open vocabulary semantic mapping,
replacing the hardcoded 80 COCO categories with a flexible system that can handle
unlimited categories from any detection system.
"""

import threading
from typing import List, Set, Dict, Optional
import json
import os

class OpenVocabularyCategoryManager:
    """
    Manages categories dynamically for open vocabulary semantic mapping.
    
    This class replaces the hardcoded COCO categories and allows TALOS or any
    other open vocabulary detector to add new categories dynamically.
    """
    
    def __init__(self):
        self._categories: List[str] = []
        self._category_to_index: Dict[str, int] = {}
        self._new_categories: Set[str] = set()
        self._lock = threading.Lock()
        self._initialized = False
        
        # Always start with unknown and background categories
        self._add_category_internal("unknown")
        self._add_category_internal("background")
    
    def add_category(self, category_name: str) -> int:
        """
        Add a new category dynamically.
        
        Args:
            category_name: Name of the category to add
            
        Returns:
            Index of the category (existing or newly created)
        """
        with self._lock:
            if category_name in self._category_to_index:
                return self._category_to_index[category_name]
            
            index = self._add_category_internal(category_name)
            self._new_categories.add(category_name)
            return index
    
    def _add_category_internal(self, category_name: str) -> int:
        """Internal method to add category without locking."""
        index = len(self._categories)
        self._categories.append(category_name)
        self._category_to_index[category_name] = index
        return index
    
    def get_category_index(self, category_name: str) -> Optional[int]:
        """Get the index of a category."""
        with self._lock:
            return self._category_to_index.get(category_name)
    
    def get_category_name(self, index: int) -> Optional[str]:
        """Get the name of a category by index."""
        with self._lock:
            if 0 <= index < len(self._categories):
                return self._categories[index]
            return None
    
    def get_all_categories(self) -> List[str]:
        """Get all categories."""
        with self._lock:
            return self._categories.copy()
    
    def get_num_categories(self) -> int:
        """Get the total number of categories."""
        with self._lock:
            return len(self._categories)
    
    def has_category(self, category_name: str) -> bool:
        """Check if a category exists."""
        with self._lock:
            return category_name in self._category_to_index
    
    def initialize_with_categories(self, default_categories: List[str]):
        """
        Initialize with a set of default categories.
        
        Args:
            default_categories: List of default category names
        """
        with self._lock:
            if self._initialized:
                raise RuntimeError("CategoryManager already initialized. Use reset() first if needed.")
            
            for category in default_categories:
                if category not in self._category_to_index:
                    self._add_category_internal(category)
            
            self._initialized = True
    
    def reset(self):
        """Reset all categories."""
        with self._lock:
            self._categories.clear()
            self._category_to_index.clear()
            self._new_categories.clear()
            self._initialized = False
            
            # Re-add default categories
            self._add_category_internal("unknown")
            self._add_category_internal("background")
    
    def get_new_categories(self) -> List[str]:
        """Get categories added since last check."""
        with self._lock:
            new_cats = list(self._new_categories)
            self._new_categories.clear()
            return new_cats
    
    def has_new_categories(self) -> bool:
        """Check if new categories have been added since last check."""
        with self._lock:
            return len(self._new_categories) > 0
    
    def save_to_file(self, filepath: str):
        """Save categories to a JSON file."""
        with self._lock:
            data = {
                'categories': self._categories,
                'category_to_index': self._category_to_index
            }
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load categories from a JSON file."""
        with self._lock:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                self._categories = data.get('categories', [])
                self._category_to_index = data.get('category_to_index', {})
                
                # Ensure unknown and background are present
                if "unknown" not in self._category_to_index:
                    self._add_category_internal("unknown")
                if "background" not in self._category_to_index:
                    self._add_category_internal("background")
    
    def save_categories_to_file(self, filepath: str) -> None:
        """Save categories to a JSON file."""
        with self._lock:
            data = {
                'categories': self._categories.copy(),
                'category_to_index': self._category_to_index.copy(),
                'num_categories': len(self._categories),
                'timestamp': self._get_timestamp()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        import datetime
        return datetime.datetime.now().isoformat()


# Global instance for easy access
_global_category_manager = OpenVocabularyCategoryManager()

def get_category_manager() -> OpenVocabularyCategoryManager:
    """Get the global category manager instance."""
    return _global_category_manager