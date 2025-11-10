"""
Repository Finder Tool - Constructs search queries for finding code repositories related to research papers.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Repository:
    
    name: str
    url: str
    description: Optional[str]
    stars: int
    language: Optional[str]
    last_updated: Optional[str]
    is_official: bool = False
    source: str = "github"


class RepositoryFinder:
    
    def __init__(self):
        """Initialize the repository finder."""
        self.logger = logger
    
    def find_repository(
        self,
        paper_title: str,
        authors: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[Repository]:
        
        # Construct search queries
        search_queries = self._build_search_queries(paper_title, authors)
        
        self.logger.info(f"Generated {len(search_queries)} search queries for paper: {paper_title}")
        self.logger.info(f"Search queries: {search_queries}")
        
        # Return empty list since we're not actually searching anymore
        # The agent will use these queries to search
        return []
    
    def _build_search_queries(
        self,
        paper_title: str,
        authors: Optional[List[str]]
    ) -> List[str]:
        
        search_queries = []
        
        # Extract key terms from title (remove common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'is', 'for', 'on', 'with'}
        title_words = [w for w in paper_title.lower().split() if w not in common_words]
        
        # Build search queries
        if len(title_words) >= 2:
            # Try combining first few title words
            search_queries.append(" ".join(title_words[:3]))
        
        # Include author names in search
        if authors and len(authors) > 0:
            first_author = authors[0].split()[-1]  # Last name
            if title_words:
                search_queries.append(f"{first_author} {title_words[0]}")
        
        return search_queries
    
    def get_search_url(self, paper_title: str, authors: Optional[List[str]] = None) -> str:
        queries = self._build_search_queries(paper_title, authors)
        if queries:
            primary_query = queries[0]
            # URL encode the query
            import urllib.parse
            encoded_query = urllib.parse.quote(primary_query)
            return f"https://github.com/search?q={encoded_query}&type=repositories&s=stars&o=desc"
        return "https://github.com/search?type=repositories"


def find_paper_repository(
    paper_title: str,
    authors: Optional[List[str]] = None
) -> Dict[str, any]:
    
    finder = RepositoryFinder()
    queries = finder._build_search_queries(paper_title, authors)
    search_url = finder.get_search_url(paper_title, authors)
    
    return {
        "search_queries": queries,
        "search_url": search_url,
        "paper_title": paper_title,
        "authors": authors or []
    }
