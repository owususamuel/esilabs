"""
PDF Parser Tool - Extracts text and structure from research papers.
Uses PyMuPDF for efficient PDF processing.
"""

import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PDFContent:
    
    full_text: str
    pages_count: int
    title: Optional[str]
    abstract: Optional[str]
    sections: Dict[str, str]
    metadata: Dict[str, str]


class PDFParser:
    # Common section headers in research papers
    SECTION_PATTERNS = {
        'abstract': r'^(abstract|summary)',
        'introduction': r'^(introduction|overview)',
        'methodology': r'^(methodology|method|approach|proposed|model)',
        'experiments': r'^(experiments|experimental setup|evaluation)',
        'results': r'^(results|findings|performance)',
        'related': r'^(related work|background)',
        'conclusion': r'^(conclusion|conclusions|future work)',
        'references': r'^(references|bibliography)',
    }
    
    def __init__(self):
        """Initialize the PDF parser."""
        self.logger = logger
    
    def parse_pdf(self, pdf_path: str) -> PDFContent:
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            document = fitz.open(pdf_path)
        except Exception as e:
            raise ValueError(f"Failed to open PDF: {e}")
        
        # Extract text from all pages
        full_text = ""
        pages_count = len(document)
        for page_num in range(pages_count):
            page = document[page_num]
            full_text += page.get_text() + "\n"
        
        # Extract metadata
        metadata = document.metadata or {}
        
        # Extract title (usually from first page or metadata)
        title = self._extract_title(full_text, metadata)
        
        # Extract abstract
        abstract = self._extract_section(full_text, 'abstract')
        
        # Extract sections
        sections = self._extract_sections(full_text)
        
        document.close()
        
        self.logger.info(f"Successfully parsed PDF: {pdf_path}")
        
        return PDFContent(
            full_text=full_text,
            pages_count=pages_count,
            title=title,
            abstract=abstract,
            sections=sections,
            metadata=metadata
        )
    
    def _extract_title(self, text: str, metadata: Dict) -> Optional[str]:
        
        # Try metadata first
        if isinstance(metadata, dict) and metadata.get('title'):
            return metadata['title']
        
        # Try to find title in first page (before abstract)
        lines = text.split('\n')[:20]  # Check first 20 lines
        
        for line in lines:
            if len(line.strip()) > 20 and len(line.strip()) < 300:
                # Title is usually 20-300 characters
                if not line.startswith(('http', 'arXiv', 'doi', 'ISBN')):
                    return line.strip()
        
        return None
    
    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        
        if section_name not in self.SECTION_PATTERNS:
            return None
        
        pattern = self.SECTION_PATTERNS[section_name]
        
        # Find section start
        lines = text.split('\n')
        section_start = None
        section_end = None
        
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                section_start = i
                break
        
        if section_start is None:
            return None
        
        # Find next section (section end)
        for i in range(section_start + 1, len(lines)):
            line = lines[i]
            # Check if this is a new section header
            for other_pattern in self.SECTION_PATTERNS.values():
                if other_pattern != pattern and re.search(other_pattern, line, re.IGNORECASE):
                    section_end = i
                    break
            if section_end is not None:
                break
        
        if section_end is None:
            section_end = len(lines)
        
        # Extract section content
        section_lines = lines[section_start:section_end]
        section_text = '\n'.join(section_lines)
        
        return section_text.strip()
    
    def _extract_sections(self, text: str) -> Dict[str, str]:

        sections = {}
        
        for section_name in self.SECTION_PATTERNS.keys():
            section_content = self._extract_section(text, section_name)
            if section_content:
                sections[section_name] = section_content
        
        return sections
    
    def extract_hyperparameters(self, text: str) -> Dict[str, str]:
        
        hyperparams = {}
        
        # Common patterns for hyperparameters
        patterns = {
            'learning_rate': r'(?:learning rate|lr|α|alpha)[:\s]+([0-9.e\-]+)',
            'batch_size': r'(?:batch size|batch)[:\s]+(\d+)',
            'epochs': r'(?:epoch|epochs)[:\s]+(\d+)',
            'dropout': r'(?:dropout)[:\s]+([0-9.]+)',
            'optimizer': r'(?:optimizer)[:\s]+(\w+)',
            'activation': r'(?:activation)[:\s]+(\w+)',
        }
        
        for param_name, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                hyperparams[param_name] = match.group(1)
        
        return hyperparams
    
    def extract_github_urls(self, text: str) -> List[str]:
        
        # Normalize text to counter PDF quirks (soft hyphens, zero-width spaces, split domains)
        normalized = (
            text
            .replace('\u00ad', '')  # soft hyphen
            .replace('\u200b', '')  # zero-width space
            .replace('\u200c', '')
            .replace('\u200d', '')
            .replace('\u2060', '')
        )
        # Join common split domain cases like "github.\ncom" or "github . com"
        normalized = re.sub(r'github\s*[\.\u00b7]?\s*[\r\n ]*\s*com', 'github.com', normalized, flags=re.IGNORECASE)
        # Join breaks immediately after domain slash: "github.com/\nusername"
        normalized = re.sub(r'(?i)(github\.com/)\s+', r'\1', normalized)
        # Join breaks when scheme is separated: "https://github.com \n /user"
        normalized = re.sub(r'(?i)(https?://\s*(?:www\.)?\s*github\.com)\s*/\s*', r'\1/', normalized)
        # Join breaks between path segments: "/user \n /repo" or "/user/\nrepo"
        normalized = re.sub(r'(?i)(github\.com/[^\s/]+)\s*/\s*', r'\1/', normalized)
        
        # Match full GitHub URLs, with or without scheme
        github_pattern = r'(?i)\b(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+(?:/[^\s)\]]+)?'
        raw_urls = re.findall(github_pattern, normalized)
        
        cleaned_urls: List[str] = []
        for url in raw_urls:
            # Ensure scheme
            if not url.lower().startswith(('http://', 'https://')):
                url = f'https://{url}'
            # Strip trailing punctuation that often follows in prose
            url = url.rstrip('.,;)]}')
            cleaned_urls.append(url)
        
        urls = list(dict.fromkeys(cleaned_urls))  # preserve order, remove duplicates
        
        return urls


def extract_paper_metadata(pdf_path: str) -> Dict[str, any]:
    parser = PDFParser()
    content = parser.parse_pdf(pdf_path)
    
    return {
        'title': content.title,
        'abstract': content.abstract,
        'pages': content.pages_count,
        'github_urls': parser.extract_github_urls(content.full_text),
        'hyperparameters': parser.extract_hyperparameters(content.full_text),
        'sections': list(content.sections.keys()),
    }
