"""
PDF Parser Tool - Extracts text and structure from research papers.
Uses PyMuPDF for efficient PDF processing.
"""

import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import io
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Figure:
    """Represents a figure/image extracted from the paper."""
    image_data: bytes
    page_number: int
    caption: Optional[str] = None
    figure_number: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)


@dataclass
class Table:
    """Represents a table extracted from the paper."""
    content: str  # Table as text
    page_number: int
    caption: Optional[str] = None
    table_number: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    metrics: Optional[Dict[str, float]] = None  # Extracted numerical metrics


@dataclass
class ExperimentalResults:
    """Container for all experimental results extracted from the paper."""
    results_text: str  # Text from results/experiments section
    figures: List[Figure] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    metrics_mentioned: List[str] = field(default_factory=list)


@dataclass
class PDFContent:
    
    full_text: str
    pages_count: int
    title: Optional[str]
    abstract: Optional[str]
    sections: Dict[str, str]
    metadata: Dict[str, str]
    experimental_results: Optional[ExperimentalResults] = None


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
    
    def extract_figures(self, pdf_path: str) -> List[Figure]:
        """Extract all figures/images from the PDF with their captions.
        
        Uses two methods:
        1. Extract embedded images directly (for raster graphics)
        2. Render page and crop figures (for vector graphics)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        figures = []
        
        try:
            document = fitz.open(pdf_path)
            
            for page_num in range(len(document)):
                page = document[page_num]
                page_text = page.get_text()
                
                # Extract images from page
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = document.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Filter out small images (likely icons/logos)
                        # Check image dimensions
                        try:
                            pil_img = Image.open(io.BytesIO(image_bytes))
                            width, height = pil_img.size
                            # Skip small images (likely not figures)
                            if width < 100 or height < 100:
                                continue
                            
                            # Check if image is mostly black/empty (vector graphics extraction failure)
                            if self._is_mostly_black(pil_img):
                                self.logger.warning(f"Figure {img_index} on page {page_num + 1} appears black - may be vector graphic")
                                # Try rendering the page instead
                                rendered_image = self._render_figure_from_page(page, img, xref)
                                if rendered_image:
                                    image_bytes = rendered_image
                                    self.logger.info(f"Successfully re-rendered figure {img_index} from page {page_num + 1}")
                                    
                        except Exception as e:
                            self.logger.debug(f"Error checking image quality: {e}")
                            continue
                        
                        # Try to find caption near this image
                        caption = self._find_figure_caption(page_text, img_index)
                        
                        # Extract figure number if present
                        figure_number = None
                        if caption:
                            fig_match = re.search(r'(?i)(?:figure|fig\.?)\s*(\d+)', caption)
                            if fig_match:
                                figure_number = fig_match.group(1)
                        
                        figures.append(Figure(
                            image_data=image_bytes,
                            page_number=page_num + 1,
                            caption=caption,
                            figure_number=figure_number,
                            bbox=None  # Could extract from page.get_image_bbox if needed
                        ))
                    except Exception as e:
                        self.logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            document.close()
            self.logger.info(f"Extracted {len(figures)} figures from PDF")
            
        except Exception as e:
            self.logger.error(f"Error extracting figures: {e}")
        
        return figures
    
    def _is_mostly_black(self, img: Image.Image, threshold: float = 0.95) -> bool:
        """Check if an image is mostly black (indicates failed vector graphic extraction)."""
        try:
            # Convert to grayscale
            img_gray = img.convert('L')
            # Get pixel data
            pixels = list(img_gray.getdata())
            total_pixels = len(pixels)
            
            # Count pixels that are very dark (< 20 out of 255)
            dark_pixels = sum(1 for p in pixels if p < 20)
            
            # If more than threshold% of pixels are black, it's likely a failed extraction
            return (dark_pixels / total_pixels) > threshold
            
        except Exception as e:
            self.logger.debug(f"Error checking if image is black: {e}")
            return False
    
    def _render_figure_from_page(self, page, img, xref) -> Optional[bytes]:
        """Render a figure by rendering the page as image and cropping figure region.
        
        This is a fallback for vector graphics that don't extract well directly.
        """
        try:
            # Get the bounding box of the image on the page
            img_rect = None
            for item in page.get_images(full=True):
                if item[0] == xref:
                    # Get image rectangle from page
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        img_rect = img_rects[0]  # Use first occurrence
                        break
            
            if not img_rect:
                return None
            
            # Render the page at higher resolution
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat, clip=img_rect)
            
            # Convert to PNG bytes
            img_bytes = pix.tobytes("png")
            
            # Verify the rendered image is not also black
            pil_img = Image.open(io.BytesIO(img_bytes))
            if not self._is_mostly_black(pil_img):
                return img_bytes
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error rendering figure from page: {e}")
            return None
    
    def _find_figure_caption(self, page_text: str, img_index: int) -> Optional[str]:
        """Try to find the caption for a figure on the page."""
        # Look for common caption patterns
        caption_patterns = [
            r'(?i)(figure\s+\d+[:\.]?\s*[^\n]+(?:\n[^\n]+){0,3})',
            r'(?i)(fig\.\s+\d+[:\.]?\s*[^\n]+(?:\n[^\n]+){0,3})',
        ]
        
        for pattern in caption_patterns:
            matches = re.findall(pattern, page_text)
            if matches and img_index < len(matches):
                return matches[img_index].strip()
        
        return None
    
    def extract_tables(self, pdf_path: str) -> List[Table]:
        """Extract tables from the PDF with their captions."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        tables = []
        
        try:
            document = fitz.open(pdf_path)
            
            for page_num in range(len(document)):
                page = document[page_num]
                page_text = page.get_text()
                
                # Find table captions and extract surrounding content
                table_pattern = r'(?i)table\s+(\d+)[:\.]?\s*([^\n]+(?:\n(?!(?:figure|table|section|\d+\.))[^\n]+){0,20})'
                table_matches = re.finditer(table_pattern, page_text)
                
                for match in table_matches:
                    table_number = match.group(1)
                    caption = match.group(2).strip()
                    
                    # Extract table content (text after caption until next section/table)
                    start_pos = match.end()
                    # Look for next major delimiter (but not decimal numbers like 0.4288)
                    # Match section numbers (e.g., "1. Introduction") or keywords (Figure, Table, Section)
                    # Require space after period to avoid matching decimals
                    next_section = re.search(r'(?i)\n(?:figure|table|section|\d+\.\s+[A-Z])', page_text[start_pos:])
                    end_pos = start_pos + (next_section.start() if next_section else min(1000, len(page_text) - start_pos))
                    
                    table_content = page_text[start_pos:end_pos].strip()
                    
                    if table_content:
                        # Try to extract structured metrics from table
                        table_metrics = self._extract_table_metrics(table_content)
                        
                        tables.append(Table(
                            content=table_content,
                            page_number=page_num + 1,
                            caption=caption,
                            table_number=table_number,
                            bbox=None,
                            metrics=table_metrics
                        ))
            
            document.close()
            self.logger.info(f"Extracted {len(tables)} tables from PDF")
            
        except Exception as e:
            self.logger.error(f"Error extracting tables: {e}")
        
        return tables
    
    def _extract_table_metrics(self, table_text: str) -> Dict[str, float]:
        """Extract numerical metrics from table text."""
        import re
        
        metrics = {}
        lines = table_text.split('\n')
        
        for line in lines:
            # Look for patterns like "Method: 0.95" or "Accuracy 89.5%"
            # Also look for lines with metric names followed by numbers (even without colons)
            pattern = r'([A-Za-z][\w\s@-]+?)\s*[:=\s]+\s*([0-9]+\.?[0-9]*%?)'
            matches = re.finditer(pattern, line)
            
            for match in matches:
                metric_name = match.group(1).strip()
                value_str = match.group(2).replace('%', '')
                
                try:
                    value = float(value_str)
                    # Convert percentages to decimals if > 1
                    if '%' in match.group(2) and value > 1:
                        value = value / 100
                    metrics[metric_name] = value
                except ValueError:
                    continue
        
        return metrics
    
    def extract_evaluation_metrics(self, text: str) -> List[str]:
        """Extract common evaluation metrics mentioned in the paper."""
        metrics = []
        
        # Common ML/AI evaluation metrics
        metric_patterns = [
            r'\b(accuracy|acc)\b',
            r'\b(precision)\b',
            r'\b(recall)\b',
            r'\b(f1[- ]score|f1)\b',
            r'\b(auc|auroc|roc)\b',
            r'\b(mean average precision|map|mAP)\b',
            r'\b(bleu|rouge|meteor)\b',
            r'\b(perplexity|ppl)\b',
            r'\b(mean squared error|mse|rmse)\b',
            r'\b(mean absolute error|mae)\b',
            r'\b(r2[- ]score|r²|r-squared)\b',
            r'\b(iou|intersection over union)\b',
            r'\b(dice coefficient|dice score)\b',
        ]
        
        for pattern in metric_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Extract the metric name
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    metric = match.group(1)
                    if metric.lower() not in [m.lower() for m in metrics]:
                        metrics.append(metric)
        
        return metrics
    
    def extract_experimental_results(self, pdf_path: str) -> ExperimentalResults:
        """
        Extract all experimental results from the paper:
        - Results section text
        - Figures and plots
        - Tables
        - Mentioned evaluation metrics
        """
        # Parse the PDF
        content = self.parse_pdf(pdf_path)
        
        # Extract results section text
        results_text = ""
        for section_name in ['results', 'experiments', 'experimental_results']:
            if section_name in content.sections:
                results_text += content.sections[section_name] + "\n\n"
        
        if not results_text:
            results_text = content.sections.get('evaluation', '')
        
        # Extract figures (plots, charts, visualizations)
        figures = self.extract_figures(pdf_path)
        
        # Extract tables
        tables = self.extract_tables(pdf_path)
        
        # Extract mentioned metrics
        metrics = self.extract_evaluation_metrics(content.full_text)
        
        self.logger.info(
            f"Extracted experimental results: {len(figures)} figures, "
            f"{len(tables)} tables, {len(metrics)} metrics"
        )
        
        return ExperimentalResults(
            results_text=results_text,
            figures=figures,
            tables=tables,
            metrics_mentioned=metrics
        )


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
