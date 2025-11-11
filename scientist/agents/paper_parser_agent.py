
import ast
import json
import re
import base64
from typing import Dict, Any, Optional, List
from pathlib import Path

from scientist.agents.base_agent import BaseAgent
from scientist.tools.pdf_parser import PDFParser, ExperimentalResults, Figure, Table
from scientist.tools.tool_wrappers import ParsePDF


class PaperParserAgent(BaseAgent):
    """
    Autonomous agent that extracts structured information from research papers.
    
    The agent autonomously:
    1. Reads and parses the PDF
    2. Extracts title, authors, abstract
    3. Identifies methodology and experiments
    4. Extracts hyperparameters, datasets, and experimental results
    5. Finds repository links and code references
    """
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        
        # Load system prompt from YAML configuration
        system_prompt = BaseAgent._load_agent_instructions('paper_parser_agent')
        
        # Fail fast if configuration is missing
        if not system_prompt:
            raise ValueError(
                "Failed to load system prompt for paper_parser_agent from config/agent_instructions.yaml. "
                "Please ensure the configuration file exists and contains the 'paper_parser_agent' section."
            )
        
        super().__init__(
            agent_name="paper_parser",
            system_prompt=system_prompt,
            config_path=config_path
        )
        
        # Initialize backing tools
        self.pdf_parser = PDFParser()
        
        # Register autonomous tools
        self.register_tool("parse_pdf", None, ParsePDF(self.pdf_parser))
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        
        pdf_path = task.get('pdf_path')
        
        if not pdf_path:
            return {
                'success': False,
                'error': 'pdf_path is required'
            }
        
        if not Path(pdf_path).exists():
            return {
                'success': False,
                'error': f'PDF file not found: {pdf_path}'
            }
        
        try:
            self.logger.info(f"🤖 Two-stage analysis starting for: {pdf_path}")
            
            # STAGE 1: PDFParser extracts experimental results (fast, deterministic)
            self.logger.info("📊 Stage 1: Extracting experimental results (figures, tables, metrics)...")
            experimental_results = self.pdf_parser.extract_experimental_results(pdf_path)
            
            self.logger.info(
                f"✅ Extracted: {len(experimental_results.figures)} figures, "
                f"{len(experimental_results.tables)} tables, "
                f"{len(experimental_results.metrics_mentioned)} metrics"
            )
            
            # STAGE 2: LLM Agent analyzes extracted results (semantic understanding)
            self.logger.info("🧠 Stage 2: LLM analyzing experimental results semantically...")
            
            # Load task prompt from YAML
            task_prompt_template = BaseAgent._load_task_prompt('paper_parser_agent')
            if not task_prompt_template:
                raise ValueError(
                    "Failed to load task prompt for paper_parser_agent from config/agent_instructions.yaml. "
                    "Please ensure the configuration file exists and contains the 'task_prompt' field."
                )
            
            # Format experimental results for LLM
            formatted_context = self._format_experimental_results_for_llm(experimental_results)
            
            # Fill in template variables
            agent_task = BaseAgent._render_template(
                task_prompt_template,
                {
                    "pdf_path": str(pdf_path),
                    "experimental_results": formatted_context
                }
            )
            
            # Agent works autonomously on extracted results
            agent_response = self.call_llm(agent_task)
            
            self.logger.info("✅ LLM completed semantic analysis of experimental results")
            
            structured_block = self._extract_structured_block(agent_response)
            extracted_data = None
            
            if structured_block:
                self.logger.debug(f"Structured block detected in agent response ({len(structured_block)} chars)")
                extracted_data = self._parse_structured_dict(structured_block)
                if extracted_data is None:
                    self.logger.warning("Structured block found but parsing failed")
            else:
                self.logger.warning("No structured data block detected in agent response")
            
            if extracted_data:
                # Normalize data types for Pydantic model
                hyperparams = extracted_data.get('hyperparameters', {})
                if not isinstance(hyperparams, dict):
                    hyperparams = {}  # Convert non-dict to empty dict
                
                datasets = extracted_data.get('datasets', [])
                if not isinstance(datasets, list):
                    datasets = []  # Convert non-list to empty list
                
                authors = extracted_data.get('authors', [])
                if not isinstance(authors, list):
                    authors = []  # Convert non-list to empty list
                
                # Helper function to normalize string fields (handle lists, etc.)
                def normalize_string_field(value):
                    if isinstance(value, list):
                        return ' '.join(str(item) for item in value if item)
                    elif not isinstance(value, str):
                        return str(value) if value else ''
                    return value
                
                # Normalize string fields
                title = normalize_string_field(extracted_data.get('title', ''))
                abstract = normalize_string_field(extracted_data.get('abstract', ''))
                methodology = normalize_string_field(extracted_data.get('methodology', '') or extracted_data.get('experimental_setup', ''))
                github_url = normalize_string_field(extracted_data.get('repository_url', '') or extracted_data.get('repository_mentioned', ''))
                
                # Validate that LLM-provided URL is actually GitHub
                if github_url and 'github.com' not in github_url.lower():
                    self.logger.warning(f"LLM provided non-GitHub URL: {github_url}, ignoring")
                    github_url = ''
                
                # Deterministic fallback: directly scan PDF text for GitHub URLs
                try:
                    parsed_pdf = self.pdf_parser.parse_pdf(pdf_path)
                    github_urls_found = self.pdf_parser.extract_github_urls(parsed_pdf.full_text)
                except Exception:
                    github_urls_found = []
                
                # If LLM didn't provide a valid GitHub URL but we detected one, use it
                if not github_url and github_urls_found:
                    github_url = github_urls_found[0]
                
                # Extract experimental results summary from LLM (this is different from the Stage 1 experimental_results object)
                results_summary = normalize_string_field(extracted_data.get('experimental_results', '') or extracted_data.get('results', ''))
                evaluation_metrics = extracted_data.get('evaluation_metrics', [])
                if not isinstance(evaluation_metrics, list):
                    evaluation_metrics = []
                key_findings = normalize_string_field(extracted_data.get('key_findings', ''))
                
                paper_data = {
                    'title': title,
                    'authors': authors,
                    'abstract': abstract,
                    'github_url': github_url,
                    'methodology': methodology,
                    'datasets_used': datasets,
                    'hyperparameters': hyperparams,
                    'experimental_results': results_summary,
                    'evaluation_metrics': evaluation_metrics,
                    'key_findings': key_findings,
                    'pdf_path': pdf_path
                }
                
                result = {
                    'success': True,
                    'pdf_path': pdf_path,
                    'paper_data': paper_data,
                    'github_urls_found': github_urls_found,
                    'sections_found': len([k for k, v in extracted_data.items() if v]),
                    'evaluation_metrics': extracted_data.get('evaluation_metrics', []),
                    'key_findings': extracted_data.get('key_findings', ''),
                    'experimental_results': {
                        'figures_count': len(experimental_results.figures),
                        'tables_count': len(experimental_results.tables),
                        'metrics_found': experimental_results.metrics_mentioned,
                        'results_text_length': len(experimental_results.results_text)
                    },
                    'figures': self._prepare_images_for_llm(experimental_results.figures) if experimental_results.figures else [],
                    'tables': [
                        {
                            'table_number': t.table_number,
                            'caption': t.caption,
                            'content': t.content,
                            'page_number': t.page_number
                        } for t in experimental_results.tables
                    ] if experimental_results.tables else [],
                    'execution_type': 'two_stage',
                    'note': 'Stage 1: PDFParser extracted results; Stage 2: LLM analyzed semantically'
                }
            else:
                result = {
                    'success': False,
                    'error': 'Could not extract structured data from agent response',
                    'pdf_path': pdf_path,
                    'agent_response': agent_response,
                    'structured_block': structured_block,
                    'execution_type': 'autonomous',
                    'note': 'Agent completed but JSON parsing failed, check agent_response'
                }
            
            self.log_execution("autonomous_parse", result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in autonomous execution: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'pdf_path': pdf_path
            }
    
    def _format_experimental_results_for_llm(self, results: ExperimentalResults) -> str:
        """
        Format experimental results (text, tables, figures) for LLM analysis.
        Includes base64-encoded images for multi-modal understanding.
        """
        formatted = []
        
        # Add results text
        if results.results_text:
            formatted.append("=== RESULTS SECTION TEXT ===")
            formatted.append(results.results_text)
            formatted.append("")
        
        # Add metrics
        if results.metrics_mentioned:
            formatted.append("=== EVALUATION METRICS FOUND ===")
            formatted.append(", ".join(results.metrics_mentioned))
            formatted.append("")
        
        # Add tables
        if results.tables:
            formatted.append(f"=== TABLES ({len(results.tables)}) ===")
            for i, table in enumerate(results.tables, 1):
                formatted.append(f"\n--- Table {table.table_number or i} (Page {table.page_number}) ---")
                if table.caption:
                    formatted.append(f"Caption: {table.caption}")
                formatted.append(f"Content:\n{table.content}")
                formatted.append("")
        
        # Add figures with metadata (images will be passed separately if vision model is used)
        if results.figures:
            formatted.append(f"=== FIGURES/PLOTS/CHARTS ({len(results.figures)}) ===")
            for i, figure in enumerate(results.figures, 1):
                formatted.append(f"\n--- Figure {figure.figure_number or i} (Page {figure.page_number}) ---")
                if figure.caption:
                    formatted.append(f"Caption: {figure.caption}")
                formatted.append(f"Image size: {len(figure.image_data)} bytes")
                # Note: Images are available for vision models
                formatted.append("[Image data available for analysis]")
                formatted.append("")
        
        return "\n".join(formatted)
    
    def _prepare_images_for_llm(self, figures: List[Figure]) -> List[Dict[str, str]]:
        """
        Convert figure images to base64 for vision-capable LLMs.
        Returns list of dicts with figure metadata and base64 image data.
        """
        images = []
        for i, figure in enumerate(figures, 1):
            try:
                # Encode image as base64
                image_b64 = base64.b64encode(figure.image_data).decode('utf-8')
                
                images.append({
                    'figure_number': figure.figure_number or str(i),
                    'page_number': figure.page_number,
                    'caption': figure.caption or '',
                    'image_base64': image_b64
                })
            except Exception as e:
                self.logger.warning(f"Failed to encode figure {i}: {e}")
                continue
        
        return images
    
    @staticmethod
    def _extract_structured_block(agent_response: Any) -> Optional[Any]:
        """Extract data from agent's final_answer() response."""
        if not agent_response:
            return None
        
        # smolagents returns dict directly from final_answer()
        if isinstance(agent_response, dict):
            return agent_response
        
        # If string, try to extract JSON
        if isinstance(agent_response, str):
            # Try markdown code block
            code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', agent_response)
            if code_block_match:
                return code_block_match.group(1)
            
            # Try to find JSON object
            json_match = re.search(r'\{[\s\S]*\}', agent_response)
            if json_match:
                return json_match.group(0)
        
        return None
    
    @staticmethod
    def _parse_structured_dict(block: Any) -> Optional[Dict[str, Any]]:
        """Parse data from agent response."""
        if not block:
            return None
        
        # Already a dict
        if isinstance(block, dict):
            return block
        
        # Try JSON parsing
        if isinstance(block, str):
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass
            
            # Fallback to ast.literal_eval
            try:
                parsed = ast.literal_eval(block)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError, MemoryError):
                pass
        
        return None


def parse_paper(pdf_path: str) -> Dict[str, Any]:
    agent = PaperParserAgent()
    result = agent.execute({'pdf_path': pdf_path})
    
    return result


