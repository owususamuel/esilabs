
import ast
import json
import re
from typing import Dict, Any, Optional
from pathlib import Path

from scientist.agents.base_agent import BaseAgent
from scientist.tools.pdf_parser import PDFParser
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
            self.logger.info(f"🤖 Autonomous agent analyzing paper: {pdf_path}")
            
            # Load task prompt from YAML
            task_prompt_template = BaseAgent._load_task_prompt('paper_parser_agent')
            if not task_prompt_template:
                raise ValueError(
                    "Failed to load task prompt for paper_parser_agent from config/agent_instructions.yaml. "
                    "Please ensure the configuration file exists and contains the 'task_prompt' field."
                )
            
            # Fill in template variables
            agent_task = task_prompt_template.format(pdf_path=pdf_path)
            
            # Agent works autonomously
            self.logger.info("Agent is now parsing and analyzing the paper...")
            agent_response = self.call_llm(agent_task)
            
            self.logger.info("✅ Agent completed autonomous paper analysis")
            
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
                
                # Extract experimental results and findings
                experimental_results = normalize_string_field(extracted_data.get('experimental_results', '') or extracted_data.get('results', ''))
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
                    'experimental_results': experimental_results,
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
                    'execution_type': 'autonomous',
                    'note': 'Agent autonomously parsed PDF and extracted structured information'
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
    
    @staticmethod
    def _extract_structured_block(agent_response: str) -> Optional[str]:
        
        if not agent_response:
            return None
        
        def extract_from_index(text: str, start_index: int) -> Optional[str]:
            brace_count = 0
            in_string: Optional[str] = None
            escape_next = False
            
            for idx in range(start_index, len(text)):
                char = text[idx]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if in_string:
                    if char == in_string:
                        in_string = None
                    elif char in ('\n', '\r'):
                        # Strings can span lines; keep state
                        pass
                    continue
                
                if char in ('"', "'"):
                    in_string = char
                    continue
                
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start_index:idx + 1]
            
            return None
        
        final_match = re.search(r'Final answer:\s*(\{)', agent_response, re.IGNORECASE)
        if final_match:
            block = extract_from_index(agent_response, final_match.start(1))
            if block:
                return block
        
        code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', agent_response)
        if code_block_match:
            return code_block_match.group(1)
        
        for brace_match in re.finditer(r'\{', agent_response):
            block = extract_from_index(agent_response, brace_match.start())
            if block:
                return block
        
        return None
    
    @staticmethod
    def _sanitize_block_for_literal(block: str) -> str:
        """
        Convert multiline strings within single/double quotes to use explicit newline escape sequences.
        This helps ast.literal_eval handle strings that the LLM emitted across multiple lines.
        """
        
        sanitized_chars = []
        in_single = False
        in_double = False
        escape_next = False
        
        for char in block:
            if escape_next:
                sanitized_chars.append(char)
                escape_next = False
                continue
            
            if char == '\\':
                sanitized_chars.append(char)
                escape_next = True
                continue
            
            if in_single:
                if char == "'":
                    in_single = False
                    sanitized_chars.append(char)
                elif char in ('\n', '\r'):
                    sanitized_chars.append('\\n')
                else:
                    sanitized_chars.append(char)
                continue
            
            if in_double:
                if char == '"':
                    in_double = False
                    sanitized_chars.append(char)
                elif char in ('\n', '\r'):
                    sanitized_chars.append('\\n')
                else:
                    sanitized_chars.append(char)
                continue
            
            if char == "'":
                in_single = True
            elif char == '"':
                in_double = True
            
            sanitized_chars.append(char)
        
        return ''.join(sanitized_chars)
    
    @staticmethod
    def _parse_structured_dict(block: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse the extracted structured block into a dictionary.
        Handles both JSON and Python literal formats, including multiline strings.
        """
        
        if not block:
            return None
        
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass
        
        sanitized_block = PaperParserAgent._sanitize_block_for_literal(block)
        
        try:
            return json.loads(sanitized_block)
        except json.JSONDecodeError:
            pass
        
        for candidate in (sanitized_block, block):
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError, MemoryError):
                continue
        
        return None


def parse_paper(pdf_path: str) -> Dict[str, Any]:
    agent = PaperParserAgent()
    result = agent.execute({'pdf_path': pdf_path})
    
    return result


