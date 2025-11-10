import ast
import json
import re
from typing import Dict, Any, Optional

from scientist.agents.base_agent import BaseAgent
from scientist.tools.github_search import RepositoryFinder
from scientist.tools.tool_wrappers import SearchGitHub


class RepoFinderAgent(BaseAgent):
    """
    Autonomous agent that finds the best repository for a research paper.
    
    The agent autonomously:
    1. Searches GitHub for matching repositories
    2. Analyzes repository metadata (stars, descriptions, authors)
    3. Reasons about which is most likely the official implementation
    4. Returns the best match with confidence score
    """
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):        
        # Load system prompt from YAML configuration
        system_prompt = BaseAgent._load_agent_instructions('repo_finder_agent')
        
        # Fail fast if configuration is missing
        if not system_prompt:
            raise ValueError(
                "Failed to load system prompt for repo_finder_agent from config/agent_instructions.yaml. "
                "Please ensure the configuration file exists and contains the 'repo_finder_agent' section."
            )
        
        super().__init__(
            agent_name="repo_finder",
            system_prompt=system_prompt,
            config_path=config_path
        )
        
        # Initialize backing tools
        self.github_finder = RepositoryFinder()
        
        # Register autonomous tools
        self.register_tool("search_github", None, SearchGitHub(self.github_finder))
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        
        paper_title = task.get('paper_title')
        authors = task.get('authors', [])
        keywords = task.get('keywords', [])
        
        if not paper_title:
            return {
                'success': False,
                'error': 'paper_title is required'
            }
        
        try:
            self.logger.info(f"🤖 Autonomous agent searching for repository: {paper_title}")
            
            # Build search context
            author_str = ", ".join(authors[:3]) if authors else "not provided"
            keyword_str = ", ".join(keywords) if keywords else "not provided"
            
            # Load task prompt from YAML
            task_prompt_template = BaseAgent._load_task_prompt('repo_finder_agent')
            if not task_prompt_template:
                raise ValueError(
                    "Failed to load task prompt for repo_finder_agent from config/agent_instructions.yaml. "
                    "Please ensure the configuration file exists and contains the 'task_prompt' field."
                )
            
            # Fill in template variables
            agent_task = task_prompt_template.format(
                paper_title=paper_title,
                author_str=author_str,
                keyword_str=keyword_str
            )
            
            # Agent works autonomously
            self.logger.info("Agent is now searching GitHub and analyzing repositories...")
            agent_response = self.call_llm(agent_task)
            
            self.logger.info("✅ Agent completed autonomous repository search")
            
            # Extract structured data from agent response
            structured_block = self._extract_structured_block(agent_response)
            repo_data = None
            
            if structured_block:
                self.logger.debug(f"Structured block detected in agent response ({len(structured_block)} chars)")
                repo_data = self._parse_structured_dict(structured_block)
                if repo_data is None:
                    self.logger.warning("Structured block found but parsing failed")
            else:
                self.logger.warning("No structured data block detected in agent response")
            
            if repo_data:
                result = {
                    'success': True,
                    'paper_title': paper_title,
                    'selected_repository': repo_data.get('selected_repository', {}),
                    'alternatives': repo_data.get('alternatives', []),
                    'execution_type': 'autonomous',
                    'note': 'Agent autonomously searched and selected repository'
                }
            else:
                result = {
                    'success': False,
                    'error': 'Could not extract structured data from agent response',
                    'paper_title': paper_title,
                    'agent_response': agent_response,
                    'structured_block': structured_block,
                    'execution_type': 'autonomous',
                    'note': 'Agent completed but JSON parsing failed, check agent_response'
                }
            
            self.log_execution("autonomous_search", result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in autonomous execution: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'paper_title': paper_title
            }
    
    @staticmethod
    def _extract_structured_block(agent_response: str) -> Optional[str]:
        """
        Extract the first balanced JSON/Python dict-like block from the agent response.
        Prefers blocks following 'Final answer:' but falls back to any balanced braces.
        """
        
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
        
        sanitized_block = RepoFinderAgent._sanitize_block_for_literal(block)
        
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


def find_repository(paper_title: str, authors: Optional[list] = None) -> Dict[str, Any]:
    
    agent = RepoFinderAgent()
    result = agent.execute({
        'paper_title': paper_title,
        'authors': authors or []
    })
    
    return result


def find_paper_repository(
    paper_title: str,
    authors: Optional[list] = None
) -> Dict[str, Any]:

    agent = RepoFinderAgent()
    agent_result = agent.execute({
        'paper_title': paper_title,
        'authors': authors or []
    })

    if not agent_result.get('success'):
        return agent_result

    best_repo = agent_result.get('selected_repository') or {}

    # Adapt keys expected by scripts/docs
    adapted = {
        'success': True,
        'paper_title': paper_title,
        'best_repository': best_repo,
        'alternative_repositories': agent_result.get('alternatives', []),
        'execution_type': agent_result.get('execution_type', 'autonomous'),
        'note': agent_result.get('note')
    }
    return adapted
