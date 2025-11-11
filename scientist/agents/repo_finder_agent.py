import ast
import json
import re
from typing import Dict, Any, Optional

from scientist.agents.base_agent import BaseAgent
from scientist.tools.github_search import RepositoryFinder


class RepoFinderAgent(BaseAgent):
    """
    Agent that helps identify potential repositories for a research paper.
    
    The agent autonomously:
    1. Generates search queries from paper metadata
    2. Builds GitHub search URLs
    3. Provides queries and URLs for manual repository searching
    4. Returns search assistance with reasoning
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
        self.query_builder = RepositoryFinder()
    
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
            agent_task = BaseAgent._render_template(
                task_prompt_template,
                {
                    "paper_title": paper_title,
                    "author_str": author_str,
                    "keyword_str": keyword_str
                }
            )
            
            # Agent works autonomously
            self.logger.info("Agent is now generating search queries for repository discovery...")
            agent_response = self.call_llm(agent_task)
            
            self.logger.info("✅ Agent completed search query generation")
            
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
