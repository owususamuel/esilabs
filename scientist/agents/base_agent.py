
#Uses smolagents for orchestration and LLM integration.

import logging
import os
import json
from typing import Any, Dict, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime
import yaml
import requests

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents in the reproducibility pipeline.
    
    This class provides:
    1. Common LLM client setup using smolagents
    2. Tool management interface
    3. Logging and error handling
    4. Agent state management
    """
    
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        config_path: str = "config/agent_config.yaml"
    ):
        
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(f"reproducibility_agent.{agent_name}")
        self.tools = {}
        self.smolagent_tools = []
        self.execution_history = []
        self.agent_instance = None
        self.model = None
        self._structured_outputs_supported = False
        self._last_executions = []  # Track recent executions for loop detection
        
        # Load configuration
        self.config = self._load_config(config_path)
        self._setup_llm_client()

        # Retry configuration for local/non-compliant models
        try:
            self._max_code_retries = max(1, int(os.getenv("LLM_CODE_RETRIES", "3")))
        except ValueError:
            self._max_code_retries = 3
        self._code_retry_hints = [
            (
                "\n\nIMPORTANT: You MUST respond with ONLY executable Python code inside a ```python``` code block. "
                "Do NOT write thoughts, explanations, or natural language outside the code block. "
                "ALL top-level statements MUST start at column 0 (NO leading spaces). "
                "NEVER EVER use backslash (\\) for line continuation - write complete statements on one line. "
                "Example:\n```python\ndata = parse_pdf('file.pdf')\nresult = {'key': 'value'}\nfinal_answer(result)\n```"
            ),
            (
                "\n\nCRITICAL ERROR DETECTED: Your code has syntax errors. Follow these rules EXACTLY:\n"
                "1. NO backslash (\\) characters anywhere in your code\n"
                "2. NO leading spaces - start all lines at column 0\n"
                "3. Write complete statements on ONE line\n"
                "4. Close all brackets on the same line they open\n"
                "Example:\n```python\ndata = tool()\nresult = {'a': 'b', 'c': 'd'}\nfinal_answer(result)\n```"
            ),
            (
                "\n\n⚠️ FINAL ATTEMPT - Use this EXACT format:\n"
                "```python\n"
                "content = parse_pdf('path.pdf')\n"
                "data = {'title': 'text', 'authors': ['name']}\n"
                "final_answer(data)\n"
                "```\n"
                "Rules: (1) No backslashes (\\), (2) No leading spaces, (3) Complete lines, (4) Simple values only"
            ),
        ]
    
    @staticmethod
    def _load_agent_instructions(agent_key: str, instructions_path: str = "config/agent_instructions.yaml") -> str:
        
        if not os.path.exists(instructions_path):
            logger.warning(f"Agent instructions file not found: {instructions_path}")
            return ""
        
        try:
            with open(instructions_path, 'r') as f:
                instructions = yaml.safe_load(f)
            
            agent_config = instructions.get(agent_key, {})
            system_prompt = agent_config.get('system_prompt', '')
            output_format = agent_config.get('output_format', '')
            
            # Combine system prompt and output format if both exist
            if system_prompt and output_format:
                return f"{system_prompt}\n\n{output_format}"
            elif system_prompt:
                return system_prompt
            else:
                logger.warning(f"No system prompt found for agent: {agent_key}")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to load agent instructions from {instructions_path}: {e}")
            return ""
    
    @staticmethod
    def _load_task_prompt(agent_key: str, instructions_path: str = "config/agent_instructions.yaml") -> str:
        
        if not os.path.exists(instructions_path):
            logger.warning(f"Agent instructions file not found: {instructions_path}")
            return ""
        
        try:
            with open(instructions_path, 'r') as f:
                instructions = yaml.safe_load(f)
            
            agent_config = instructions.get(agent_key, {})
            task_prompt = agent_config.get('task_prompt', '')
            
            if not task_prompt:
                logger.warning(f"No task prompt found for agent: {agent_key}")
            
            return task_prompt
                
        except Exception as e:
            logger.error(f"Failed to load task prompt from {instructions_path}: {e}")
            return ""
    
    @staticmethod
    def _render_template(template: str, variables: Optional[Dict[str, Any]]) -> str:
        """
        Safely interpolate {placeholders} in a template without disturbing other
        brace-delimited content that should remain literal (e.g. JSON examples).
        """
        if not template or not variables:
            return template or ""
        
        result = template
        for key, value in variables.items():
            replacement = "" if value is None else str(value)
            result = result.replace(f"{{{key}}}", replacement)
        return result

    def _should_retry_due_to_parse_error(self, error_text: str) -> bool:
        """Heuristic detection for parse errors coming from non-code LLM replies."""
        if not error_text:
            return False
        
        lowered = error_text.lower()
        triggers = [
            "code parsing failed",
            "syntaxerror",
            "there is no code in this text",
            "expected python",
            "no code block found",
            "regex pattern",
            "code snippet is invalid",
            "was not found in it",
            "perhaps you forgot a comma",  # Common with backslash continuations
            "invalid syntax",  # General syntax errors
            "indentationerror",  # Indentation errors
            "unexpected indent",  # Leading spaces issue
            "unindent does not match",  # Mismatched indentation
            "was never closed",  # Unclosed brackets/parens
            "unexpected character after line continuation",  # Backslash issues
            "line continuation character",  # Backslash problems
        ]
        return any(trigger in lowered for trigger in triggers)

    def _reinforce_code_prompt(self, base_prompt: str, attempt_index: int) -> str:
        """Append progressively stricter instructions to coerce code-style outputs."""
        hint_idx = min(max(attempt_index - 1, 0), len(self._code_retry_hints) - 1)
        reminder = self._code_retry_hints[hint_idx]
        return f"{base_prompt}\n\n{reminder}"
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        
        return {
            'model': {
                'provider': os.getenv('MODEL_PROVIDER', 'openai'),
                'name': os.getenv('MODEL_NAME', 'gpt-4'),
                'api_key': os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY'),
                'temperature': 0.7,
                'max_tokens': 2048
            },
            'azure': {
                'endpoint': os.getenv('AZURE_ENDPOINT', ''),
                'api_key': os.getenv('AZURE_API_KEY', ''),
                'deployment_name': os.getenv('AZURE_DEPLOYMENT_NAME', ''),
            },
            'ollama': {
                'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            }
        }
    
    def _setup_llm_client(self):
        """Setup LLM client using environment variables."""
        try:
            from smolagents import (
                OpenAIServerModel,
                AzureOpenAIServerModel,
                LiteLLMModel,
                CodeAgent
            )
            
            provider = os.getenv('MODEL_PROVIDER', 'openai').lower()
            model_name = os.getenv('MODEL_NAME', 'gpt-4')
            
            # Get agent-specific configuration
            agent_config = self.config.get('agents', {}).get(self.agent_name, {})
            
            # Extract max_tokens and temperature for this agent
            max_tokens = agent_config.get('max_tokens')
            temperature = agent_config.get('model_temperature')
            
            if max_tokens:
                self.logger.info(f"Using max_tokens={max_tokens} for {self.agent_name}")
            if temperature is not None:
                self.logger.info(f"Using temperature={temperature} for {self.agent_name}")
            
            structured_supported = False

            if provider == 'openai':
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    # Build kwargs for model
                    model_kwargs = {'model_id': model_name, 'api_key': api_key}
                    if max_tokens:
                        model_kwargs['max_tokens'] = max_tokens
                    if temperature is not None:
                        model_kwargs['temperature'] = temperature
                    
                    self.model = OpenAIServerModel(**model_kwargs)
                    self.logger.info(f"Initialized OpenAI model: {model_name}")
                    structured_supported = True
                else:
                    self.logger.warning("OPENAI_API_KEY not found in environment")
                    self.model = None
            
            elif provider == 'azure':
                endpoint = os.getenv('AZURE_ENDPOINT')
                api_key = os.getenv('AZURE_API_KEY')
                api_version = os.getenv('AZURE_API_VERSION', '2024-02-15-preview')
                deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME', model_name)
                
                if endpoint and api_key:
                    # Build kwargs for Azure model
                    model_kwargs = {
                        'model_id': deployment_name,
                        'azure_endpoint': endpoint,
                        'api_key': api_key,
                        'api_version': api_version
                    }
                    if max_tokens:
                        model_kwargs['max_tokens'] = max_tokens
                    if temperature is not None:
                        model_kwargs['temperature'] = temperature
                    
                    self.model = AzureOpenAIServerModel(**model_kwargs)
                    self.logger.info(f"Initialized Azure OpenAI model: {deployment_name}")
                    structured_supported = self._azure_supports_structured_outputs(api_version)
                else:
                    self.logger.warning("Azure OpenAI credentials not found in environment")
                    self.model = None
            
            elif provider == 'anthropic':
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if api_key:
                    # Build kwargs for Anthropic model
                    model_kwargs = {
                        'model_id': f"anthropic/{model_name}",
                        'api_key': api_key
                    }
                    if max_tokens:
                        model_kwargs['max_tokens'] = max_tokens
                    if temperature is not None:
                        model_kwargs['temperature'] = temperature
                    
                    self.model = LiteLLMModel(**model_kwargs)
                    self.logger.info(f"Initialized Anthropic model via LiteLLM: {model_name}")
                    structured_supported = True
                else:
                    self.logger.warning("ANTHROPIC_API_KEY not found in environment")
                    self.model = None
            
            elif provider == 'ollama':
                base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                # Build kwargs for Ollama model
                model_kwargs = {
                    'model_id': f"ollama/{model_name}",
                    'api_base': base_url
                }
                if max_tokens:
                    model_kwargs['max_tokens'] = max_tokens
                if temperature is not None:
                    model_kwargs['temperature'] = temperature
                
                self.model = LiteLLMModel(**model_kwargs)
                self.logger.info(f"Initialized Ollama local model: {model_name} at {base_url}")
                structured_supported = True
            
            elif provider == 'lmstudio':
                base_url = os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1')
                # Validate model exists in LM Studio
                try:
                    models_url = f"{base_url}/models"
                    response = requests.get(models_url, timeout=5)
                    response.raise_for_status()
                    models_data = response.json()
                    available_models = [m.get('id', '') for m in models_data.get('data', [])]
                    if model_name not in available_models:
                        raise ValueError(
                            f"Model '{model_name}' not found in LM Studio. "
                            f"Available models: {available_models}"
                        )
                except requests.exceptions.RequestException as e:
                    raise ConnectionError(f"Cannot connect to LM Studio at {base_url}: {e}")
                
                # Build kwargs for LM Studio model
                model_kwargs = {
                    'model_id': f"openai/{model_name}",
                    'api_base': base_url
                }
                if max_tokens:
                    model_kwargs['max_tokens'] = max_tokens
                if temperature is not None:
                    model_kwargs['temperature'] = temperature
                
                self.model = LiteLLMModel(**model_kwargs)
                self.logger.info(f"Initialized LM Studio model: {model_name} at {base_url}")
                structured_supported = True
            
            else:
                self.logger.error(f"Unknown provider: {provider}")
                self.model = None
            
            if self.model:
                self._structured_outputs_supported = structured_supported
                self.agent_instance = CodeAgent(
                    tools=self.smolagent_tools,
                    model=self.model,
                    instructions=self.system_prompt,
                    additional_authorized_imports=['json', 'numpy', 'pandas'],
                    max_steps=30,
                    code_block_tags="markdown",
                    add_base_tools=False,  # Disable base tools to avoid ddgs dependency
                    use_structured_outputs_internally=structured_supported
                )
        
        except ImportError as e:
            self.logger.warning(f"smolagents not available: {e}")
            self.model = None
        except Exception as e:
            self.logger.error(f"Failed to setup LLM client: {e}")
            self.model = None
    
    def register_tool(self, tool_name: str, tool_function: callable, tool_instance: Optional[Any] = None):
        
        self.tools[tool_name] = tool_function
        
        if tool_instance:
            self.smolagent_tools.append(tool_instance)
            if self.model:
                from smolagents import CodeAgent
                self.agent_instance = CodeAgent(
                    tools=self.smolagent_tools,
                    model=self.model,
                    instructions=self.system_prompt,
                    additional_authorized_imports=['json', 'numpy', 'pandas'],
                    max_steps=30,
                    code_block_tags="markdown",
                    add_base_tools=False,  # Disable base tools to avoid ddgs dependency
                    use_structured_outputs_internally=self._structured_outputs_supported
                )
        
        self.logger.info(f"Registered tool: {tool_name}")

    @staticmethod
    def _azure_supports_structured_outputs(api_version: str) -> bool:
        """
        Azure only supports JSON schema responses on API versions 2024-08-01-preview and later.
        """
        if not api_version:
            return False
        # Expect strings like "2024-08-01-preview". Extract the date component.
        date_part = api_version.split('-preview')[0]
        tokens = date_part.split('-')
        if len(tokens) < 3:
            return False
        try:
            year, month, day = (int(tokens[0]), int(tokens[1]), int(tokens[2]))
        except ValueError:
            return False
        return (year, month, day) >= (2024, 8, 1)
    
    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main task. Must be implemented by subclasses.
        
        Args:
            task: Task dictionary with parameters
            
        Returns:
            Result dictionary
        """
        pass
    
    async def execute_async(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async version of execute. Override if supporting async operations.
        
        Args:
            task: Task dictionary with parameters
            
        Returns:
            Result dictionary
        """
        
        # Default implementation calls sync version
        return self.execute(task)
    
    def call_llm(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Any:
        
        if not self.agent_instance:
            self.logger.warning("Agent not available, returning mock response")
            return "Mock LLM response"
        
        import time
        import hashlib
        
        base_prompt = prompt
        prompt_to_use = base_prompt
        attempt = 1
        
        while attempt <= self._max_code_retries:
            try:
                start_time = time.time()
                
                # Run the agent with the prompt
                result = self.agent_instance.run(prompt_to_use)
                
                duration = time.time() - start_time
                
                # Detect repeated identical executions (infinite loop detection)
                if hasattr(self.agent_instance, 'logs') and self.agent_instance.logs:
                    # Extract executed code from recent logs
                    recent_code = []
                    for log in self.agent_instance.logs[-10:]:  # Check last 10 steps
                        if hasattr(log, 'content') and log.content:
                            code_hash = hashlib.md5(str(log.content).encode()).hexdigest()
                            recent_code.append(code_hash)
                    
                    # Check for repeated patterns (same code executed 3+ times)
                    if len(recent_code) >= 6:
                        last_three = recent_code[-3:]
                        prev_three = recent_code[-6:-3]
                        if last_three == prev_three:
                            self.logger.error("⚠️  INFINITE LOOP DETECTED: Agent is repeating the same execution!")
                            self.logger.error("Recent execution hashes: " + str(recent_code[-6:]))
                            raise RuntimeError(
                                "Agent stuck in infinite loop - repeated identical executions detected. "
                                "This suggests the agent doesn't understand its goal or completion criteria."
                            )
                
                # Try to extract token usage if available
                tokens_used = None
                if hasattr(self.agent_instance, 'logs') and self.agent_instance.logs:
                    for log in self.agent_instance.logs:
                        if hasattr(log, 'token_usage') and log.token_usage:
                            tokens_used = log.token_usage
                            break
                
                self.logger.info(
                    f"LLM call completed - Duration: {duration:.2f}s, Response length: {len(str(result))} chars"
                )
                if tokens_used:
                    self.logger.info(f"Token usage: {tokens_used}")
                
                # Return result directly - smolagents returns dicts from final_answer()
                return result
            
            except Exception as e:
                error_text = str(e)
                should_retry = (
                    self._should_retry_due_to_parse_error(error_text)
                    and attempt < self._max_code_retries
                )
                
                if should_retry:
                    self.logger.warning(
                        "LLM output was not executable (attempt %d/%d): %s",
                        attempt,
                        self._max_code_retries,
                        error_text.splitlines()[0] if error_text else e.__class__.__name__
                    )
                    prompt_to_use = self._reinforce_code_prompt(base_prompt, attempt)
                    attempt += 1
                    continue
                
                self.logger.error(f"Error calling agent: {e}")
                raise
    
    def log_execution(self, step_name: str, result: Dict[str, Any]):
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'result': result
        }
        
        self.execution_history.append(entry)
        self.logger.info(f"Logged step: {step_name}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        return {
            'agent': self.agent_name,
            'total_steps': len(self.execution_history),
            'steps': self.execution_history,
            'status': 'completed' if self.execution_history else 'not_started'
        }
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.agent_name})>"
