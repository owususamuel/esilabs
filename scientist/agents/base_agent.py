
#Uses smolagents for orchestration and LLM integration.

import logging
import os
import json
from typing import Any, Dict, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime
import yaml

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
        
        # Load configuration
        self.config = self._load_config(config_path)
        self._setup_llm_client()
    
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
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Substitute environment variables in config
                config = self._substitute_env_vars(config)
                
                self.logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return self._get_default_config()
    
    def _substitute_env_vars(self, config: Any) -> Any:
        
        import re
        
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Pattern to match ${VAR_NAME}
            pattern = r'\$\{([^}]+)\}'
            
            def replace_env(match):
                env_var = match.group(1)
                value = os.getenv(env_var)
                if value is None:
                    self.logger.warning(f"Environment variable {env_var} not set, keeping placeholder")
                    return match.group(0)  # Keep original ${VAR_NAME}
                return value
            
            return re.sub(pattern, replace_env, config)
        else:
            return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        
        return {
            'model': {
                'provider': os.getenv('MODEL_PROVIDER', 'openai'),  # openai, anthropic, azure, ollama
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
        
        try:
            from smolagents import (
                OpenAIServerModel,
                AzureOpenAIServerModel,
                LiteLLMModel,
            )
            
            model_config = self.config.get('model', {})
            provider = model_config.get('provider', 'openai').lower()
            model_name = model_config.get('name', 'gpt-4')
            api_key = model_config.get('api_key')
            
            # Setup model based on provider
            if provider == 'openai':
                api_key = api_key or os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.model = OpenAIServerModel(
                        model_id=model_name,
                        api_key=api_key
                    )
                    self.logger.info(f"Initialized OpenAI model: {model_name}")
                else:
                    self.logger.warning("OpenAI API key not configured")
                    self.model = None
            
            elif provider == 'azure':
                # Azure OpenAI uses AzureOpenAIServerModel
                azure_config = self.config.get('azure', {})
                endpoint = azure_config.get('endpoint')
                api_key = azure_config.get('api_key') or api_key
                api_version = azure_config.get('api_version', '2024-02-15-preview')
                # For Azure, use deployment_name if available, otherwise fall back to model_name
                deployment_name = azure_config.get('deployment_name', model_name)
                
                if endpoint and api_key:
                    self.model = AzureOpenAIServerModel(
                        model_id=deployment_name,
                        azure_endpoint=endpoint,
                        api_key=api_key,
                        api_version=api_version
                    )
                    self.logger.info(f"Initialized Azure OpenAI model: {deployment_name}")
                else:
                    self.logger.warning("Azure OpenAI credentials not configured")
                    self.model = None
            
            elif provider == 'anthropic':
                api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
                if api_key:
                    # Use LiteLLMModel for Anthropic models
                    self.model = LiteLLMModel(
                        model_id=f"anthropic/{model_name}",
                        api_key=api_key
                    )
                    self.logger.info(f"Initialized Anthropic model via LiteLLM: {model_name}")
                else:
                    self.logger.warning("Anthropic API key not configured")
                    self.model = None
            
            elif provider == 'ollama':
                # Ollama local LLM server integration
                ollama_config = self.config.get('ollama', {})
                base_url = ollama_config.get('base_url', os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'))
                
                # Use LiteLLMModel with ollama/ prefix
                self.model = LiteLLMModel(
                    model_id=f"ollama/{model_name}",
                    api_base=base_url,
                )
                self.logger.info(f"Initialized Ollama local model: {model_name} at {base_url}")
            
            else:
                self.logger.error(f"Unknown provider: {provider}")
                self.model = None
            
            # Create agent instance with model
            if self.model:
                from smolagents import CodeAgent
                self.agent_instance = CodeAgent(
                    tools=self.smolagent_tools,
                    model=self.model,
                    instructions=self.system_prompt,
                    additional_authorized_imports=['json', 'numpy', 'pandas'],
                    max_steps=50 
                )
        
        except ImportError as e:
            self.logger.warning(f"smolagents not available: {e}")
            self.model = None
        except Exception as e:
            self.logger.error(f"Failed to setup LLM client: {e}")
            self.model = None
    
    def register_tool(self, tool_name: str, tool_function: callable, tool_instance: Optional[Any] = None):
        
        self.tools[tool_name] = tool_function
        
        # If a smolagents Tool instance is provided, add it
        if tool_instance:
            self.smolagent_tools.append(tool_instance)
            # Recreate agent with new tools
            if self.model:
                from smolagents import CodeAgent
                self.agent_instance = CodeAgent(
                    tools=self.smolagent_tools,
                    model=self.model,
                    instructions=self.system_prompt,
                    additional_authorized_imports=['json', 'numpy', 'pandas'],
                    max_steps=50  # Increased from default 20 to allow more exploration
                )
        
        self.logger.info(f"Registered tool: {tool_name}")
    
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
    ) -> str:
        
        if not self.agent_instance:
            self.logger.warning("Agent not available, returning mock response")
            return "Mock LLM response"
        
        try:
            # Run the agent with the prompt
            result = self.agent_instance.run(prompt)
            self.logger.debug(f"Agent response received ({len(str(result))} chars)")
            
            return str(result)
        
        except Exception as e:
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
