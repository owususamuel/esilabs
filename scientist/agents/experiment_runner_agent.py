import ast
import json
import os
import re
import shutil
import hashlib
import shlex
import time
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from pathlib import Path
from datetime import datetime

if TYPE_CHECKING:
    from scientist.utils.interactive import InteractiveInputHandler

from scientist.agents.base_agent import BaseAgent
from scientist.tools.code_executor import CodeExecutor
from scientist.tools.result_comparator import ResultComparator
from scientist.tools.tool_wrappers import (
    ReadFileContents,
    ListDirectoryFiles,
    RunCommandInRepo,
    CreateFileOrDirectory,
    ExtractMetrics,
    RequestCredentials
)


class ExperimentRunnerAgent(BaseAgent):
    """
    Autonomous agent that runs code from research repositories.
    
    The agent autonomously:
    1. Explores the repository structure
    2. Reads documentation (README, help text)
    3. Determines how to run the code
    4. Creates necessary files/directories
    5. Executes experiments
    6. Extracts and returns metrics
    
    All experiments run in data/temp/experiments for isolation and consistency.
    """
    
    # Resolve project root relative to this file
    ROOT_DIR = Path(__file__).resolve().parents[2]
    
    # Working directory for all experiments (absolute path)
    WORK_DIR = ROOT_DIR / "data" / "temp" / "experiments"
    
    # Cache directory for reusable venvs (absolute path shared across repos)
    VENV_CACHE_DIR = ROOT_DIR / "data" / "temp" / "venv_cache"
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        """Initialize the Experiment Runner Agent with autonomous tools."""
        
        # Load system prompt from YAML configuration
        system_prompt = BaseAgent._load_agent_instructions('experiment_runner_agent')
        
        # Fail fast if configuration is missing
        if not system_prompt:
            raise ValueError(
                "Failed to load system prompt for experiment_runner_agent from config/agent_instructions.yaml. "
                "Please ensure the configuration file exists and contains the 'experiment_runner_agent' section."
            )
        
        super().__init__(
            agent_name="experiment_runner",
            system_prompt=system_prompt,
            config_path=config_path
        )
        
        # Initialize backing tools
        self.executor = CodeExecutor(sandbox_mode=True, max_timeout=300)
        self.comparator = ResultComparator()
        
        # Register autonomous tools
        self.read_file_tool = ReadFileContents()
        self.list_dir_tool = ListDirectoryFiles()
        self.run_command_tool = RunCommandInRepo()
        
        # Import ExtractMetrics tool
        from scientist.tools.tool_wrappers import ExtractMetrics
        self.create_tool = CreateFileOrDirectory()
        self.extract_metrics_tool = ExtractMetrics(self.comparator, llm_client=self.model)
        self.request_credentials_tool = RequestCredentials()
        
        self.register_tool("read_file_contents", None, self.read_file_tool)
        self.register_tool("list_directory_files", None, self.list_dir_tool)
        self.register_tool("extract_metrics", None, self.extract_metrics_tool)
        self.register_tool("run_command_in_repo", None, self.run_command_tool)
        self.register_tool("create_file_or_directory", None, self.create_tool)
        self.register_tool("request_credentials", None, self.request_credentials_tool)
        
        # Store reference to interactive handler (will be set in execute)
        self._interactive_handler = None
        
        # Ensure cache directory exists
        self.VENV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Detect what's available on the system
        self._has_uv = self._check_command("uv --version")
        self._has_pip3 = self._check_command("pip3 --version")
        
        # Detect how Python should be invoked on this system
        self._python_invocation = self._detect_python_invocation()
    
    def _check_command(self, cmd: str) -> bool:
        """Check if a command is available."""
        try:
            return self.executor.execute_command(cmd).success
        except Exception:
            return False
    
    def _detect_python_invocation(self) -> str:
        """
        Detect how Python should be invoked on the host system.
        Tries various methods and returns the working command.
        """
        # Priority order: uv run, python3, python
        candidates = [
            ("uv run python", "uv run python --version"),
            ("python3", "python3 --version"),
            ("python", "python --version"),
        ]
        
        for invocation, test_cmd in candidates:
            if self._check_command(test_cmd):
                self.logger.info(f"Detected Python invocation method: {invocation}")
                return invocation
        
        # Fallback to python3 if nothing works
        self.logger.warning("Could not detect Python invocation, defaulting to 'python3'")
        return "python3"
    
    def check_environment_requirements(self, repo_path: Path) -> Dict[str, Any]:
        
        self.logger.info(f"Checking environment requirements for {repo_path}")
        
        required_vars = []
        missing_vars = []
        
        try:
            env_example = repo_path / '.env.example'
            if env_example.exists():
                content = env_example.read_text(encoding='utf-8', errors='ignore')
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key = line.split('=')[0].strip()
                        if key:
                            required_vars.append({'key': key, 'description': ''})
                            if not os.getenv(key):
                                missing_vars.append(key)
            
            if missing_vars:
                self.logger.warning(f"Missing environment variables: {missing_vars}")
            
            return {
                'required_vars': required_vars,
                'missing_vars': missing_vars,
                'found_in': ['.env.example'] if required_vars else [],
                'has_missing': len(missing_vars) > 0
            }
        
        except Exception as e:
            self.logger.error(f"Error checking environment requirements: {e}")
            return {
                'required_vars': [],
                'missing_vars': [],
                'found_in': [],
                'has_missing': False
            }
    
    def _copy_local_repo(self, source_path: str) -> Optional[Path]:
        try:
            source = Path(source_path).resolve()
            if not source.exists():
                self.logger.error(f"Source repository not found: {source}")
                return None
            
            # Create work directory
            self.WORK_DIR.mkdir(parents=True, exist_ok=True)
            
            # Create unique destination name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            repo_name = source.name
            dest_name = f"{repo_name}_{timestamp}"
            dest_path = self.WORK_DIR / dest_name
            
            self.logger.info(f"Copying local repository from {source} to {dest_path}")
            
            # Copy the repository
            shutil.copytree(
                source,
                dest_path,
                ignore=shutil.ignore_patterns(
                    '__pycache__', '*.pyc', '.git', '.env', 
                    'venv', 'env', '.venv', 'node_modules',
                    '*.egg-info', '.DS_Store'
                ),
                symlinks=False
            )
            
            self.logger.info(f"Repository copied successfully to: {dest_path}")
            
            return dest_path
            
        except Exception as e:
            self.logger.error(f"Error copying local repository: {e}", exc_info=True)
            return None
    
    def _hash_requirements(self, requirements_file: Optional[Path]) -> Optional[str]:

        try:
            if not requirements_file or not requirements_file.exists():
                return None
            
            content = requirements_file.read_text()
            lines = [line.strip() for line in content.splitlines() 
                    if line.strip() and not line.strip().startswith('#')]
            normalized = '\n'.join(sorted(lines))
            
            hash_obj = hashlib.sha256(normalized.encode())
            return hash_obj.hexdigest()[:16]
            
        except Exception as e:
            self.logger.warning(f"Failed to hash requirements: {e}")
            return None

    def _resolve_requirements_file(
        self,
        repo_path: Path,
        requirements_spec: Optional[str]
    ) -> Optional[Path]:

        try:
            if requirements_spec:
                specified_path = Path(requirements_spec)
                if not specified_path.is_absolute():
                    specified_path = repo_path / specified_path
                if specified_path.exists():
                    self.logger.info(f"Using specified requirements file: {specified_path.name}")
                    return specified_path
            
            default_path = repo_path / "requirements.txt"
            if default_path.exists():
                self.logger.info("Found requirements.txt in repository root")
                return default_path
            
            ignore_dirs = {'.venv', 'venv', 'env', 'node_modules', '__pycache__'}
            matches: List[Path] = []
            for candidate in repo_path.rglob("requirements.txt"):
                if any(part in ignore_dirs for part in candidate.parts):
                    continue
                matches.append(candidate)
            
            if matches:
                matches.sort(key=lambda p: (len(p.relative_to(repo_path).parts), str(p)))
                chosen = matches[0]
                self.logger.info(f"Found nested requirements file: {chosen.relative_to(repo_path)}")
                return chosen
        
        except Exception as e:
            self.logger.warning(f"Failed to resolve requirements file: {e}")
        
        return None
    
    def _get_or_create_cached_venv(
        self,
        repo_path: Path,
        requirements_spec: Optional[str] = None
    ) -> Optional[Path]:
 
        try:
            requirements_file = self._resolve_requirements_file(repo_path, requirements_spec)
            
            if not requirements_file:
                self.logger.info("No requirements file found, creating empty venv in repo")
                venv_path = repo_path / ".venv"
                cmd = f"uv venv {venv_path}" if self._has_uv else f"python3 -m venv {venv_path}"
                result = self.executor.execute_command(cmd, working_directory=str(repo_path))
                return venv_path if result.success else None
            
            req_hash = self._hash_requirements(requirements_file)
            if not req_hash:
                self.logger.warning("Failed to hash requirements, creating non-cached venv")
                venv_path = repo_path / ".venv"
                cmd = f"uv venv {venv_path}" if self._has_uv else f"python3 -m venv {venv_path}"
                result = self.executor.execute_command(cmd, working_directory=str(repo_path))
                return venv_path if result.success else None
            
            cached_venv_path = self.VENV_CACHE_DIR / f"venv_{req_hash}"
            
            if cached_venv_path.exists():
                self.logger.info(f"♻️  Reusing cached venv: {req_hash}")
                venv_dir = repo_path / ".venv"
                if venv_dir.exists():
                    shutil.rmtree(venv_dir, ignore_errors=True)
                shutil.copytree(cached_venv_path, venv_dir, symlinks=True)
                return venv_dir
            
            self.logger.info(f"🆕 Creating new cached venv: {req_hash}")
            cached_venv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create venv
            cmd = f"uv venv {cached_venv_path}" if self._has_uv else f"python3 -m venv {cached_venv_path}"
            result = self.executor.execute_command(cmd, working_directory=str(repo_path))
            if not result.success:
                self.logger.error(f"Failed to create venv: {result.stderr}")
                return None
            
            # Install requirements
            self.logger.info(f"📦 Installing requirements into cached venv...")
            req_file = shlex.quote(str(requirements_file))
            
            if self._has_uv:
                python = shlex.quote(str(cached_venv_path / "bin" / "python"))
                install_cmd = f"uv pip install --python {python} -r {req_file}"
            else:
                pip = shlex.quote(str(cached_venv_path / "bin" / "pip"))
                install_cmd = f"{pip} install -r {req_file}"
            
            install_result = self.executor.execute_command(install_cmd, working_directory=str(repo_path))
            if not install_result.success:
                self.logger.warning(f"Failed to install requirements: {install_result.stderr}")
            
            venv_dir = repo_path / ".venv"
            if venv_dir.exists():
                shutil.rmtree(venv_dir, ignore_errors=True)
            shutil.copytree(cached_venv_path, venv_dir, symlinks=True)
            
            self.logger.info(f"✅ Cached venv ready: {cached_venv_path}")
            return venv_dir
            
        except Exception as e:
            self.logger.error(f"Error creating cached venv: {e}", exc_info=True)
            return None
    
    def execute(
        self,
        task: Dict[str, Any],
        interactive_handler: Optional["InteractiveInputHandler"] = None
    ) -> Dict[str, Any]:
        
        repo_path = task.get('repo_path')
        repo_url = task.get('repo_url')
        original_path = None
        
        # Determine where to get the repository from
        if repo_url:
            # Clone remote repository to temp/experiments
            self.logger.info(f"Cloning remote repository: {repo_url}")
            repo_path = self._clone_repository(repo_url, task.get('repo_branch'))
            if not repo_path:
                return {
                    'success': False,
                    'error': 'Failed to clone repository'
                }
        elif repo_path:
            # Check if repo is already in work directory
            repo_path_obj = Path(repo_path).resolve()
            if not self.WORK_DIR.resolve() in repo_path_obj.parents:
                # Only copy if not already in work directory
                original_path = str(repo_path)
                self.logger.info(f"Copying repository to work directory: {repo_path}")
                repo_path = self._copy_local_repo(repo_path)
                if not repo_path:
                    return {
                        'success': False,
                        'error': 'Failed to copy repository'
                    }
            else:
                self.logger.info(f"Using repository from work directory: {repo_path}")
                repo_path = repo_path_obj
        else:
            return {
                'success': False,
                'error': 'Either repo_path or repo_url is required'
            }

        repo_path = Path(repo_path).resolve()

        # Store interactive handler for use by tools during execution
        self._interactive_handler = interactive_handler
        if interactive_handler:
            if hasattr(self.run_command_tool, 'set_interactive_handler'):
                self.run_command_tool.set_interactive_handler(interactive_handler)
            if hasattr(self.request_credentials_tool, 'set_interactive_handler'):
                self.request_credentials_tool.set_interactive_handler(interactive_handler)

        # Prompt for missing environment variables/API keys before continuing
        # The LLM intelligently scans README and .env.example to detect requirements
        try:
            env_check = self.check_environment_requirements(repo_path)
            if env_check.get('has_missing'):
                self._prompt_for_missing_environment(
                    env_check,
                    interactive_handler=interactive_handler
                )
        except Exception as env_error:
            self.logger.error(
                f"Environment variable availability check failed: {env_error}",
                exc_info=True
            )
        
        # Restrict file creation tool to the working repository directory
        if hasattr(self, "create_tool") and self.create_tool:
            self.create_tool.set_base_directory(repo_path)
        
        requirements_spec = task.get('requirements_file')
        
        self.logger.info("🔧 Setting up isolated venv with smart caching...")
        venv_path = self._get_or_create_cached_venv(repo_path, requirements_spec)
        if venv_path:
            self.logger.info(f"✅ Venv ready at: {venv_path}")
        else:
            self.logger.warning("⚠️  Failed to create venv, will try without it")
        
        try:
            start_time = time.time()
            
            self.logger.info(f"Autonomous agent analyzing repository: {repo_path}")
            
            if venv_path and (Path(venv_path) / "bin" / "python").exists():
                python_cmd = str((Path(venv_path) / "bin" / "python").resolve())
                venv_info = f"Virtual environment available at {repo_path}/.venv (PATH configured)"
                install_note = "uv is available for package management" if self._has_uv else "Standard pip available"
            else:
                python_cmd = self._python_invocation  # Use detected invocation method
                venv_info = f"Using system Python (invoked via: {python_cmd})"
                install_note = "uv is available for package management" if self._has_uv else "Standard pip available"
            
            # Load task prompt from YAML
            task_prompt_template = BaseAgent._load_task_prompt('experiment_runner_agent')
            if not task_prompt_template:
                raise ValueError(
                    "Failed to load task prompt for experiment_runner_agent from config/agent_instructions.yaml. "
                    "Please ensure the configuration file exists and contains the 'task_prompt' field."
                )
            
            # Fill in template variables
            agent_task = BaseAgent._render_template(
                task_prompt_template,
                {
                    "repo_path": str(repo_path),
                    "venv_info": venv_info,
                    "python_cmd": python_cmd,
                    "install_note": install_note,
                }
            )
            
            self.logger.info("Agent is now exploring the repository and running experiments...")
            agent_response = self.call_llm(agent_task)
            
            duration_seconds = time.time() - start_time
            
            self.logger.info(f"Agent completed autonomous execution in {duration_seconds:.1f}s")
            
            execution_data = self._parse_execution_response(agent_response)

            if execution_data:
                command_ran = execution_data.get('recommended_command', '')
                exit_code = execution_data.get('exit_code')
                stdout_text = execution_data.get('stdout_snippet', '')
                stderr_text = execution_data.get('stderr_snippet', '')
                metrics = execution_data.get('metrics_extracted', {})
                artifacts = self._detect_artifacts(repo_path)
                
                success_flag = execution_data.get('succeeded', False)
                if exit_code == 0:
                    success_flag = True
                if metrics:
                    success_flag = True

                result = {
                    'success': success_flag,
                    'repo_path': str(repo_path),
                    'original_repo_path': str(original_path) if original_path else None,
                    'repo_url': repo_url,
                    'work_dir': str(self.WORK_DIR),
                    'command_ran': command_ran,
                    'exit_code': exit_code,
                    'stdout': stdout_text,
                    'stderr': stderr_text,
                    'metrics': metrics,
                    'artifacts': artifacts,
                    'duration_seconds': duration_seconds,
                    'agent_response': agent_response,
                    'execution_data': execution_data,
                    'execution_type': 'autonomous'
                }
            else:
                result = {
                    'success': False,
                    'repo_path': str(repo_path),
                    'original_repo_path': str(original_path) if original_path else None,
                    'repo_url': repo_url,
                    'work_dir': str(self.WORK_DIR),
                    'duration_seconds': duration_seconds,
                    'agent_response': agent_response,
                    'error': 'Could not parse structured execution data from agent response',
                    'execution_type': 'autonomous',
                    'note': 'Agent completed but structured parsing failed; inspect agent_response'
                }
            
            self.log_execution("autonomous_run", result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in autonomous execution: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'repo_path': str(repo_path) if repo_path else None
            }

    def _prompt_for_missing_environment(
        self,
        env_check: Dict[str, Any],
        interactive_handler: Optional["InteractiveInputHandler"] = None
    ) -> None:

        missing_vars = env_check.get('missing_vars', [])
        if not missing_vars:
            return

        self.logger.warning(f"Missing environment variables: {missing_vars}")
        print("\n⚠️  Missing environment variables:", ", ".join(missing_vars))
        print("The experiment may fail without these values.\n")

        provided: Dict[str, str] = {}
        if interactive_handler:
            if interactive_handler.confirm("Provide the missing keys now?"):
                required_vars = env_check.get('required_vars', [])
                var_lookup = {item['key']: item for item in required_vars}
                prompt_keys = [var_lookup[k] for k in missing_vars if k in var_lookup]
                provided = interactive_handler.prompt_for_api_keys(prompt_keys)
        else:
            try:
                import getpass
                for key in missing_vars:
                    value = getpass.getpass(f"{key} (press Enter to skip): ")
                    if value and len(value.strip()) >= 5:
                        provided[key] = value.strip()
            except Exception:
                pass

        if provided:
            for key, value in provided.items():
                os.environ[key] = value
            self.logger.info(f"Set {len(provided)} environment variable(s)")
            print(f"✅ Set {len(provided)} environment variable(s)\n")
    
    @classmethod
    def _parse_execution_response(cls, agent_response: Any) -> Optional[Dict[str, Any]]:
        """Parse the agent's final_answer() response.
        
        smolagents CodeAgent returns the result from final_answer() directly as a dict.
        """
        if not agent_response:
            return None
        
        # smolagents returns the dict directly from final_answer()
        if isinstance(agent_response, dict):
            return agent_response
        
        # If it's a string, try to parse it as JSON
        if isinstance(agent_response, str):
            try:
                return json.loads(agent_response)
            except json.JSONDecodeError:
                pass
            
            # If wrapped in markdown code block, extract it
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', agent_response)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Last resort: find JSON object in response
            brace_match = re.search(r'\{[\s\S]*\}', agent_response)
            if brace_match:
                try:
                    return json.loads(brace_match.group(0))
                except json.JSONDecodeError:
                    pass
        
        return None
    
    def _clone_repository(self, repo_url: str, branch: Optional[str] = None) -> Optional[Path]:
        """Clone a repository to the temp experiments directory."""
        try:
            # Use the class WORK_DIR constant
            self.WORK_DIR.mkdir(parents=True, exist_ok=True)
            
            # Create unique destination name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            dest_name = f"{repo_name}_{timestamp}"
            final_path = self.WORK_DIR / dest_name
            
            self.logger.info(f"Cloning repository: {repo_url}")
            self.logger.info(f"Target directory: {final_path}")
            self.logger.info(f"Branch: {branch or 'main'}")
            
            clone_result = self.executor.clone_repository(
                repo_url,
                str(final_path)
            )
            
            # Handle ExecutionResult object
            clone_success = clone_result.get('success') if isinstance(clone_result, dict) else clone_result.success
            
            if clone_success:
                self.logger.info(f"Repository cloned successfully to: {final_path}")
                return final_path
            else:
                error_msg = f"Failed to clone repository"
                if hasattr(clone_result, 'stderr') and clone_result.stderr:
                    error_msg += f": {clone_result.stderr}"
                if hasattr(clone_result, 'error_message') and clone_result.error_message:
                    error_msg += f" ({clone_result.error_message})"
                self.logger.error(error_msg)
                return None
        except Exception as e:
            self.logger.error(f"Error cloning repository: {e}", exc_info=True)
            return None


    def _detect_artifacts(self, output_dir: Path) -> Dict[str, Any]:
        
        figures: List[Dict[str, Any]] = []
        tables: List[Dict[str, Any]] = []
        
        try:
            if not output_dir.exists():
                return {'output_dir': str(output_dir), 'figures': figures, 'tables': tables}
            
            image_exts = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.eps'}
            table_exts = {'.csv', '.tsv', '.json', '.xlsx'}
            
            for item in output_dir.rglob('*'):
                if not item.is_file():
                    continue
                suffix = item.suffix.lower()
                if suffix in image_exts:
                    try:
                        rel_path = str(item.relative_to(output_dir))
                    except Exception:
                        rel_path = str(item)
                    figures.append({
                        'path': rel_path,
                        'extension': suffix,
                        'size_bytes': item.stat().st_size
                    })
                elif suffix in table_exts:
                    try:
                        rel_path = str(item.relative_to(output_dir))
                    except Exception:
                        rel_path = str(item)
                    tables.append({
                        'path': rel_path,
                        'extension': suffix,
                        'size_bytes': item.stat().st_size
                    })
        except Exception as e:
            self.logger.warning(f"Artifact detection error: {e}")
        
        return {
            'output_dir': str(output_dir),
            'figures': figures,
            'tables': tables
        }

def run_experiment(
    repo_path: Optional[str] = None,
    repo_url: Optional[str] = None
) -> Dict[str, Any]:
    
    agent = ExperimentRunnerAgent()
    result = agent.execute({
        'repo_path': repo_path,
        'repo_url': repo_url
    })
    
    return result
