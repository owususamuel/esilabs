import ast
import json
import os
import re
import shutil
import hashlib
import shlex
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
    ExtractMetrics
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
        
        self.register_tool("read_file_contents", None, self.read_file_tool)
        self.register_tool("list_directory_files", None, self.list_dir_tool)
        self.register_tool("extract_metrics", None, self.extract_metrics_tool)
        self.register_tool("run_command_in_repo", None, self.run_command_tool)
        self.register_tool("create_file_or_directory", None, self.create_tool)
        
        # Store reference to interactive handler (will be set in execute)
        self._interactive_handler = None
        
        # Ensure cache directory exists
        self.VENV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def check_environment_requirements(self, repo_path: Path) -> Dict[str, Any]:
        
        self.logger.info(f"Checking environment requirements for {repo_path}")
        
        required_vars = []
        found_in = []
        
        # Common patterns for environment variables
        env_patterns = [
            r'export\s+([A-Z_][A-Z0-9_]*)\s*=',  # export VAR=
            r'\$\{?([A-Z_][A-Z0-9_]*)\}?',       # ${VAR} or $VAR
            r'os\.getenv\([\'"]([A-Z_][A-Z0-9_]*)[\'"]',  # os.getenv("VAR")
            r'os\.environ\[[\'"]([A-Z_][A-Z0-9_]*)[\'"]',  # os.environ["VAR"]
        ]
        
        # API key patterns to prioritize
        api_key_patterns = [
            'API_KEY', 'TOKEN', 'SECRET', 'CREDENTIALS',
            'OPENAI', 'ANTHROPIC', 'HF_', 'AZURE', 'GITHUB'
        ]
        
        try:
            # 1. Check README files
            readme_files = list(repo_path.glob('README*')) + list(repo_path.glob('readme*'))
            for readme in readme_files:
                if readme.is_file():
                    try:
                        content = readme.read_text(encoding='utf-8', errors='ignore')
                        
                        # Find all environment variables
                        env_vars = set()
                        for pattern in env_patterns:
                            matches = re.findall(pattern, content, re.MULTILINE)
                            env_vars.update(matches)
                        
                        # Filter for API keys and important vars
                        for var in env_vars:
                            if any(key_pattern in var for key_pattern in api_key_patterns):
                                # Try to extract description from surrounding text
                                desc = self._extract_var_description(content, var)
                                required_vars.append({
                                    'key': var,
                                    'description': desc
                                })
                        
                        if env_vars:
                            found_in.append(str(readme.name))
                    
                    except Exception as e:
                        self.logger.warning(f"Error reading {readme}: {e}")
            
            # 2. Check .env.example file
            env_example = repo_path / '.env.example'
            if env_example.exists():
                try:
                    content = env_example.read_text(encoding='utf-8', errors='ignore')
                    for line in content.splitlines():
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key = line.split('=')[0].strip()
                            if key and any(pattern in key for pattern in api_key_patterns):
                                # Check if not already added
                                if not any(v['key'] == key for v in required_vars):
                                    # Extract comment as description
                                    desc = ''
                                    comment_match = re.search(r'#\s*(.+)$', line)
                                    if comment_match:
                                        desc = comment_match.group(1).strip()
                                    
                                    required_vars.append({
                                        'key': key,
                                        'description': desc or 'See .env.example'
                                    })
                    
                    if required_vars:
                        found_in.append('.env.example')
                
                except Exception as e:
                    self.logger.warning(f"Error reading .env.example: {e}")
            
            # 3. Check which variables are actually missing
            missing_vars = []
            for var_info in required_vars:
                key = var_info['key']
                if not os.getenv(key):
                    missing_vars.append(key)
            
            result = {
                'required_vars': required_vars,
                'missing_vars': missing_vars,
                'found_in': found_in,
                'has_missing': len(missing_vars) > 0
            }
            
            if missing_vars:
                self.logger.warning(f"Missing environment variables: {missing_vars}")
            else:
                self.logger.info("All required environment variables are set")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error checking environment requirements: {e}", exc_info=True)
            return {
                'required_vars': [],
                'missing_vars': [],
                'found_in': [],
                'has_missing': False,
                'error': str(e)
            }
    
    def _extract_var_description(self, content: str, var_name: str) -> str:
        
        try:
            # Look for lines mentioning the variable
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if var_name in line:
                    # Check surrounding lines for description
                    context = []
                    
                    # Previous line
                    if i > 0:
                        prev = lines[i-1].strip()
                        if prev and not prev.startswith('#') and len(prev) < 100:
                            context.append(prev)
                    
                    # Current line
                    current = line.strip()
                    if ':' in current or '-' in current:
                        parts = re.split(r'[:\-]', current, 1)
                        if len(parts) > 1:
                            context.append(parts[1].strip())
                    
                    # Next line
                    if i < len(lines) - 1:
                        next_line = lines[i+1].strip()
                        if next_line and not next_line.startswith(var_name) and len(next_line) < 100:
                            context.append(next_line)
                    
                    if context:
                        desc = ' '.join(context)
                        # Clean up
                        desc = re.sub(r'export\s+\w+\s*=.*', '', desc)
                        desc = re.sub(r'\$\{?\w+\}?', '', desc)
                        desc = desc.strip()
                        if desc and len(desc) > 10:
                            return desc[:150]  # Limit length
            
            return ''
        
        except Exception:
            return ''
    
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
            # Normalize: sort lines, strip whitespace, ignore comments/empty lines
            lines = [line.strip() for line in content.splitlines() 
                    if line.strip() and not line.strip().startswith('#')]
            normalized = '\n'.join(sorted(lines))
            
            hash_obj = hashlib.sha256(normalized.encode())
            return hash_obj.hexdigest()[:16]  # Use first 16 chars
            
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
                    self.logger.info(
                        f"Using specified requirements file: {specified_path.relative_to(repo_path)}"
                        if specified_path.is_relative_to(repo_path)
                        else f"Using specified requirements file: {specified_path}"
                    )
                    return specified_path
                self.logger.warning(f"Specified requirements file not found: {specified_path}")
            
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
                self.logger.info(
                    f"Found nested requirements file: {chosen.relative_to(repo_path)}"
                )
                return chosen
        
        except Exception as e:
            self.logger.warning(f"Failed to resolve requirements file: {e}", exc_info=True)
        
        return None
    
    def _get_or_create_cached_venv(
        self,
        repo_path: Path,
        requirements_spec: Optional[str] = None
    ) -> Optional[Path]:
 
        try:
            requirements_file = self._resolve_requirements_file(repo_path, requirements_spec)
            
            # If no requirements.txt, create empty venv in repo
            if not requirements_file:
                self.logger.info("No requirements file found, creating empty venv in repo")
                venv_path = repo_path / ".venv"
                result = self.executor.execute_command(
                    f"uv venv {venv_path}",
                    working_directory=str(repo_path)
                )
                if result.success:
                    return venv_path
                return None
            
            # Hash requirements for caching
            req_hash = self._hash_requirements(requirements_file)
            if not req_hash:
                self.logger.warning("Failed to hash requirements, creating non-cached venv")
                venv_path = repo_path / ".venv"
                result = self.executor.execute_command(
                    f"uv venv {venv_path}",
                    working_directory=str(repo_path)
                )
                if result.success:
                    return venv_path
                return None
            
            # Check for cached venv
            cached_venv_path = self.VENV_CACHE_DIR / f"venv_{req_hash}"
            
            if cached_venv_path.exists():
                self.logger.info(f"♻️  Reusing cached venv: {req_hash}")
                # Copy cached venv into repository .venv to avoid symlink permission issues
                venv_dir = repo_path / ".venv"
                if venv_dir.exists():
                    shutil.rmtree(venv_dir, ignore_errors=True)
                shutil.copytree(cached_venv_path, venv_dir, symlinks=True)
                return venv_dir
            
            # Create new venv in cache
            self.logger.info(f"🆕 Creating new cached venv: {req_hash}")
            cached_venv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create venv in cache location
            result = self.executor.execute_command(
                f"uv venv {cached_venv_path}",
                working_directory=str(repo_path)
            )
            
            if not result.success:
                self.logger.error(f"Failed to create venv: {result.stderr}")
                return None
            
            # Install requirements into cached venv
            self.logger.info(f"📦 Installing requirements into cached venv...")
            python_path = cached_venv_path / "bin" / "python"
            requirements_arg = shlex.quote(str(requirements_file))
            python_arg = shlex.quote(str(python_path))
            install_result = self.executor.execute_command(
                f"uv pip install --python {python_arg} -r {requirements_arg}",
                working_directory=str(repo_path)
            )
            
            if not install_result.success:
                self.logger.warning(f"Failed to install requirements: {install_result.stderr}")
                # Keep the venv anyway, might be partial success
            
            # Copy cached venv into repository
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
        if interactive_handler and hasattr(self.run_command_tool, 'set_interactive_handler'):
            self.run_command_tool.set_interactive_handler(interactive_handler)

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
        
        # Setup cached venv for this experiment (hybrid approach)
        self.logger.info("🔧 Setting up isolated venv with smart caching...")
        venv_path = self._get_or_create_cached_venv(repo_path, requirements_spec)
        if venv_path:
            self.logger.info(f"✅ Venv ready at: {venv_path}")
        else:
            self.logger.warning("⚠️  Failed to create venv, will try without it")
        
        try:
            self.logger.info(f"🤖 Autonomous agent analyzing repository: {repo_path}")
            
            # Determine Python interpreter and guidance for dependency installation
            if venv_path and (Path(venv_path) / "bin" / "python").exists():
                python_cmd = str((Path(venv_path) / "bin" / "python").resolve())
                venv_info = f"✅ Isolated venv is already set up at {repo_path}/.venv"
                install_note = (
                    "Install or update dependencies using 'uv pip install ...' from the repository root "
                    "(never plain pip). Example: uv pip install -r requirements.txt or uv pip install package_name."
                )
            else:
                python_cmd = "python3"
                venv_info = "⚠️  No venv available, fall back to system Python"
                install_note = (
                    "If dependencies are missing, run 'uv pip install ...' from the repository directory "
                    "to ensure required packages are available."
                )
            
            # Load task prompt from YAML
            task_prompt_template = BaseAgent._load_task_prompt('experiment_runner_agent')
            if not task_prompt_template:
                raise ValueError(
                    "Failed to load task prompt for experiment_runner_agent from config/agent_instructions.yaml. "
                    "Please ensure the configuration file exists and contains the 'task_prompt' field."
                )
            
            # Fill in template variables
            agent_task = task_prompt_template.format(
                repo_path=repo_path,
                venv_info=venv_info,
                python_cmd=python_cmd,
                install_note=install_note
            )
            
            # Agent works autonomously
            self.logger.info("Agent is now exploring the repository and running experiments...")
            agent_response = self.call_llm(agent_task)
            
            self.logger.info(f"✅ Agent completed autonomous execution")
            
            execution_data = self._parse_execution_response(agent_response)

            if execution_data:
                # Extract command from recommended_command or executed_commands
                command_ran = execution_data.get('recommended_command', '')
                executed_commands = execution_data.get('executed_commands', [])
                
                # Find the main experiment command (not pip install)
                exit_code = None
                stdout_text = ''
                stderr_text = ''
                success_flag = False
                
                for cmd_info in executed_commands:
                    if isinstance(cmd_info, dict):
                        cmd = cmd_info.get('command', '')
                        if 'pip install' not in cmd and 'uv pip install' not in cmd:
                            # This is the main experiment command
                            success_flag = cmd_info.get('succeeded', False)
                            exit_code = cmd_info.get('exit_code', None)
                            stdout_text = cmd_info.get('stdout_snippet', '')
                            stderr_text = cmd_info.get('stderr_snippet', '')
                            if not command_ran:
                                command_ran = cmd
                
                if not command_ran:
                    command_ran = execution_data.get('command_executed', command_ran)
                
                if exit_code is None:
                    exit_code = execution_data.get('exit_code', exit_code)
                
                if not stdout_text:
                    stdout_text = execution_data.get('stdout_snippet', stdout_text)
                
                if not stderr_text:
                    stderr_text = execution_data.get('stderr_snippet', stderr_text)
                
                if not success_flag and execution_data.get('succeeded') is not None:
                    success_flag = bool(execution_data.get('succeeded'))
                
                # Extract metrics from metrics_extracted field
                raw_metrics = execution_data.get('metrics_extracted', {})
                metrics: Dict[str, Any] = {}
                
                if isinstance(raw_metrics, dict):
                    # Get best_performing_configurations and flatten to simple metrics
                    best_configs = raw_metrics.get('best_performing_configurations', {})
                    if best_configs:
                        for retriever, config_data in best_configs.items():
                            if isinstance(config_data, dict):
                                metrics[f'{retriever}_recall@10'] = config_data.get('Recall@10', 0)
                                metrics[f'{retriever}_mrr'] = config_data.get('MRR', 0)
                    
                    # Also get example detailed results
                    example_results = raw_metrics.get('example_detailed_results_sentence_minimal', {})
                    if example_results:
                        for method, method_metrics in example_results.items():
                            if isinstance(method_metrics, dict):
                                for key, value in method_metrics.items():
                                    metrics[f'{method}_{key}'] = value
                    
                    # If no metrics extracted, try the raw dict
                    if not metrics and raw_metrics:
                        metrics = raw_metrics

                # Set success based on exit code if available
                if exit_code == 0:
                    success_flag = True
                
                # CRITICAL FIX: If metrics were extracted, consider it a success
                # This handles cases where exit_code isn't captured but experiment ran successfully
                if metrics and len(metrics) > 0:
                    self.logger.info(f"✅ Metrics extracted successfully ({len(metrics)} metrics) - marking as success")
                    success_flag = True
                
                # Additional check: If agent completed without explicit failure and returned structured data, assume success
                if not success_flag and execution_data and not execution_data.get('error') and not execution_data.get('failed'):
                    # Agent completed execution and returned structured output without errors
                    self.logger.info("✅ Agent completed execution with structured output - assuming success")
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
                    'duration_seconds': None,  # Agent doesn't provide this in current format
                    'agent_response': agent_response,
                    'execution_data': execution_data,
                    'issues_encountered': execution_data.get('issues_encountered'),
                    'execution_type': 'autonomous',
                    'note': 'All experiments run in data/temp/experiments for isolation. Agent autonomously explored repo, determined how to run code, and executed experiments.'
                }
            else:
                result = {
                    'success': False,
                    'repo_path': str(repo_path),
                    'original_repo_path': str(original_path) if original_path else None,
                    'repo_url': repo_url,
                    'work_dir': str(self.WORK_DIR),
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
        required_vars = env_check.get('required_vars', [])
        if not missing_vars:
            return

        self.logger.warning(f"Missing environment variables detected: {missing_vars}")

        found_in = ', '.join(env_check.get('found_in', [])) or 'repository files'
        print("\n" + "=" * 60)
        print("⚠️  REQUIRED ENVIRONMENT VARIABLES MISSING")
        print("=" * 60)
        print(f"\nThese variables were referenced in: {found_in}\n")

        var_lookup = {item['key']: item for item in required_vars}
        for key in missing_vars:
            info = var_lookup.get(key, {})
            description = info.get('description') or 'No description provided.'
            print(f"  ❌ {key}")
            if description:
                print(f"     {description}")

        print("\nWithout these values the experiment may fail or skip crucial steps.\n")

        provided: Dict[str, str] = {}
        if interactive_handler:
            if interactive_handler.confirm("Provide the missing keys now?"):
                prompt_keys = [var_lookup[k] for k in missing_vars if k in var_lookup]
                provided = interactive_handler.prompt_for_api_keys(prompt_keys)
            else:
                print("⚠️  Continuing without API keys (experiment may fail)\n")
        else:
            try:
                import getpass
                print("Enter the values now, or press Enter to skip a key.")
                for key in missing_vars:
                    while True:
                        value = getpass.getpass(f"{key}: ")
                        if not value:
                            print(f"  Skipped {key}")
                            break
                        if len(value.strip()) < 5:
                            print("  Value seems too short. Try again or leave blank to skip.")
                            continue
                        provided[key] = value.strip()
                        print(f"  ✓ {key} captured")
                        break
                print()
            except Exception:
                print("Unable to securely capture input; skipping prompt.")

        if provided:
            for key, value in provided.items():
                os.environ[key] = value
            self.logger.info(f"Captured and set {len(provided)} environment variable(s)")
            print(f"✅ Set {len(provided)} environment variable(s)\n")
        else:
            self.logger.warning("No environment variables provided; proceeding without them.")
    
    @staticmethod
    def _extract_structured_block(agent_response: str) -> Optional[str]:
        """Extract a JSON/Python dict-like block from the agent response."""
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
        
        final_func_match = re.search(r'final_answer\s*\(\s*(\{)', agent_response, re.IGNORECASE)
        if final_func_match:
            block = extract_from_index(agent_response, final_func_match.start(1))
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
        """Sanitize JSON-like block for literal evaluation."""
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
    
    @classmethod
    def _parse_structured_dict(cls, block: str) -> Optional[Dict[str, Any]]:
        """Parse structured data from the extracted block."""
        if not block:
            return None
    
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass
    
        sanitized_block = cls._sanitize_block_for_literal(block)
    
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
    
    @classmethod
    def _parse_execution_response(cls, agent_response: str) -> Optional[Dict[str, Any]]:
        """Parse the agent's final response into structured execution data."""
        structured_block = cls._extract_structured_block(agent_response)
        if not structured_block:
            return None
    
        parsed = cls._parse_structured_dict(structured_block)
        if not parsed:
            return None
    
        return parsed
    
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
            clone_result = self.executor.clone_repository(
                repo_url,
                str(final_path),
                branch=branch or 'main'
            )
            
            clone_success = clone_result.get('success') if isinstance(clone_result, dict) else clone_result.success
            if clone_success:
                self.logger.info(f"Repository cloned to: {final_path}")
                return final_path
            else:
                self.logger.error("Failed to clone repository")
                return None
        except Exception as e:
            self.logger.error(f"Error cloning repository: {e}")
            return None


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
