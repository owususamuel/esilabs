"""
Smolagents Tool Wrappers - Wraps our tools for autonomous agent use.

These wrappers allow the LLM agent to autonomously decide which tools to use
and call them with appropriate parameters.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from smolagents import Tool


class ReadFileContents(Tool):
    name = "read_file_contents"
    description = """Reads the contents of a file from the repository.
    
    Args:
        file_path: Path to the file to read (relative to repository root)
        
    Returns:
        The contents of the file as a string
    """
    
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to read"
        }
    }
    output_type = "string"
    
    def forward(self, file_path: str) -> str:
        """Read and return file contents."""
        try:
            path = Path(file_path)
            if path.exists():
                return path.read_text(encoding='utf-8', errors='ignore')
            else:
                return f"Error: File not found: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class ListDirectoryFiles(Tool):
    name = "list_directory_files"
    description = """Lists files in a directory, useful for discovering entry points and structure.
    
    Args:
        directory_path: Path to the directory to list
        pattern: Optional glob pattern to filter files (e.g., "*.py", "*.txt")
        
    Returns:
        JSON string with list of files found
    """
    
    inputs = {
        "directory_path": {
            "type": "string",
            "description": "Path to the directory"
        },
        "pattern": {
            "type": "string",
            "description": "Optional glob pattern (e.g., '*.py')",
            "nullable": True
        }
    }
    output_type = "string"
    
    def forward(self, directory_path: str, pattern: Optional[str] = None) -> str:
        """List files in a directory."""
        try:
            path = Path(directory_path)
            if not path.exists():
                return json.dumps({"error": f"Directory not found: {directory_path}"})
            
            if pattern:
                files = [str(f.relative_to(path)) for f in path.rglob(pattern)]
            else:
                files = [str(f.relative_to(path)) for f in path.rglob("*") if f.is_file()]
            
            return json.dumps({"files": files[:50]})  # Limit to 50 files
        except Exception as e:
            return json.dumps({"error": str(e)})


class RunCommandInRepo(Tool):
    name = "run_command_in_repo"
    description = """Executes a command in the repository directory.
    Use this to run scripts with --help, install dependencies, or execute experiments.
    
    ⚠️  CRITICAL: For dependency installation, ALWAYS use 'uv pip install', NEVER plain 'pip install'
    ✅ CORRECT: uv pip install -r requirements.txt
    ❌ WRONG:   pip install -r requirements.txt
    
    Args:
        command: The command to run (e.g., "python script.py --help", "uv pip install -r requirements.txt")
        working_directory: Directory to run the command in
        timeout: Optional timeout in seconds (default: 60)
        
    Returns:
        JSON string with stdout, stderr, exit_code, and any error suggestions
    """
    
    inputs = {
        "command": {
            "type": "string",
            "description": "The command to execute"
        },
        "working_directory": {
            "type": "string",
            "description": "Directory to run command in"
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds",
            "nullable": True
        },
        "capture_output": {
            "type": "boolean",
            "description": "Set to false to disable capturing stdout/stderr (not recommended)",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self):
        super().__init__()
        self.interactive_handler = None
    
    def set_interactive_handler(self, handler):
        """Set the interactive handler for prompting user input."""
        self.interactive_handler = handler
    
    def forward(
        self,
        command: str,
        working_directory: str,
        timeout: Optional[int] = 60,
        capture_output: Optional[bool] = True
    ) -> str:
        """Execute a command and return results."""
        try:
            # Block unsafe attempts to inject or fabricate secrets
            secret_indicators = ("OPENAI", "ANTHROPIC", "AZURE", "HF_", "API_KEY", "TOKEN", "SECRET", "CREDENTIAL")
            lower_cmd = command.lower()
            # Heuristics: writing to .env, exporting sensitive vars, or echoing keys
            if (
                ".env" in command
                and any(ind.lower() in lower_cmd for ind in secret_indicators)
            ) or (
                ("export " in command or "set " in lower_cmd)
                and any(ind.lower() in lower_cmd for ind in secret_indicators)
            ):
                return json.dumps({
                    "success": False,
                    "stdout": "",
                    "stderr": (
                        "Blocked command that attempts to set or write secrets. "
                        "Secrets must be provided by the user via environment variables. "
                        "Please provide the required keys (e.g., OPENAI_API_KEY, AZURE_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN) "
                        "when prompted."
                    ),
                    "exit_code": -1,
                    "validation_error": True,
                    "error_type": "secrets_policy_violation"
                })
            
            # Detect repository virtual environment
            venv_dir = Path(working_directory) / ".venv"
            venv_bin_path = None
            venv_python = None
            if venv_dir.exists():
                for candidate in ("bin", "Scripts"):
                    bin_path = venv_dir / candidate
                    if bin_path.exists():
                        venv_bin_path = bin_path.resolve()
                        for python_name in ("python", "python3", "python.exe"):
                            python_path = bin_path / python_name
                            if python_path.exists():
                                venv_python = python_path.resolve()
                                break
                        if venv_python:
                            break
            
            # Validate pip commands - enforce 'uv pip install' usage
            if 'pip install' in command and not command.strip().startswith('uv pip'):
                warning_msg = (
                    "⚠️  WARNING: You are using 'pip install' without 'uv'. "
                    "This will likely fail. Use 'uv pip install' instead.\n"
                    f"Command attempted: {command}\n"
                    f"Recommended: {command.replace('pip install', 'uv pip install')}"
                )
                return json.dumps({
                    "success": False,
                    "stdout": "",
                    "stderr": warning_msg,
                    "exit_code": -1,
                    "validation_error": True
                })
            
            # If using 'uv pip install' within a repo that has a .venv,
            # ensure packages are installed into that exact interpreter.
            # This avoids installing into a different environment than the one used to run the script.
            if (
                command.strip().startswith('uv pip install')
                and '--python' not in command
                and venv_python
            ):
                parts = command.split()
                if len(parts) >= 3 and parts[0] == 'uv' and parts[1] == 'pip' and parts[2] == 'install':
                    prefix = parts[:3]
                    rest = parts[3:]
                    command = ' '.join(prefix + ['--python', str(venv_python)] + rest)
            
            env = os.environ.copy()
            if venv_bin_path:
                env_path = env.get('PATH', '')
                env['PATH'] = f"{venv_bin_path}{os.pathsep}{env_path}" if env_path else str(venv_bin_path)
                env['VIRTUAL_ENV'] = str(venv_dir.resolve())
            
            if capture_output is False:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=working_directory,
                    env=env,
                    stdout=None,
                    stderr=None,
                    text=True
                )
                try:
                    process.wait(timeout=timeout)
                    return json.dumps({
                        "success": process.returncode == 0,
                        "stdout": "",
                        "stderr": "",
                        "exit_code": process.returncode,
                        "note": "Output streaming not captured by this tool; consider capture_output=true for logs."
                    })
                except subprocess.TimeoutExpired:
                    process.kill()
                    return json.dumps({
                        "success": False,
                        "error": f"Command timed out after {timeout} seconds",
                        "error_type": "timeout",
                        "suggestion": "Command may be waiting for input or is blocking. Try with non-interactive flags or different arguments."
                    })
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=working_directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Enhanced error detection for pip/uv issues
            # Capture larger output for metrics extraction (experiments often print results at the end)
            response = {
                "success": result.returncode == 0,
                "stdout": result.stdout[:50000],  # Increased from 2000 to 50000 for metrics
                "stderr": result.stderr[:50000],  # Increased from 2000 to 50000 for metrics
                "exit_code": result.returncode
            }
            
            # Detect common package manager errors
            stderr_lower = result.stderr.lower()
            if 'pip: command not found' in stderr_lower or 'pip: not found' in stderr_lower:
                response["error_type"] = "pip_not_found"
                response["suggestion"] = "Use 'uv pip install' instead of 'pip install'"
            elif 'uv: command not found' in stderr_lower or 'uv: not found' in stderr_lower:
                response["error_type"] = "uv_not_found"
                response["suggestion"] = "uv package manager is not available in this environment"
            
            # Detect missing API keys / environment variables in stderr
            missing_keys = self._detect_missing_api_keys(result.stderr)
            if missing_keys and self.interactive_handler:
                response["error_type"] = "missing_api_keys"
                response["missing_keys"] = missing_keys
                response["suggestion"] = f"Missing API keys detected: {', '.join(missing_keys)}. Prompting user for input..."
                
                # Prompt user for missing keys
                prompted_keys = self._prompt_for_keys(missing_keys)
                if prompted_keys:
                    response["keys_provided"] = list(prompted_keys.keys())
                    response["suggestion"] += f" User provided {len(prompted_keys)} key(s). Retry the command."
                else:
                    response["suggestion"] += " User skipped providing keys. Command will likely fail again."
            
            return json.dumps(response)
            
        except subprocess.TimeoutExpired:
            return json.dumps({
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "error_type": "timeout",
                "suggestion": "Command may be waiting for input or is blocking. Try with non-interactive flags or different arguments."
            })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
                "error_type": "execution_error"
            })
    
    def _detect_missing_api_keys(self, stderr: str) -> list:
        """
        Detect missing API keys from error messages.
        
        Args:
            stderr: Standard error output
            
        Returns:
            List of missing key names
        """
        import re
        
        missing_keys = []
        
        # Common patterns for missing API keys
        patterns = [
            r"KeyError:\s*['\"]([A-Z_][A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|CREDENTIALS)[A-Z0-9_]*)['\"]",
            r"Missing\s+(?:environment\s+)?variable:\s*([A-Z_][A-Z0-9_]*)",
            r"([A-Z_][A-Z0-9_]*(?:API_KEY|TOKEN|SECRET)[A-Z0-9_]*)\s+(?:is\s+)?not\s+(?:set|found|defined)",
            r"Environment\s+variable\s+['\"]?([A-Z_][A-Z0-9_]*)['\"]?\s+(?:is\s+)?not\s+set",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, stderr, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                key = match.upper() if isinstance(match, str) else match[0].upper()
                if key and key not in missing_keys:
                    missing_keys.append(key)
        
        return missing_keys
    
    def _prompt_for_keys(self, missing_keys: list) -> dict:
        """
        Prompt user for missing API keys.
        
        Args:
            missing_keys: List of missing key names
            
        Returns:
            Dictionary of key -> value pairs
        """
        if not self.interactive_handler:
            return {}
        
        import os
        
        # Format keys for prompt
        key_info = [{'key': key, 'description': f'Required by experiment'} for key in missing_keys]
        
        # Prompt user
        provided = self.interactive_handler.prompt_for_api_keys(key_info)
        
        # Set in environment
        for key, value in provided.items():
            os.environ[key] = value
        
        return provided


class CreateFileOrDirectory(Tool):
    name = "create_file_or_directory"
    description = """Creates a file or directory in the repository.
    Use this when the code needs input files or output directories.
    
    Args:
        path: Path to create (if ends with /, creates directory, otherwise creates empty file)
        content: Optional content to write to the file
        
    Returns:
        Success message or error
    """
    
    inputs = {
        "path": {
            "type": "string",
            "description": "Path to create"
        },
        "content": {
            "type": "string",
            "description": "Optional content for files",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, base_directory: Optional[Path] = None):
        super().__init__()
        self.base_directory = Path(base_directory).resolve() if base_directory else None
    
    def set_base_directory(self, base_directory: Optional[Path]) -> None:
        """Limit file creation to within the provided base directory."""
        self.base_directory = Path(base_directory).resolve() if base_directory else None
    
    def forward(self, path: str, content: Optional[str] = None) -> str:
        """Create a file or directory."""
        try:
            is_directory = path.endswith('/')
            raw_path = path[:-1] if is_directory else path
            p = Path(raw_path)
            
            if self.base_directory:
                base_dir = self.base_directory
                if not p.is_absolute():
                    p = (base_dir / p).resolve()
                else:
                    p = p.resolve()
                
                if base_dir != p and base_dir not in p.parents:
                    return f"Error: Path '{p}' is outside allowed workspace '{base_dir}'"
            else:
                p = p.resolve()
            
            if is_directory:
                # Directory
                p.mkdir(parents=True, exist_ok=True)
                return f"Created directory: {p}"
            else:
                # File
                p.parent.mkdir(parents=True, exist_ok=True)
                if content:
                    p.write_text(content)
                else:
                    p.touch()
                return f"Created file: {p}"
        except Exception as e:
            return f"Error: {str(e)}"


class SearchGitHub(Tool):
    name = "search_github"
    description = """Searches GitHub for repositories matching a query.
    Use this to find repositories related to a research paper.
    
    Args:
        query: Search query (paper title, author name, keywords)
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        JSON string with repository information (name, url, stars, description)
    """
    
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query"
        },
        "max_results": {
            "type": "integer",
            "description": "Max results",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, github_tool):
        """Initialize with actual GitHub search tool."""
        super().__init__()
        self.github_tool = github_tool
    
    def forward(self, query: str, max_results: Optional[int] = 5) -> str:
        """Search GitHub."""
        try:
            # Use find_repository method which returns Repository objects
            results = self.github_tool.find_repository(
                paper_title=query,
                authors=None,
                max_results=max_results
            )
            
            repos = []
            for repo in results[:max_results]:
                repos.append({
                    "name": repo.name,
                    "url": repo.url,
                    "stars": repo.stars,
                    "description": (repo.description or "")[:200],
                    "language": repo.language or "",
                    "last_updated": repo.last_updated or ""
                })
            
            return json.dumps({"repositories": repos})
        except Exception as e:
            return json.dumps({"error": str(e)})


class ExtractMetrics(Tool):
    name = "extract_metrics"
    description = """Intelligently extracts numerical metrics from experiment output using LLM.
    Can handle any format: JSON files, text logs, tables, mixed formats.
    
    Use this to extract results from experiment output files or stdout.
    
    Args:
        text: Text containing metrics (from files, logs, or command output)
        
    Returns:
        JSON string with extracted metrics like {"metric_name": value}
    """
    
    inputs = {
        "text": {
            "type": "string",
            "description": "Text containing metrics (any format)"
        }
    }
    output_type = "string"
    
    def __init__(self, comparator_tool, llm_client=None):
        """
        Initialize with result comparator and optional LLM client.
        
        Args:
            comparator_tool: ResultComparator instance
            llm_client: Optional LLM client for intelligent extraction
        """
        super().__init__()
        self.comparator = comparator_tool
        self.llm_client = llm_client
    
    def forward(self, text: str) -> str:
        """Extract metrics from text using LLM-powered extraction."""
        try:
            # Use LLM-powered extraction if available
            metrics = self.comparator.extract_metrics(text, llm_client=self.llm_client)
            return json.dumps({"metrics": metrics})
        except Exception as e:
            return json.dumps({"error": str(e), "metrics": {}})


class ParsePDF(Tool):
    name = "parse_pdf"
    description = """Extracts comprehensive text content from a research paper PDF for LLM analysis.
    Returns structured text including title, abstract, sections, and full document text.
    Includes both the beginning (for authors, title) and end (for references, GitHub links) of the document.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Structured text with sections labeled for easy parsing by LLM
    """
    
    inputs = {
        "pdf_path": {
            "type": "string",
            "description": "Path to PDF file"
        }
    }
    output_type = "string"
    
    def __init__(self, pdf_parser):
        """Initialize with PDF parser."""
        super().__init__()
        self.pdf_parser = pdf_parser
    
    def forward(self, pdf_path: str) -> str:
        """Parse PDF and return comprehensive structured text for LLM analysis."""
        try:
            result = self.pdf_parser.parse_pdf(pdf_path)
            
            # Return the full text with structured sections
            # Let the LLM intelligently analyze the entire document
            text_parts = []
            
            # Add metadata if available
            if result.metadata:
                text_parts.append("=== PDF METADATA ===")
                for key, value in result.metadata.items():
                    if value:
                        text_parts.append(f"{key}: {value}")
            
            # Add basic extracted info
            if result.title:
                text_parts.append(f"\n=== EXTRACTED TITLE ===\n{result.title}")
            if result.abstract:
                text_parts.append(f"\n=== EXTRACTED ABSTRACT ===\n{result.abstract}")
            
            # Add structured sections (these are already intelligently extracted)
            if result.sections:
                text_parts.append("\n=== DOCUMENT SECTIONS ===")
                for section_name, section_content in result.sections.items():
                    text_parts.append(f"\n--- {section_name.upper()} ---")
                    text_parts.append(section_content)
            
            # Add full document text for comprehensive LLM analysis
            # The LLM can intelligently identify authors, GitHub URLs, datasets, etc.
            # anywhere in the document without arbitrary character limits
            if result.full_text:
                text_parts.append(f"\n\n=== FULL DOCUMENT TEXT ===")
                text_parts.append(result.full_text)
            
            return "\n".join(text_parts) if text_parts else "Could not extract text from PDF"
        except Exception as e:
            return f"Error parsing PDF: {str(e)}"

