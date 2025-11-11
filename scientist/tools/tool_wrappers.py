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
    description = """Lists files and directories at the specified path.
    
    Args:
        directory_path: Path to the directory to list
        pattern: Optional glob pattern to filter files (e.g., "*.py", "*.txt")
        
    Returns:
        JSON string with list of files found
    
    Note: This tool excludes common directories like .git, node_modules, .venv, __pycache__, etc.
    It will stop scanning if more than 10,000 files are found to prevent excessive resource usage.
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
    
    # Common directories to exclude from scanning
    EXCLUDED_DIRS = {
        '.git', '.venv', 'venv', 'node_modules', '__pycache__', 
        '.pytest_cache', '.mypy_cache', '.tox', '.eggs', 
        'dist', 'build', '.next', '.nuxt', 'target',
        '.idea', '.vscode', '.DS_Store', 'venv_cache'
    }
    
    MAX_FILES_TO_SCAN = 10000  # Safety limit
    
    def _should_skip_directory(self, dir_path: Path) -> bool:
        """Check if a directory should be skipped."""
        return any(excluded in dir_path.parts for excluded in self.EXCLUDED_DIRS)
    
    def forward(self, directory_path: str, pattern: Optional[str] = None) -> str:
        """List files in a directory with safety limits and exclusions."""
        try:
            path = Path(directory_path).resolve()
            if not path.exists():
                return json.dumps({"error": f"Directory not found: {directory_path}"})
            
            if not path.is_dir():
                return json.dumps({"error": f"Not a directory: {directory_path}"})
            
            files = []
            directories = []
            files_scanned = 0
            
            # Walk the directory tree manually to have better control
            for item in path.rglob(pattern if pattern else "*"):
                # Safety check: stop if we've scanned too many files
                files_scanned += 1
                if files_scanned > self.MAX_FILES_TO_SCAN:
                    return json.dumps({
                        "error": f"Directory scan limit exceeded ({self.MAX_FILES_TO_SCAN} files). "
                                f"Please specify a more specific directory or pattern.",
                        "files": files[:50],
                        "directories": directories[:50],
                        "total_files": len(files),
                        "total_directories": len(directories),
                        "scan_incomplete": True
                    })
                
                # Skip excluded directories
                if self._should_skip_directory(item):
                    continue
                
                try:
                    rel_path = str(item.relative_to(path))
                    if item.is_dir():
                        directories.append(rel_path)
                    elif item.is_file():
                        files.append(rel_path)
                except (ValueError, OSError):
                    # Skip items that can't be processed
                    continue
            
            unique_dirs = list(dict.fromkeys(directories))
            unique_files = list(dict.fromkeys(files))
            
            return json.dumps({
                "files": unique_files[:100],  # Show more files in the preview
                "directories": unique_dirs[:100],
                "total_files": len(unique_files),
                "total_directories": len(unique_dirs)
            })
        except Exception as e:
            return json.dumps({"error": f"Error listing directory: {str(e)}"})


class RunCommandInRepo(Tool):
    name = "run_command_in_repo"
    description = """Executes a command in the repository directory.
    Use this to run scripts, install dependencies, or execute experiments.
    
    Args:
        command: The command to run (e.g., "python script.py --help", "uv pip install torch")
        working_directory: Directory to run the command in
        timeout: Optional timeout in seconds (default: 60)
        capture_output: Whether to capture stdout/stderr (default: True)
        
    Returns:
        JSON string with stdout, stderr, exit_code, and success flag
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
            # Detect repository virtual environment and set up PATH
            venv_dir = Path(working_directory) / ".venv"
            venv_bin_path = None
            
            if venv_dir.exists():
                for candidate in ("bin", "Scripts"):
                    bin_path = venv_dir / candidate
                    if bin_path.exists():
                        venv_bin_path = bin_path.resolve()
                        break
            
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
                        "error": f"Command timed out after {timeout} seconds"
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
            
            # Return results - let the agent interpret errors and decide what to do
            return json.dumps({
                "success": result.returncode == 0,
                "stdout": result.stdout[:50000],
                "stderr": result.stderr[:50000],
                "exit_code": result.returncode
            })
            
        except subprocess.TimeoutExpired:
            return json.dumps({
                "success": False,
                "error": f"Command timed out after {timeout} seconds"
            })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class CreateFileOrDirectory(Tool):
    name = "create_file_or_directory"
    description = """Creates a file or directory in the repository.
    
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


class ExtractMetrics(Tool):
    name = "extract_metrics"
    description = """Extracts numerical metrics from text using LLM.
    Can handle multiple formats: JSON, logs, tables, etc.
    
    Args:
        text: Text containing metrics
        
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


class RequestCredentials(Tool):
    name = "request_credentials"
    description = """Prompts the user for API keys or credentials.
    
    Args:
        keys: Comma-separated list of required API key names
        reason: Brief explanation of why these keys are needed
        
    Returns:
        JSON string indicating which keys were provided by the user
    """
    
    inputs = {
        "keys": {
            "type": "string",
            "description": "Comma-separated list of required API key names"
        },
        "reason": {
            "type": "string",
            "description": "Brief explanation of why these keys are needed"
        }
    }
    output_type = "string"
    
    def __init__(self):
        super().__init__()
        self.interactive_handler = None
    
    def set_interactive_handler(self, handler):
        """Set the interactive handler for prompting user input."""
        self.interactive_handler = handler
    
    def forward(self, keys: str, reason: str) -> str:
        """Request credentials from the user."""
        try:
            import os
            
            # Parse key names
            key_names = [k.strip() for k in keys.split(',') if k.strip()]
            
            if not key_names:
                return json.dumps({
                    "success": False,
                    "error": "No key names provided"
                })
            
            # Check which keys are already set
            already_set = [k for k in key_names if os.getenv(k)]
            missing = [k for k in key_names if not os.getenv(k)]
            
            if not missing:
                return json.dumps({
                    "success": True,
                    "message": f"All keys already set: {', '.join(already_set)}",
                    "keys_provided": [],
                    "keys_already_set": already_set
                })
            
            # Prompt user for missing keys
            if not self.interactive_handler:
                return json.dumps({
                    "success": False,
                    "error": "No interactive handler available",
                    "missing_keys": missing,
                    "suggestion": "Set these environment variables manually and retry"
                })
            
            print(f"\n⚠️  Missing required credentials: {', '.join(missing)}")
            print(f"Reason: {reason}\n")
            
            # Format keys for prompt
            key_info = [{'key': key, 'description': reason} for key in missing]
            
            # Prompt user
            provided = self.interactive_handler.prompt_for_api_keys(key_info)
            
            # Set in environment
            for key, value in provided.items():
                os.environ[key] = value
            
            if provided:
                return json.dumps({
                    "success": True,
                    "message": f"User provided {len(provided)} credential(s)",
                    "keys_provided": list(provided.keys()),
                    "keys_already_set": already_set
                })
            else:
                return json.dumps({
                    "success": False,
                    "message": "User skipped providing credentials",
                    "keys_provided": [],
                    "keys_already_set": already_set,
                    "warning": "Experiment will likely fail without these credentials"
                })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class AnalyzePlotSemantics(Tool):
    name = "analyze_plot_semantics"
    description = """Uses vision-language model to semantically understand and compare plots/figures.
    This provides deeper analysis than pixel comparison - it interprets what the plot shows.
    
    Args:
        image_path: Path to the plot/figure image file
        question: What to analyze (e.g., "What trends does this plot show?", "What metrics are displayed?")
        reference_image_path: Optional reference image to compare against
        
    Returns:
        Semantic description and analysis of the plot
    """
    
    inputs = {
        "image_path": {
            "type": "string",
            "description": "Path to the plot/figure image"
        },
        "question": {
            "type": "string",
            "description": "What to analyze about the plot"
        },
        "reference_image_path": {
            "type": "string",
            "description": "Optional reference image path for comparison",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, vision_model_client=None):
        """Initialize with optional vision model client."""
        super().__init__()
        self.vision_client = vision_model_client
    
    def forward(self, image_path: str, question: str, reference_image_path: Optional[str] = None) -> str:
        """Analyze plot using vision-language model."""
        try:
            import base64
            from pathlib import Path
            
            # Load image
            img_path = Path(image_path)
            if not img_path.exists():
                return json.dumps({"error": f"Image not found: {image_path}"})
            
            # Encode image as base64
            with open(img_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Prepare reference image if provided
            reference_data = None
            if reference_image_path:
                ref_path = Path(reference_image_path)
                if ref_path.exists():
                    with open(ref_path, 'rb') as f:
                        reference_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Use vision model if available
            if self.vision_client:
                messages = []
                
                if reference_data:
                    # Comparison mode
                    prompt = f"""{question}
                    
                    Compare these two plots/figures:
                    - First image: Reference/original from paper
                    - Second image: Reproduced from running code
                    
                    Analyze:
                    1. What data/trends does each show?
                    2. Are the trends/patterns similar?
                    3. Are the scales/axes comparable?
                    4. Overall: Do these plots show reproducible results?
                    """
                    
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{reference_data}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                            ]
                        }
                    ]
                else:
                    # Single image analysis
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                            ]
                        }
                    ]
                
                # Call vision model
                response = self.vision_client.chat.completions.create(
                    model="gpt-4o",  # or another vision-capable model
                    messages=messages,
                    max_tokens=1000
                )
                
                analysis = response.choices[0].message.content
                
                return json.dumps({
                    "success": True,
                    "analysis": analysis,
                    "image_analyzed": str(img_path),
                    "reference_used": str(reference_image_path) if reference_image_path else None
                })
            else:
                return json.dumps({
                    "error": "Vision model not available",
                    "suggestion": "Configure OpenAI API key with vision-capable model"
                })
                
        except Exception as e:
            return json.dumps({"error": f"Error analyzing plot: {str(e)}"})


class ExtractTableMetrics(Tool):
    name = "extract_table_metrics"
    description = """Extracts numerical metrics from tables in text or images.
    Can parse markdown tables, CSV-like text, or use OCR on table images.
    
    Args:
        source: Table source (text with table, or path to image)
        source_type: Either 'text' or 'image'
        
    Returns:
        JSON with extracted metrics from the table
    """
    
    inputs = {
        "source": {
            "type": "string",
            "description": "Table content as text or path to table image"
        },
        "source_type": {
            "type": "string",
            "description": "Either 'text' or 'image'"
        }
    }
    output_type = "string"
    
    def __init__(self, llm_client=None):
        super().__init__()
        self.llm_client = llm_client
    
    def forward(self, source: str, source_type: str = "text") -> str:
        """Extract metrics from table."""
        try:
            if source_type == "image":
                # Use OCR or vision model for table extraction
                from pathlib import Path
                import base64
                
                img_path = Path(source)
                if not img_path.exists():
                    return json.dumps({"error": f"Image not found: {source}"})
                
                if self.llm_client and hasattr(self.llm_client, 'chat'):
                    # Use vision model to extract table data
                    with open(img_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all numerical metrics from this table. Return as JSON with metric names as keys and values as numbers."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                        ]
                    }]
                    
                    response = self.llm_client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=1000
                    )
                    
                    content = response.choices[0].message.content
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\{[^}]+\}', content)
                    if json_match:
                        metrics = json.loads(json_match.group(0))
                        return json.dumps({"success": True, "metrics": metrics})
                    
                return json.dumps({"error": "Could not extract table from image"})
            
            else:
                # Parse text table
                metrics = {}
                lines = source.split('\n')
                
                for line in lines:
                    # Look for patterns like "Metric: Value" or "| Metric | Value |"
                    import re
                    
                    # Markdown table row
                    if '|' in line:
                        parts = [p.strip() for p in line.split('|') if p.strip()]
                        if len(parts) >= 2:
                            key = parts[0]
                            for val in parts[1:]:
                                try:
                                    num = float(val.replace('%', '').strip())
                                    metrics[f"{key}_{parts.index(val)}"] = num
                                except ValueError:
                                    continue
                    
                    # Key: Value format
                    elif ':' in line:
                        key, val = line.split(':', 1)
                        try:
                            num = float(val.strip().replace('%', ''))
                            metrics[key.strip()] = num
                        except ValueError:
                            continue
                
                return json.dumps({"success": True, "metrics": metrics})
                
        except Exception as e:
            return json.dumps({"error": f"Error extracting table metrics: {str(e)}"})


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

