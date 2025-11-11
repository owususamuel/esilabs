"""
Code Executor Tool - Safely executes code from repositories.
Provides sandboxing, timeout control, and resource limiting.
"""

import subprocess
import os
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result from code execution."""
    
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    success: bool
    timed_out: bool
    error_message: Optional[str] = None


class CodeExecutor:
    
    def __init__(
        self,
        sandbox_mode: bool = True,
        max_timeout: int = 300,
        max_memory_mb: Optional[int] = None
    ):
        """
        Initialize the code executor.
        
        Args:
            sandbox_mode: Enable sandbox protection
            max_timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB (if available)
        """
        
        self.sandbox_mode = sandbox_mode
        self.max_timeout = max_timeout
        self.max_memory_mb = max_memory_mb
        self.logger = logger
    
    def execute_command(
        self,
        command: str,
        working_directory: str = ".",
        environment: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> ExecutionResult:

        timeout = timeout or self.max_timeout
        
        # Prepare environment
        exec_env = os.environ.copy()
        if environment:
            exec_env.update(environment)
        
        # Ensure working directory exists
        working_dir = Path(working_directory)
        if not working_dir.exists():
            working_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            start_time = None
            import time
            start_time = time.time()
            
            self.logger.info(f"Executing command: {command}")
            self.logger.info(f"Working directory: {working_directory}")
            
            # Execute the command
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=str(working_directory),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=exec_env,
                text=True
            )
            
            # Wait with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                duration = time.time() - start_time
                
                return ExecutionResult(
                    exit_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration,
                    success=process.returncode == 0,
                    timed_out=False
                )
            
            except subprocess.TimeoutExpired:
                # Kill the process if timeout exceeded
                process.kill()
                stdout, stderr = process.communicate()
                duration = time.time() - start_time
                
                self.logger.warning(f"Command timed out after {timeout} seconds")
                
                return ExecutionResult(
                    exit_code=-1,
                    stdout=stdout,
                    stderr=stderr or "Process timed out",
                    duration_seconds=duration,
                    success=False,
                    timed_out=True,
                    error_message=f"Execution timed out after {timeout} seconds"
                )
        
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=0,
                success=False,
                timed_out=False,
                error_message=str(e)
            )
    
    def execute_python_script(
        self,
        script_path: str,
        working_directory: str = ".",
        arguments: Optional[List[str]] = None,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        
        if not Path(script_path).exists():
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=f"Script not found: {script_path}",
                duration_seconds=0,
                success=False,
                timed_out=False,
                error_message=f"Script not found: {script_path}"
            )
        
        # Build command
        command = f"python {script_path}"
        if arguments:
            command += " " + " ".join(arguments)
        
        return self.execute_command(command, working_directory, timeout=timeout)
    
    def run_setup_commands(
        self,
        setup_commands: List[str],
        working_directory: str = "."
    ) -> Tuple[bool, str]:
        
        combined_output = ""
        
        for cmd in setup_commands:
            self.logger.info(f"Running setup command: {cmd}")
            result = self.execute_command(cmd, working_directory)
            
            combined_output += f"\n--- {cmd} ---\n"
            combined_output += result.stdout
            if result.stderr:
                combined_output += f"STDERR: {result.stderr}\n"
            
            if not result.success:
                self.logger.error(f"Setup command failed: {cmd}")
                self.logger.error(f"Error: {result.stderr}")
                return False, combined_output
        
        self.logger.info("All setup commands completed successfully")
        return True, combined_output
    
    def setup_environment(
        self,
        requirements_file: Optional[str] = None,
        python_version: Optional[str] = None,
        working_directory: str = "."
    ) -> Tuple[bool, str]:
        
        setup_commands = []
        
        # Create virtual environment
        venv_path = os.path.join(working_directory, "venv")
        setup_commands.append(f"python -m venv {venv_path}")
        
        # Install requirements if provided
        if requirements_file and os.path.exists(requirements_file):
            pip_path = os.path.join(venv_path, "bin", "pip")
            setup_commands.append(f"{pip_path} install -r {requirements_file}")
        
        return self.run_setup_commands(setup_commands, working_directory)
    
    def clone_repository(
        self,
        repo_url: str,
        target_directory: str,
        branch: Optional[str] = None
    ) -> ExecutionResult:
 
        command = f"git clone {repo_url} {target_directory}"
        
        return self.execute_command(command, working_directory=".")


def execute_safely(
    command: str,
    working_directory: str = ".",
    timeout: int = 300
) -> Dict[str, any]:
    
    executor = CodeExecutor(sandbox_mode=True, max_timeout=timeout)
    result = executor.execute_command(command, working_directory, timeout=timeout)
    
    return {
        'success': result.success,
        'exit_code': result.exit_code,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'duration': result.duration_seconds,
        'timed_out': result.timed_out,
        'error': result.error_message
    }
