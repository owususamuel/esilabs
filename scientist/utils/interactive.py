
import os
import re
import logging
import getpass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from urllib.parse import urlparse
from datetime import datetime
import shutil
import subprocess

logger = logging.getLogger(__name__)


class InteractiveInputHandler:
    """
    Handles interactive user input for repository discovery fallback.
    
    When automated search fails, this class:
    1. Prompts user for repository URL or local zip file
    2. Validates inputs
    3. Prepares repository for experiment execution
    """
    
    # Use project temp directory for consistency
    TEMP_DIR = Path("./data/temp/experiments")
    
    def __init__(self, timeout: int = 300):
        """
        Initialize the interactive input handler.
        
        Args:
            timeout: Maximum wait time for user input (seconds)
        """
        self.timeout = timeout
        self.logger = logger
        # Ensure temp directory exists
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load .env file if exists
        self._load_env_file()
    
    def prompt_for_repository(self) -> Optional[Dict[str, Any]]:
        """
        Prompt user to provide repository via multiple options.
        
        Returns:
            Dictionary with repository info or None if user cancels
        """
        
        print("\n" + "="*60)
        print("NO REPOSITORY FOUND IN PAPER - INTERACTIVE MODE")
        print("="*60)
        print("\nNo GitHub repository link was found in the paper.")
        print("Please choose how to proceed:\n")
        print("  1. Upload Local ZIP file")
        print("  2. Add GitHub Repository URL manually")
        print("  3. Let agent search GitHub automatically\n")
        
        while True:
            user_choice = input("Choose [1/2/3] or [q]uit: ").strip().lower()
            
            if user_choice == 'q':
                print("\nAborted by user.")
                return None
            
            if user_choice == '1':
                return self._prompt_for_zip()
            elif user_choice == '2':
                return self._prompt_for_url()
            elif user_choice == '3':
                return {'type': 'agent_search', 'manual': False}
            else:
                print("Invalid choice. Please enter 1, 2, 3, or q.")
    
    def prompt_after_failed_search(self) -> Optional[str]:
        """
        Prompt user after agent search returns empty results.
        
        Returns:
            'generate' to generate code, 'quit' to quit, or None
        """
        
        print("\n" + "="*60)
        print("AGENT SEARCH RETURNED NO RESULTS")
        print("="*60)
        print("\nThe agent couldn't find any matching repositories on GitHub.")
        print("What would you like to do?\n")
        print("  1. Generate code from methodology and experimentation setup")
        print("  2. Quit\n")
        
        while True:
            user_choice = input("Choose [1/2]: ").strip().lower()
            
            if user_choice == '1':
                print("\n✓ Will generate code from paper methodology...")
                return 'generate'
            elif user_choice == '2':
                print("\nQuitting...")
                return 'quit'
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    def _prompt_for_url(self) -> Optional[Dict[str, Any]]:
        """
        Prompt user for repository URL.
        
        Returns:
            Dictionary with repository info or None if invalid
        """
        
        print("\n" + "-"*60)
        print("Enter Repository URL")
        print("-"*60)
        print("Examples: https://github.com/user/repo")
        print("          (or just 'user/repo' for GitHub)\n")
        
        url = input("Repository URL: ").strip()
        
        if not url:
            print("Cancelled.")
            return None
        
        # Handle GitHub shorthand (user/repo)
        if re.match(r'^[\w\-]+/[\w\-]+$', url):
            url = f"https://github.com/{url}"
        
        # Validate URL format
        if not self._is_valid_url(url):
            print(f"❌ Invalid URL format: {url}")
            return None
        
        # Parse repository info
        repo_info = self._parse_repo_url(url)
        if not repo_info:
            print(f"❌ Could not parse repository URL: {url}")
            return None
        
        print(f"\n✓ Repository URL: {url}")
        print(f"  Name: {repo_info['name']}")
        print(f"  Source: {repo_info['source']}")
        
        # Confirm
        confirm = input("\nProceed with this repository? [y/n]: ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return None
        
        return {
            'type': 'url',
            'url': url,
            'name': repo_info['name'],
            'source': repo_info['source'],
            'manual': True
        }
    
    def _prompt_for_zip(self) -> Optional[Dict[str, Any]]:
        """
        Prompt user for local ZIP file.
        
        Returns:
            Dictionary with repository info or None if invalid
        """
        
        print("\n" + "-"*60)
        print("Select ZIP File")
        print("-"*60)
        print("Enter path to ZIP file containing the repository code.\n")
        
        while True:
            zip_path = input("ZIP file path: ").strip()
            
            if not zip_path:
                print("Cancelled.")
                return None
            
            # Expand user path
            zip_path = os.path.expanduser(zip_path)
            
            # Check if file exists
            if not os.path.isfile(zip_path):
                print(f"❌ File not found: {zip_path}")
                continue
            
            # Check if it's a zip file
            if not zip_path.endswith('.zip'):
                print(f"❌ File must be a ZIP archive: {zip_path}")
                continue
            
            # Try to validate zip
            if not self._is_valid_zip(zip_path):
                print(f"❌ Invalid ZIP file: {zip_path}")
                continue
            
            print(f"✓ ZIP file: {zip_path}")
            
            # Extract repo name from zip filename
            repo_name = Path(zip_path).stem
            print(f"  Name: {repo_name}")
            
            # Confirm
            confirm = input("\nProceed with this file? [y/n]: ").strip().lower()
            if confirm not in ('', 'y', 'yes'):
                print("Cancelled.")
                return None
            
            return {
                'type': 'zip',
                'path': zip_path,
                'name': repo_name,
                'manual': True
            }
    
    def _is_valid_url(self, url: str) -> bool:
        
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _is_valid_zip(self, zip_path: str) -> bool:

        try:
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Check if zip has content
                if not zip_ref.namelist():
                    return False
            return True
        except Exception as e:
            self.logger.warning(f"Invalid ZIP file {zip_path}: {e}")
            return False
    
    def _parse_repo_url(self, url: str) -> Optional[Dict[str, str]]:

        try:
            parsed = urlparse(url)
            
            # Only support GitHub
            if 'github.com' not in parsed.netloc:
                self.logger.warning(f"Only GitHub URLs are supported: {url}")
                return None
            
            source = 'github'
            
            # Extract repo name from path
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) >= 2:
                name = path_parts[1].replace('.git', '')
            else:
                name = 'unknown'
            
            return {
                'name': name,
                'source': source,
                'url': url
            }
        
        except Exception as e:
            self.logger.warning(f"Error parsing repo URL {url}: {e}")
            return None
    
    def prepare_repository(self, repo_info: Dict[str, Any]) -> Optional[str]:
        """
        Prepare repository for experiment execution.
        
        For URLs: clones the repository
        For ZIP files: extracts to temporary directory
        For agent_search: returns None (handled separately in main flow)
        
        Args:
            repo_info: Repository information from prompt
            
        Returns:
            Path to repository directory or None if failed/agent_search
        """
        
        try:
            if repo_info['type'] == 'url':
                return self._clone_repository(repo_info['url'], repo_info['name'])
            elif repo_info['type'] == 'zip':
                return self._extract_zip(repo_info['path'], repo_info['name'])
            elif repo_info['type'] == 'agent_search':
                # This will be handled by the agent in the main flow
                return None
        
        except Exception as e:
            self.logger.error(f"Error preparing repository: {e}")
            print(f"\n❌ Error preparing repository: {e}")
            return None
    
    def _clone_repository(self, url: str, name: str) -> Optional[str]:

        try:
            # Create temp directory in project temp (not system temp)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = self.TEMP_DIR / f"{name}_{timestamp}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nCloning repository to project temp directory...")
            print(f"  Path: {temp_dir}")
            
            # Clone repository
            result = subprocess.run(
                ["git", "clone", url, str(temp_dir)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                self.logger.error(f"Git clone failed: {result.stderr}")
                print(f"❌ Failed to clone repository: {result.stderr}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None
            
            print(f"✓ Repository cloned successfully")
            return str(temp_dir)
        
        except subprocess.TimeoutExpired:
            print("❌ Clone operation timed out")
            return None
        except Exception as e:
            self.logger.error(f"Error cloning repository: {e}")
            print(f"❌ Error cloning repository: {e}")
            return None
    
    def _extract_zip(self, zip_path: str, name: str) -> Optional[str]:

        try:
            import zipfile
            
            # Create temp directory in project temp (not system temp)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = self.TEMP_DIR / f"{name}_{timestamp}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nExtracting ZIP file to project temp directory...")
            print(f"  Path: {temp_dir}")
            
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(str(temp_dir))
            
            # If zip has single root folder, use that
            contents = os.listdir(temp_dir)
            if len(contents) == 1:
                single_dir = temp_dir / contents[0]
                if single_dir.is_dir():
                    # Move contents up one level
                    final_dir = self.TEMP_DIR / f"{name}_extracted_{timestamp}"
                    final_dir.mkdir(parents=True, exist_ok=True)
                    for item in os.listdir(single_dir):
                        src = single_dir / item
                        dst = final_dir / item
                        shutil.move(str(src), str(dst))
                    shutil.rmtree(temp_dir)
                    temp_dir = final_dir
            
            print(f"✓ ZIP file extracted successfully")
            return str(temp_dir)
        
        except Exception as e:
            self.logger.error(f"Error extracting ZIP file: {e}")
            print(f"❌ Error extracting ZIP file: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None
    
    def prompt_for_api_keys(self, required_keys: List[Dict[str, str]]) -> Dict[str, str]:

        print("\n" + "="*60)
        print("⚠️  REQUIRED API KEYS MISSING")
        print("="*60)
        print("\nThis experiment requires the following API keys:\n")
        
        for item in required_keys:
            key = item['key']
            desc = item.get('description', 'Not specified')
            print(f"  • {key}")
            if desc and desc != 'Not specified':
                print(f"    {desc}")
        
        print("\nYou can provide them now, or set them in a .env file.")
        print("Leave empty to skip (experiment may fail).\n")
        
        keys_provided = {}
        
        for item in required_keys:
            key = item['key']
            
            # Check if already in environment
            if os.getenv(key):
                print(f"✓ {key}: Already set in environment")
                continue
            
            # Prompt for key (masked input)
            while True:
                value = getpass.getpass(f"Enter {key} (or press Enter to skip): ")
                
                if not value:
                    print(f"  Skipped {key}")
                    break
                
                # Basic validation
                if len(value.strip()) < 5:
                    print("  ⚠️  Value seems too short. Try again or skip.")
                    continue
                
                keys_provided[key] = value.strip()
                print(f"  ✓ {key} saved")
                break
        
        print()
        return keys_provided
    
    def confirm(self, message: str) -> bool:

        while True:
            response = input(f"{message} [y/n]: ").strip().lower()
            if response in ('y', 'yes'):
                return True
            elif response in ('n', 'no'):
                return False
            else:
                print("Please enter 'y' or 'n'.")
    
    def _load_env_file(self):
        """
        Load .env file from project root if it exists.
        Uses python-dotenv if available, otherwise manual parsing.
        """
        
        try:
            # Try using python-dotenv if available
            from dotenv import load_dotenv
            
            # Load from project root
            project_root = Path.cwd()
            env_file = project_root / ".env"
            
            if env_file.exists():
                load_dotenv(env_file)
                self.logger.info(f"Loaded .env file from {env_file}")
            
        except ImportError:
            # Fallback: manual .env parsing
            self._load_env_manual()
    
    def _load_env_manual(self):
        """
        Manually parse and load .env file.
        Simple fallback if python-dotenv is not available.
        """
        
        try:
            project_root = Path.cwd()
            env_file = project_root / ".env"
            
            if not env_file.exists():
                return
            
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse KEY=VALUE
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        
                        # Only set if not already in environment
                        if key and not os.getenv(key):
                            os.environ[key] = value
            
            self.logger.info(f"Manually loaded .env file from {env_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load .env file manually: {e}")


def get_repository_interactively() -> Optional[Tuple[str, Dict[str, Any]]]:

    handler = InteractiveInputHandler()
    repo_info = handler.prompt_for_repository()
    
    if not repo_info:
        return None
    
    repo_path = handler.prepare_repository(repo_info)
    
    if not repo_path:
        return None
    
    return repo_path, repo_info
