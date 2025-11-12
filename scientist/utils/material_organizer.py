"""
Material Organizer - Saves and organizes paper materials and reproduced results.

This module ensures proper side-by-side comparison by saving:
1. Paper materials (figures, tables, metrics) to organized folders
2. Reproduced results to organized folders
3. Creating manifest files for easy LLM access
"""

import json
import base64
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class MaterialOrganizer:
    """Organizes materials for human-like comparison."""
    
    def __init__(self, run_id: str, base_output_dir: str = "./data/outputs"):
        """
        Initialize organizer for a specific run.
        
        Args:
            run_id: Unique run identifier (e.g., "20251111_225001")
            base_output_dir: Base directory for all outputs
        """
        self.run_id = run_id
        self.base_dir = Path(base_output_dir) / run_id
        
        # Create organized folder structure
        self.paper_materials_dir = self.base_dir / "paper_materials"
        self.reproduced_results_dir = self.base_dir / "reproduced_results"
        self.comparison_dir = self.base_dir / "comparison"
        
        # Create directories
        for directory in [self.paper_materials_dir, self.reproduced_results_dir, self.comparison_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Created organized folders for run {run_id}")
        logger.info(f"   Paper materials: {self.paper_materials_dir}")
        logger.info(f"   Reproduced results: {self.reproduced_results_dir}")
        logger.info(f"   Comparison: {self.comparison_dir}")
    
    def save_paper_materials(
        self,
        figures: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        metrics: Dict[str, float],
        paper_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Save all extracted paper materials to organized folders.
        
        Args:
            figures: List of figure dicts with image_base64, caption, figure_number
            tables: List of table dicts with content, caption, table_number
            metrics: Extracted metrics from paper
            paper_data: Full paper data dictionary
            
        Returns:
            Dictionary with paths to saved materials
        """
        saved_paths = {}
        
        # 1. Save figures as actual image files
        figures_dir = self.paper_materials_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        saved_figures = []
        for i, figure in enumerate(figures, 1):
            try:
                # Get image data
                image_b64 = figure.get('image_base64', '')
                if not image_b64:
                    continue
                
                # Decode and save
                image_data = base64.b64decode(image_b64)
                fig_num = figure.get('figure_number', i)
                caption = figure.get('caption', '')
                
                # Save as PNG
                figure_path = figures_dir / f"figure_{fig_num}.png"
                with open(figure_path, 'wb') as f:
                    f.write(image_data)
                
                # Save metadata
                metadata_path = figures_dir / f"figure_{fig_num}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'figure_number': fig_num,
                        'caption': caption,
                        'page_number': figure.get('page_number', 0),
                        'image_file': f"figure_{fig_num}.png"
                    }, f, indent=2)
                
                saved_figures.append({
                    'figure_number': fig_num,
                    'image_path': str(figure_path),
                    'metadata_path': str(metadata_path),
                    'caption': caption
                })
                
                logger.info(f"   ✓ Saved Figure {fig_num}: {caption[:50]}...")
                
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to save figure {i}: {e}")
                continue
        
        saved_paths['figures'] = saved_figures
        
        # 2. Save tables as text files
        tables_dir = self.paper_materials_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        
        saved_tables = []
        for i, table in enumerate(tables, 1):
            try:
                tbl_num = table.get('table_number', i)
                caption = table.get('caption', '')
                content = table.get('content', '')
                
                # Save table content
                table_path = tables_dir / f"table_{tbl_num}.txt"
                with open(table_path, 'w') as f:
                    f.write(f"Table {tbl_num}: {caption}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(content)
                
                # Save metadata
                metadata_path = tables_dir / f"table_{tbl_num}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'table_number': tbl_num,
                        'caption': caption,
                        'page_number': table.get('page_number', 0),
                        'content_file': f"table_{tbl_num}.txt"
                    }, f, indent=2)
                
                saved_tables.append({
                    'table_number': tbl_num,
                    'content_path': str(table_path),
                    'metadata_path': str(metadata_path),
                    'caption': caption
                })
                
                logger.info(f"   ✓ Saved Table {tbl_num}: {caption[:50]}...")
                
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to save table {i}: {e}")
                continue
        
        saved_paths['tables'] = saved_tables
        
        # 3. Save extracted metrics
        metrics_path = self.paper_materials_dir / "extracted_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        saved_paths['metrics'] = str(metrics_path)
        logger.info(f"   ✓ Saved {len(metrics)} extracted metrics")
        
        # 4. Create manifest file for easy access
        manifest_path = self.paper_materials_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump({
                'run_id': self.run_id,
                'paper_title': paper_data.get('title', ''),
                'figures_count': len(saved_figures),
                'tables_count': len(saved_tables),
                'metrics_count': len(metrics),
                'figures': saved_figures,
                'tables': saved_tables,
                'paths': saved_paths
            }, f, indent=2)
        
        logger.info(f"✅ Paper materials saved to: {self.paper_materials_dir}")
        logger.info(f"   - {len(saved_figures)} figures")
        logger.info(f"   - {len(saved_tables)} tables")
        logger.info(f"   - {len(metrics)} metrics")
        
        return saved_paths
    
    def organize_reproduced_results(
        self,
        repo_path: str,
        output_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Copy and organize reproduced results from experiment execution.
        
        Args:
            repo_path: Path to repository where experiment was run
            output_files: Optional list of specific output files to copy
            
        Returns:
            Dictionary with organized result paths
        """
        organized = {
            'plots': [],
            'data_files': [],
            'metrics_files': [],
            'logs': []
        }
        
        repo_path_obj = Path(repo_path)
        if not repo_path_obj.exists():
            logger.warning(f"Repository path does not exist: {repo_path}")
            return organized
        
        logger.info(f"📦 Organizing reproduced results from: {repo_path}")
        
        # Common output directory names to search
        output_dir_names = [
            'outputs', 'output', 'results', 'figures', 'plots',
            'experiments', 'runs', 'outputs_all_methods'
        ]
        
        # Extensions to look for
        plot_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.svg', '.eps'}
        data_extensions = {'.csv', '.json', '.txt', '.tsv', '.xlsx'}
        log_extensions = {'.log'}
        
        # EXCLUDE patterns (noise filters)
        exclude_dirs = {
            '.git', '__pycache__', 'node_modules', '.venv', 'venv',
            'site-packages', 'dist-packages', 'lib', 'bin', 'include',
            'test', 'tests', '__pycache__', '.pytest_cache',
            'images', 'mpl-data'  # matplotlib data
        }
        
        # Exclude file patterns (library/test data, not experiment results)
        exclude_file_patterns = [
            # Matplotlib UI icons
            'back.', 'forward.', 'home.', 'hand.', 'move.', 'zoom_to_rect.',
            'filesave.', 'help.', 'subplots.', 'qt4_editor_options.',
            'matplotlib.', 'dots.png', 'logo2.png', 'console.png',
            # UI screenshots and demos (NOT experiment plots!)
            'chat_interface', 'screenshot', 'demo_', 'interface_',
            'gui_', 'window_', 'dialog_', 'menu_', 'button_',
            'ui_', 'app_screenshot', 'screen_capture',
            # Library test data
            'umath-validation-set', 'pdist-', 'cdist-', 'random-',
            'philox-testset', 'sfc64-testset', 'mt19937-testset', 'pcg64',
            'svmlight_', 'label_', 'test_region_', 'test_display_',
            'test_house_', 'Minduka_Present',
            # Generic test files
            'iris.csv', 'wine_data.csv', 'breast_cancer.csv', 'linnerud_',
            'mnist', 'cifar',
            # Metadata files
            'LICENSE', 'README', 'NOTICE', 'AUTHORS', 'COPYING',
            'top_level', 'entry_points', 'SOURCES', 'dependency_links',
            'Grammar.txt', 'PatternGrammar.txt', 'template.txt',
            'iso', 'xhtml1-', 'mmlalias', 'mmlextra', 'html-roles',
            'api_tests.txt', 'setupcfg_examples',
            # Package metadata
            'package.json', 'uv_build.json', 'install.json',
            'metaschema.json', 'schema.json', 'black.schema.json',
            'setuptools.schema.json', 'distutils.schema.json',
            # Generic library data  
            'grace_hopper.jpg', 'china.jpg', 'flower.jpg'
        ]
        
        # IMPORTANT: Only keep plots that are likely EXPERIMENT results
        # Patterns for scientific/experiment plots we WANT to keep
        experiment_plot_keywords = [
            'accuracy', 'loss', 'error', 'metric', 'performance',
            'confusion_matrix', 'roc_curve', 'roc', 'auc',
            'precision', 'recall', 'f1_score', 'f1',
            'training', 'validation', 'test_set',
            'histogram', 'distribution', 'scatter', 'correlation',
            'heatmap', 'comparison', 'bar_chart', 'line_plot',
            'learning_curve', 'convergence', 'epoch',
            'prediction', 'result', 'score', 'evaluation',
            'visualization', 'analysis', 'experiment',
            'figure_', 'fig_', 'plot_', 'graph_',
            'mrr', 'recall@', 'ndcg', 'map', 'bleu', 'rouge'
        ]
        
        # Search for output files
        found_dirs = []
        for dir_name in output_dir_names:
            for match in repo_path_obj.rglob(dir_name):
                if match.is_dir():
                    found_dirs.append(match)
        
        # Also check root directory
        found_dirs.append(repo_path_obj)
        
        #Helper function to check if file should be excluded
        def should_exclude_file(file_path: Path) -> bool:
            """Check if file matches exclude patterns."""
            filename = file_path.name
            
            # Check if in excluded directory
            for parent in file_path.parents:
                if parent.name in exclude_dirs:
                    return True
            
            # Check if filename matches exclude patterns
            for pattern in exclude_file_patterns:
                if pattern in filename:
                    return True
            
            return False
        
        def is_experiment_plot(file_path: Path) -> bool:
            """Check if plot file is likely an experiment result (not UI/demo)."""
            filename_lower = file_path.name.lower()
            
            # Check if filename contains experiment-related keywords
            for keyword in experiment_plot_keywords:
                if keyword in filename_lower:
                    return True
            
            return False
        
        # Copy organized files
        files_skipped = 0
        for search_dir in found_dirs:
            try:
                for file_path in search_dir.rglob('*'):
                    if not file_path.is_file():
                        continue
                    
                    # Skip hidden files
                    if file_path.name.startswith('.'):
                        continue
                    
                    # Skip excluded files (noise filtering)
                    if should_exclude_file(file_path):
                        files_skipped += 1
                        continue
                    
                    suffix = file_path.suffix.lower()
                    
                    # Categorize and copy
                    if suffix in plot_extensions:
                        # IMPORTANT: Only save plots that are likely experiment results!
                        if not is_experiment_plot(file_path):
                            files_skipped += 1
                            logger.debug(f"   ⊗ Skipping non-experiment plot: {file_path.name}")
                            continue
                        
                        # Copy to plots directory
                        plots_dir = self.reproduced_results_dir / "plots"
                        plots_dir.mkdir(exist_ok=True)
                        
                        dest_path = plots_dir / file_path.name
                        # Avoid duplicates
                        counter = 1
                        while dest_path.exists():
                            dest_path = plots_dir / f"{file_path.stem}_{counter}{suffix}"
                            counter += 1
                        
                        shutil.copy2(file_path, dest_path)
                        organized['plots'].append(str(dest_path))
                        logger.info(f"   ✓ Copied experiment plot: {file_path.name}")
                    
                    elif suffix in data_extensions:
                        # Copy to data directory
                        data_dir = self.reproduced_results_dir / "data"
                        data_dir.mkdir(exist_ok=True)
                        
                        dest_path = data_dir / file_path.name
                        counter = 1
                        while dest_path.exists():
                            dest_path = data_dir / f"{file_path.stem}_{counter}{suffix}"
                            counter += 1
                        
                        shutil.copy2(file_path, dest_path)
                        
                        # Prioritize actual experiment results as metrics files
                        filename_lower = file_path.name.lower()
                        is_metrics_file = (
                            filename_lower.startswith('results_') or
                            filename_lower.startswith('complete_results') or
                            filename_lower.startswith('metrics_') or
                            'performance' in filename_lower or
                            'evaluation' in filename_lower or
                            ('result' in filename_lower and not filename_lower.startswith('test_'))
                        )
                        
                        if is_metrics_file:
                            organized['metrics_files'].append(str(dest_path))
                            logger.info(f"   ✓ Copied metrics file: {file_path.name}")
                        else:
                            organized['data_files'].append(str(dest_path))
                            logger.info(f"   ✓ Copied data file: {file_path.name}")
                    
                    elif suffix in log_extensions:
                        # Copy to logs directory
                        logs_dir = self.reproduced_results_dir / "logs"
                        logs_dir.mkdir(exist_ok=True)
                        
                        dest_path = logs_dir / file_path.name
                        counter = 1
                        while dest_path.exists():
                            dest_path = logs_dir / f"{file_path.stem}_{counter}{suffix}"
                            counter += 1
                        
                        shutil.copy2(file_path, dest_path)
                        organized['logs'].append(str(dest_path))
                        logger.info(f"   ✓ Copied log: {file_path.name}")
            
            except Exception as e:
                logger.warning(f"Error processing directory {search_dir}: {e}")
                continue
        
        # Create manifest
        manifest_path = self.reproduced_results_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump({
                'run_id': self.run_id,
                'repo_path': str(repo_path),
                'plots_count': len(organized['plots']),
                'data_files_count': len(organized['data_files']),
                'metrics_files_count': len(organized['metrics_files']),
                'logs_count': len(organized['logs']),
                'organized': organized
            }, f, indent=2)
        
        logger.info(f"✅ Reproduced results organized in: {self.reproduced_results_dir}")
        logger.info(f"   - {len(organized['plots'])} experiment plots (scientific visualizations only)")
        logger.info(f"   - {len(organized['data_files'])} data files")
        logger.info(f"   - {len(organized['metrics_files'])} metrics files")
        logger.info(f"   - {len(organized['logs'])} log files")
        if files_skipped > 0:
            logger.info(f"   ⊗ Filtered out {files_skipped} irrelevant files (UI screenshots, library data, test files)")
        
        return organized
    
    def create_comparison_manifest(
        self,
        paper_materials_saved: Dict[str, Any],
        reproduced_organized: Dict[str, Any]
    ) -> str:
        """
        Create a comparison manifest that maps paper materials to reproduced results.
        
        This manifest makes it easy for the LLM to see what needs to be compared.
        
        Args:
            paper_materials_saved: Output from save_paper_materials()
            reproduced_organized: Output from organize_reproduced_results()
            
        Returns:
            Path to comparison manifest file
        """
        manifest = {
            'run_id': self.run_id,
            'comparison_structure': {
                'paper_materials': {
                    'directory': str(self.paper_materials_dir),
                    'figures': paper_materials_saved.get('figures', []),
                    'tables': paper_materials_saved.get('tables', []),
                    'metrics_file': paper_materials_saved.get('metrics', '')
                },
                'reproduced_results': {
                    'directory': str(self.reproduced_results_dir),
                    'plots': reproduced_organized.get('plots', []),
                    'data_files': reproduced_organized.get('data_files', []),
                    'metrics_files': reproduced_organized.get('metrics_files', []),
                    'logs': reproduced_organized.get('logs', [])
                }
            },
            'instructions_for_llm': {
                'task': 'Compare paper materials with reproduced results',
                'steps': [
                    '1. Read paper figures from paper_materials/figures/',
                    '2. Read paper tables from paper_materials/tables/',
                    '3. Read paper metrics from paper_materials/extracted_metrics.json',
                    '4. Read reproduced plots from reproduced_results/plots/',
                    '5. Read reproduced metrics from reproduced_results/data/',
                    '6. Compare figures with plots visually',
                    '7. Compare tables with data files numerically',
                    '8. Compare extracted metrics with reproduced metrics',
                    '9. Provide detailed reproducibility assessment'
                ]
            }
        }
        
        manifest_path = self.comparison_dir / "comparison_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"✅ Comparison manifest created: {manifest_path}")
        
        return str(manifest_path)
    
    def get_paper_materials_dir(self) -> Path:
        """Get the paper materials directory."""
        return self.paper_materials_dir
    
    def get_reproduced_results_dir(self) -> Path:
        """Get the reproduced results directory."""
        return self.reproduced_results_dir
    
    def get_comparison_dir(self) -> Path:
        """Get the comparison directory."""
        return self.comparison_dir

