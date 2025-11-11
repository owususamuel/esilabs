
from typing import Dict, Any, Optional, List
import io
import math
import base64
from PIL import Image

from scientist.agents.base_agent import BaseAgent
from scientist.tools.result_comparator import ResultComparator
from scientist.tools.tool_wrappers import ExtractMetrics


class EvaluatorAgent(BaseAgent):
    """
    Autonomous agent that evaluates reproducibility of research experiments.
    
    The agent autonomously:
    1. Compares original paper results with reproduced results
    2. Analyzes differences and their significance
    3. Assesses code quality and documentation
    4. Identifies likely causes of discrepancies
    5. Provides reproducibility score and recommendations
    """
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        
        # Load system prompt from YAML configuration
        system_prompt = BaseAgent._load_agent_instructions('evaluator_agent')
        
        # Fail fast if configuration is missing
        if not system_prompt:
            raise ValueError(
                "Failed to load system prompt for evaluator_agent from config/agent_instructions.yaml. "
                "Please ensure the configuration file exists and contains the 'evaluator_agent' section."
            )
        
        super().__init__(
            agent_name="evaluator",
            system_prompt=system_prompt,
            config_path=config_path
        )
        
        # Initialize backing tools
        self.comparator = ResultComparator()
        
        # Import file reading tools for autonomous output file discovery
        from scientist.tools.tool_wrappers import ReadFileContents, ListDirectoryFiles, AnalyzePlotSemantics, ExtractTableMetrics
        
        # Register autonomous tools with LLM access for intelligent extraction
        self.register_tool("extract_metrics", None, ExtractMetrics(self.comparator, llm_client=self.model))
        self.register_tool("read_file_contents", None, ReadFileContents())
        self.register_tool("list_directory_files", None, ListDirectoryFiles())
        self.register_tool("analyze_plot_semantics", None, AnalyzePlotSemantics(vision_model_client=self.model))
        self.register_tool("extract_table_metrics", None, ExtractTableMetrics(llm_client=self.model))
        
        # Store references to tools for direct file reading
        self.read_file_tool = ReadFileContents()
        self.list_dir_tool = ListDirectoryFiles()
    
    def _load_image_from_base64(self, image_b64: str) -> Optional[Image.Image]:
        
        try:
            raw = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(raw))
            return img.convert('RGB')
        except Exception:
            return None
    
    def _load_image_from_path(self, path: str) -> Optional[Image.Image]:
        
        try:
            img = Image.open(path)
            return img.convert('RGB')
        except Exception:
            return None
    
    def _compute_ahash_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        
        def ahash(img: Image.Image) -> List[int]:
            g = img.resize((8, 8), Image.Resampling.LANCZOS).convert('L')
            pixels = list(g.getdata())
            avg = sum(pixels) / len(pixels) if pixels else 0
            return [1 if p >= avg else 0 for p in pixels]
        
        h1 = ahash(img1)
        h2 = ahash(img2)
        same = sum(1 for a, b in zip(h1, h2) if a == b)
        return same / 64.0
    
    def _compute_histogram_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        
        h1 = img1.histogram()
        h2 = img2.histogram()
        if not h1 or not h2 or len(h1) != len(h2):
            return 0.0
        dot = 0.0
        n1 = 0.0
        n2 = 0.0
        for a, b in zip(h1, h2):
            dot += float(a) * float(b)
            n1 += float(a) * float(a)
            n2 += float(b) * float(b)
        denom = math.sqrt(n1) * math.sqrt(n2)
        return (dot / denom) if denom > 0 else 0.0
    
    def _compute_psnr(self, img1: Image.Image, img2: Image.Image) -> float:
        
        s1 = img1.resize((128, 128), Image.Resampling.LANCZOS).convert('L')
        s2 = img2.resize((128, 128), Image.Resampling.LANCZOS).convert('L')
        p1 = list(s1.getdata())
        p2 = list(s2.getdata())
        if not p1 or not p2 or len(p1) != len(p2):
            return 0.0
        mse = 0.0
        for a, b in zip(p1, p2):
            d = float(a) - float(b)
            mse += d * d
        mse /= len(p1)
        if mse == 0:
            return 99.0
        return 20.0 * math.log10(255.0 / math.sqrt(mse))
    
    def _compare_images(
        self,
        paper_figures_b64: List[str],
        repo_image_paths: List[str]
    ) -> Dict[str, Any]:
        
        paper_images: List[Image.Image] = []
        for b64s in paper_figures_b64:
            img = self._load_image_from_base64(b64s)
            if img:
                paper_images.append(img)
        
        repo_images: List[tuple[str, Image.Image]] = []
        for p in repo_image_paths:
            img = self._load_image_from_path(p)
            if img:
                repo_images.append((p, img))
        
        if not paper_images or not repo_images:
            return {'comparisons': [], 'visual_score': 0.0}
        
        # Precompute pairwise similarities
        pairs: List[Dict[str, Any]] = []
        for i, pimg in enumerate(paper_images):
            for rpath, rimg in repo_images:
                try:
                    ah = self._compute_ahash_similarity(pimg, rimg)
                    hist = self._compute_histogram_similarity(pimg, rimg)
                    psnr = self._compute_psnr(pimg, rimg)
                    # Combined similarity emphasizes structure and distribution
                    combined = 0.6 * ah + 0.4 * max(0.0, min(1.0, hist))
                    pairs.append({
                        'paper_index': i,
                        'repo_path': rpath,
                        'ahash': ah,
                        'histogram_sim': hist,
                        'psnr': psnr,
                        'combined': combined
                    })
                except Exception:
                    continue
        
        # Greedy one-to-one matching
        pairs.sort(key=lambda x: (x['combined'], x['psnr']), reverse=True)
        used_papers = set()
        used_repo = set()
        matches: List[Dict[str, Any]] = []
        for item in pairs:
            pi = item['paper_index']
            rp = item['repo_path']
            if pi in used_papers or rp in used_repo:
                continue
            # Heuristic threshold: good visual match if combined >= 0.85 or PSNR >= 28
            is_match = (item['combined'] >= 0.85) or (item['psnr'] >= 28.0)
            matches.append({
                'paper_figure_index': pi,
                'repo_image_path': rp,
                'ahash_similarity': item['ahash'],
                'histogram_similarity': item['histogram_sim'],
                'psnr': item['psnr'],
                'combined_similarity': item['combined'],
                'match': bool(is_match)
            })
            used_papers.add(pi)
            used_repo.add(rp)
        
        if not matches:
            return {'comparisons': [], 'visual_score': 0.0}
        
        # Visual score: average combined similarity over matched items that passed threshold
        matched = [m for m in matches if m['match']]
        visual_score = (sum(m['combined_similarity'] for m in matched) / len(matched)) if matched else 0.0
        return {'comparisons': matches, 'visual_score': visual_score}
    
    def _detect_plot_files(self, output_dir: str) -> List[Dict[str, Any]]:

        from pathlib import Path
        
        plot_info = []
        plot_extensions = ['.png', '.jpg', '.jpeg', '.pdf', '.svg', '.eps']
        
        try:
            output_path = Path(output_dir)
            if not output_path.exists():
                return []
            
            # Search for plot files recursively
            for plot_file in output_path.rglob('*'):
                if plot_file.is_file() and plot_file.suffix.lower() in plot_extensions:
                    # Skip hidden files and system files
                    if plot_file.name.startswith('.'):
                        continue
                    
                    try:
                        relative_path = plot_file.relative_to(output_path)
                    except ValueError:
                        relative_path = plot_file
                    
                    plot_info.append({
                        'filename': plot_file.name,
                        'path': str(relative_path),
                        'full_path': str(plot_file),
                        'size_kb': plot_file.stat().st_size / 1024,
                        'extension': plot_file.suffix
                    })
            
            if plot_info:
                self.logger.info(f"✅ Found {len(plot_info)} plot files")
            
            return plot_info
            
        except Exception as e:
            self.logger.error(f"Error detecting plot files: {e}")
            return []
    
    def _extract_metrics_from_output_dir(self, output_dir: str) -> Optional[Dict[str, Any]]:

        import json
        from pathlib import Path
        
        try:
            output_path = Path(output_dir)
            if not output_path.exists():
                self.logger.warning(f"Output directory does not exist: {output_dir}")
                return None
            
            # Search for common output directories
            output_dir_names = ['outputs_all_methods', 'outputs', 'results', 'logs', 'output', 'experiments', 'runs', 'figures', 'plots']
            
            # Search for these directories
            found_dirs = []
            for dir_name in output_dir_names:
                for match in output_path.rglob(dir_name):
                    if match.is_dir():
                        found_dirs.append(match)
            
            # Also check the root directory itself
            found_dirs.append(output_path)
            
            # Search for JSON result files
            result_files = []
            for search_dir in found_dirs:
                for json_file in search_dir.glob('*.json'):
                    if any(name in json_file.name.lower() for name in ['result', 'metric', 'complete', 'summary']):
                        result_files.append(json_file)
            
            # Detect plot files
            plot_files = self._detect_plot_files(output_dir)
            
            metrics = {}
            
            # Extract numerical metrics from JSON files
            if result_files:
                # Prioritize complete_results.json if it exists
                complete_results = [f for f in result_files if 'complete_results' in f.name.lower()]
                if complete_results:
                    result_files = complete_results[:1]  # Use the first complete_results.json
                
                # Read and extract metrics from the first suitable file
                for result_file in result_files[:3]:  # Try up to 3 files
                    try:
                        self.logger.info(f"Reading metrics from: {result_file.relative_to(output_path)}")
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        
                        # Extract metrics from the JSON structure
                        numerical_metrics = self._flatten_metrics(data)
                        
                        if numerical_metrics:
                            metrics.update(numerical_metrics)
                            break
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to read {result_file}: {e}")
                        continue
            
            # Add plot information to metrics
            if plot_files:
                metrics['_plot_files'] = plot_files
                metrics['_has_plots'] = True
                metrics['_plot_count'] = len(plot_files)
                self.logger.info(f"✅ Detected {len(plot_files)} plot files as visual metrics")
            
            if not metrics:
                self.logger.info("No result files or plots found in output directory")
                return None
            
            return metrics if metrics else None
            
        except Exception as e:
            self.logger.error(f"Error extracting metrics from output dir: {e}", exc_info=True)
            return None
    
    def _flatten_metrics(self, data: Any, prefix: str = '', max_depth: int = 5) -> Dict[str, float]:

        metrics = {}
        
        if max_depth == 0:
            return metrics
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}/{key}" if prefix else key
                
                # Skip very deep nesting or non-informative keys
                if '.' in key or len(new_prefix.split('/')) > 6:
                    continue
                
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    # Store numerical values
                    metrics[new_prefix] = float(value)
                elif isinstance(value, dict):
                    # Recursively flatten nested dicts
                    nested = self._flatten_metrics(value, new_prefix, max_depth - 1)
                    metrics.update(nested)
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    # Handle lists of dicts (take first few items)
                    for idx, item in enumerate(value[:3]):
                        nested = self._flatten_metrics(item, f"{new_prefix}[{idx}]", max_depth - 1)
                        metrics.update(nested)
        
        return metrics
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:

        paper_results = task.get('paper_results') or task.get('original_output')
        reproduced_results = task.get('reproduced_results') or task.get('reproduced_output')
        
        if not paper_results or not reproduced_results:
            return {
                'success': False,
                'error': 'Both paper_results/original_output and reproduced_results/reproduced_output are required'
            }
        
        try:
            self.logger.info("🤖 Autonomous agent evaluating reproducibility...")
            
            # Convert to strings if needed
            paper_str = str(paper_results) if not isinstance(paper_results, str) else paper_results
            repro_str = str(reproduced_results) if not isinstance(reproduced_results, str) else reproduced_results
            
            code_notes = task.get('code_quality_notes', 'Not provided')
            doc_notes = task.get('documentation_notes', 'Not provided')
            output_dir = task.get('output_directory', 'Not provided')

            original_metrics = task.get('original_metrics')
            reproduced_metrics = task.get('reproduced_metrics')
            
            # Try to find and read actual output files if output directory is provided
            has_plots = False
            plot_info = []
            if output_dir and output_dir != 'Not provided' and not isinstance(reproduced_metrics, dict):
                self.logger.info(f"🔍 Searching for output files in: {output_dir}")
                reproduced_metrics = self._extract_metrics_from_output_dir(output_dir)
                if reproduced_metrics:
                    # Check if plot files were detected
                    has_plots = reproduced_metrics.get('_has_plots', False)
                    plot_info = reproduced_metrics.get('_plot_files', [])
                    
                    # Remove metadata keys before comparison
                    reproduced_metrics = {k: v for k, v in reproduced_metrics.items() if not k.startswith('_')}
                    
                    if reproduced_metrics:
                        self.logger.info(f"✅ Extracted {len(reproduced_metrics)} numerical metrics from output files")
                    if has_plots:
                        self.logger.info(f"✅ Detected {len(plot_info)} plot-based metrics")
                else:
                    self.logger.warning("⚠️ No metrics found in output files, falling back to text extraction")
            
            # Fallback to LLM extraction if no structured metrics found
            if not isinstance(original_metrics, dict):
                self.logger.info("Using LLM to extract metrics from paper results...")
                original_metrics = self.comparator.extract_metrics(paper_str, llm_client=self.model)
            
            if not isinstance(reproduced_metrics, dict) or not reproduced_metrics:
                self.logger.info("Using LLM to extract metrics from reproduced results text...")
                reproduced_metrics = self.comparator.extract_metrics(repro_str, llm_client=self.model)

            # Compute comparison metrics
            comparisons = self.comparator.compare_metrics(original_metrics, reproduced_metrics, llm_client=self.model)
            overall_score = self.comparator.calculate_reproducibility_score(comparisons)
            comparison_report = self.comparator.generate_comparison_report(comparisons, overall_score)

            # Load task prompt from YAML
            task_prompt_template = BaseAgent._load_task_prompt('evaluator_agent')
            if not task_prompt_template:
                raise ValueError(
                    "Failed to load task prompt for evaluator_agent from config/agent_instructions.yaml. "
                    "Please ensure the configuration file exists and contains the 'task_prompt' field."
                )
            
            # Build plot information summary if plots were detected
            plot_summary = "No plot files detected."
            if has_plots and plot_info:
                plot_summary = f"Detected {len(plot_info)} plot/figure files:\n"
                for plot in plot_info[:10]:  # Show first 10 plots
                    plot_summary += f"  - {plot['filename']} ({plot['size_kb']:.1f} KB)\n"
                if len(plot_info) > 10:
                    plot_summary += f"  ... and {len(plot_info) - 10} more plots"
            
            # Visual comparison (paper figures vs produced plots)
            # Build structured mapping: Figure 1 -> reproduced_plot_1.png
            image_comparisons: List[Dict[str, Any]] = []
            visual_score: float = 0.0
            figure_mapping: List[Dict[str, Any]] = []
            
            try:
                paper_figures_input = task.get('paper_figures') or []
                paper_b64_list: List[str] = []
                paper_fig_metadata: List[Dict[str, Any]] = []
                
                for fig in paper_figures_input:
                    if isinstance(fig, dict):
                        b64v = fig.get('image_base64') or fig.get('b64') or fig.get('image')
                        if isinstance(b64v, str):
                            paper_b64_list.append(b64v)
                            # Store metadata for structured mapping
                            paper_fig_metadata.append({
                                'figure_number': fig.get('figure_number', len(paper_b64_list)),
                                'caption': fig.get('caption', ''),
                                'page_number': fig.get('page_number', 0)
                            })
                
                repo_paths: List[str] = [p['full_path'] for p in (plot_info or []) if isinstance(p, dict) and p.get('full_path')]
                
                if paper_b64_list and repo_paths:
                    vis = self._compare_images(paper_b64_list, repo_paths)
                    image_comparisons = vis.get('comparisons', [])
                    visual_score = float(vis.get('visual_score', 0.0))
                    
                    # Create structured mapping
                    for comp in image_comparisons:
                        paper_idx = comp.get('paper_figure_index', 0)
                        if paper_idx < len(paper_fig_metadata):
                            metadata = paper_fig_metadata[paper_idx]
                            figure_mapping.append({
                                'paper_figure': f"Figure {metadata['figure_number']}",
                                'paper_caption': metadata['caption'],
                                'paper_page': metadata['page_number'],
                                'reproduced_file': comp.get('repo_image_path', ''),
                                'similarity_score': comp.get('combined_similarity', 0.0),
                                'match': comp.get('match', False)
                            })
            except Exception:
                # Non-fatal
                pass
            
            # Fill in template variables
            agent_task = BaseAgent._render_template(
                task_prompt_template,
                {
                    "paper_str": paper_str[:2000],
                    "repro_str": repro_str[:2000],
                    "output_dir": str(output_dir),
                    "code_notes": code_notes,
                    "doc_notes": doc_notes,
                }
            )
            
            # Append plot information to the task
            agent_task += f"\n\n**Plot Files Detected:**\n{plot_summary}"
            
            # Agent works autonomously
            self.logger.info("Agent is now analyzing and comparing results...")
            agent_response = self.call_llm(agent_task)
            
            self.logger.info("✅ Agent completed autonomous evaluation")
            
            # Try to parse minimal JSON from agent response for recommendations/causes
            import json as _json
            import re as _re
            import ast as _ast
            analysis_text: str = agent_response
            likely_causes: List[str] = []
            recommendations: List[str] = []
            try:
                json_match = _re.search(r'\{[\s\S]*\}', agent_response)
                if json_match:
                    parsed_response = None
                    json_blob = json_match.group(0)
                    try:
                        parsed_response = _json.loads(json_blob)
                    except _json.JSONDecodeError:
                        try:
                            parsed_response = _ast.literal_eval(json_blob)
                        except (ValueError, SyntaxError):
                            parsed_response = None
                    if isinstance(parsed_response, dict):
                        raw_analysis = parsed_response.get('analysis')
                        if isinstance(raw_analysis, (dict, list)):
                            analysis_text = _json.dumps(raw_analysis, ensure_ascii=False)
                        elif raw_analysis is not None:
                            analysis_text = str(raw_analysis)
                        if isinstance(parsed_response.get('likely_causes'), list):
                            likely_causes = [str(x) for x in parsed_response.get('likely_causes', [])]
                        if isinstance(parsed_response.get('recommendations'), list):
                            recommendations = [str(x) for x in parsed_response.get('recommendations', [])]
                    elif isinstance(parsed_response, list) and parsed_response:
                        first_item = parsed_response[0]
                        if isinstance(first_item, dict):
                            raw_analysis = first_item.get('analysis')
                            if raw_analysis is not None:
                                analysis_text = str(raw_analysis)
                            if isinstance(first_item.get('likely_causes'), list):
                                likely_causes = [str(x) for x in first_item.get('likely_causes', [])]
                            if isinstance(first_item.get('recommendations'), list):
                                recommendations = [str(x) for x in first_item.get('recommendations', [])]
            except Exception:
                # Non-fatal; keep text-only analysis
                pass

            # Normalize output contract expected by orchestrator
            result = {
                'success': True,
                'final_reproducibility_score': comparison_report.get('overall_score', overall_score),
                'metrics_matched': comparison_report.get('metrics_matched', 0),
                'total_metrics': comparison_report.get('total_metrics', 0),
                'comparison_details': comparison_report,
                'original_metrics': original_metrics,
                'reproduced_metrics': reproduced_metrics,
                'has_plot_metrics': has_plots,
                'plot_files': plot_info if has_plots else [],
                'plot_count': len(plot_info) if has_plots else 0,
                'image_comparisons': image_comparisons,
                'visual_score': visual_score,
                'figure_mapping': figure_mapping,  # Structured mapping: Figure X -> reproduced file
                'analysis': analysis_text,
                'likely_causes': likely_causes,
                'recommendations': recommendations,
                'execution_type': 'autonomous',
                'note': 'Deterministic scoring + semantic visual analysis via vision model'
            }
            
            self.log_execution("autonomous_evaluation", result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in autonomous execution: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }


def evaluate_reproducibility(paper_results: Any, reproduced_results: Any) -> Dict[str, Any]:
    
    agent = EvaluatorAgent()
    result = agent.execute({
        'paper_results': paper_results,
        'reproduced_results': reproduced_results
    })
    
    return result
