
from typing import Dict, Any, Optional, List
import json
import re

from scientist.agents.base_agent import BaseAgent
from scientist.tools.result_comparator import LLMResultComparator, ResultComparator
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
        
        # Initialize LLM-driven comparator (no hardcoded patterns)
        self.comparator = LLMResultComparator(llm_client=self.model)
        
        # Legacy comparator for backward compatibility
        self.legacy_comparator = ResultComparator()
        
        # Import file reading tools for autonomous output file discovery
        from scientist.tools.tool_wrappers import ReadFileContents, ListDirectoryFiles, AnalyzePlotSemantics, ExtractTableMetrics
        
        # Register autonomous tools with LLM access for intelligent extraction
        self.register_tool("extract_metrics", None, ExtractMetrics(self.legacy_comparator, llm_client=self.model))
        self.register_tool("read_file_contents", None, ReadFileContents())
        self.register_tool("list_directory_files", None, ListDirectoryFiles())
        self.register_tool("analyze_plot_semantics", None, AnalyzePlotSemantics(vision_model_client=self.model))
        self.register_tool("extract_table_metrics", None, ExtractTableMetrics(llm_client=self.model))
        
        # Store references to tools for direct file reading
        self.read_file_tool = ReadFileContents()
        self.list_dir_tool = ListDirectoryFiles()
    
    # OLD REDUNDANT METHODS REMOVED
    # Pixel-based image comparison is replaced by vision model semantic analysis
    # Plot detection is now done by MaterialOrganizer
    # Metric extraction from scattered files is now done by MaterialOrganizer
    
    @staticmethod
    def _flatten_metric_map(metrics: Any, prefix: str = "") -> Dict[str, float]:
        """
        Convert nested metric structures into a flat dict of floats.
        Non-numeric values are skipped.
        """
        flattened: Dict[str, float] = {}
        
        if not isinstance(metrics, dict):
            return flattened
        
        for raw_key, raw_value in metrics.items():
            if raw_key is None:
                continue
            
            key = str(raw_key).strip()
            if not key:
                continue
            
            key_clean = re.sub(r'[^A-Za-z0-9@]+', '_', key).strip('_')
            key_path = f"{prefix}{key_clean}" if not prefix else f"{prefix}_{key_clean}"
            
            if isinstance(raw_value, dict):
                flattened.update(EvaluatorAgent._flatten_metric_map(raw_value, key_path))
                continue
            
            value = EvaluatorAgent._coerce_to_float(raw_value)
            if value is None:
                continue
            
            flattened[key_path] = value
        
        return flattened
    
    @staticmethod
    def _coerce_to_float(value: Any) -> Optional[float]:
        """Attempt to convert a value to float, handling numeric strings/percentages."""
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            
            is_percent = '%' in text
            cleaned = text.replace('%', '').replace(',', '')
            
            match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', cleaned)
            if not match:
                return None
            
            try:
                number = float(match.group(0))
                if is_percent:
                    number /= 100.0
                return number
            except ValueError:
                return None
        
        return None
    
    def _extract_metrics_from_figures(self, paper_figures: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract numerical metrics from paper figures using vision model.
        
        Args:
            paper_figures: List of figure dicts with 'image_base64', 'caption', 'figure_number', 'page_number'
            
        Returns:
            Dict of extracted metrics with figure context in key names
        """
        all_metrics = {}
        
        try:
            for fig_idx, figure in enumerate(paper_figures):
                try:
                    image_b64 = figure.get('image_base64', '')
                    caption = figure.get('caption', '')
                    fig_num = figure.get('figure_number', str(fig_idx))
                    
                    if not image_b64:
                        continue
                    
                    self.logger.info(f"  → Extracting metrics from Figure {fig_num}: {caption[:60]}...")
                    
                    # Use vision model to extract metrics from figure
                    prompt = """Analyze this figure from a research paper and extract ALL numerical metrics shown.
                    
Look for:
- Performance metrics (accuracy, precision, recall, F1, MRR, NDCG, etc.)
- Recall@k values (e.g., Recall@1, Recall@10, Recall@50)
- Tables embedded in figures
- Bar charts, line plots with specific values
- Any numbers reported in the figure

Return ONLY a JSON object with metric names as keys and numerical values as floats.
Example: {"Recall@10": 0.85, "MRR": 0.72, "F1": 0.68}

If no metrics are found, return an empty JSON object: {}
"""
                    
                    # Call vision model
                    if self.model and hasattr(self.model, '__call__'):
                        messages = [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                            ]
                        }]
                        
                        response = self.model(messages)
                        response_text = response if isinstance(response, str) else str(response)
                        
                        # Extract JSON from response
                        # Look for JSON object
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                        if json_match:
                            extracted = json.loads(json_match.group(0))
                            
                            # Prefix metrics with figure number to avoid conflicts
                            for metric_name, value in extracted.items():
                                try:
                                    # Add figure context to metric name
                                    prefixed_key = f"Figure_{fig_num}_{metric_name}"
                                    all_metrics[prefixed_key] = float(value)
                                except (ValueError, TypeError):
                                    continue
                            
                            if extracted:
                                self.logger.info(f"    ✓ Found {len(extracted)} metrics in Figure {fig_num}")
                        else:
                            self.logger.debug(f"    No metrics found in Figure {fig_num}")
                    
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Error extracting from Figure {fig_num}: {e}")
                    continue
            
            return all_metrics
            
        except Exception as e:
            self.logger.error(f"Error extracting metrics from figures: {e}", exc_info=True)
            return {}
    
    def _extract_metrics_from_tables(self, paper_tables: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract numerical metrics from paper tables using text parsing.
        
        Args:
            paper_tables: List of table dicts with 'content', 'caption', 'table_number', 'page_number'
            
        Returns:
            Dict of extracted metrics with table context in key names
        """
        all_metrics = {}
        
        try:
            for tbl_idx, table in enumerate(paper_tables):
                try:
                    content = table.get('content', '')
                    caption = table.get('caption', '')
                    tbl_num = table.get('table_number', str(tbl_idx))
                    
                    if not content:
                        continue
                    
                    self.logger.info(f"  → Extracting metrics from Table {tbl_num}: {caption[:60]}...")
                    
                    # Use LLM to intelligently extract metrics from table text
                    prompt = f"""Analyze this table from a research paper and extract ALL numerical metrics shown.

Table Caption: {caption}

Table Content:
{content[:2000]}

Extract all performance metrics, scores, and measurements. Look for:
- Accuracy, Precision, Recall, F1, MRR, NDCG metrics
- Recall@k values (e.g., Recall@1, Recall@10, Recall@50)
- Percentages, ratios, scores
- Any numerical performance measurements

Return ONLY a JSON object with metric names as keys and numerical values as floats.
Example: {{"Recall@10": 0.85, "MRR": 0.72, "F1_Score": 0.68}}

If no metrics are found, return an empty JSON object: {{}}
"""
                    
                    # Call LLM to extract
                    if self.model and hasattr(self.model, '__call__'):
                        messages = [{"role": "user", "content": prompt}]
                        response = self.model(messages)
                        response_text = response if isinstance(response, str) else str(response)
                        
                        # Extract JSON from response
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                        if json_match:
                            extracted = json.loads(json_match.group(0))
                            
                            # Prefix metrics with table number to avoid conflicts
                            for metric_name, value in extracted.items():
                                try:
                                    # Add table context to metric name
                                    prefixed_key = f"Table_{tbl_num}_{metric_name}"
                                    all_metrics[prefixed_key] = float(value)
                                except (ValueError, TypeError):
                                    continue
                            
                            if extracted:
                                self.logger.info(f"    ✓ Found {len(extracted)} metrics in Table {tbl_num}")
                        else:
                            self.logger.debug(f"    No metrics found in Table {tbl_num}")
                    
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Error extracting from Table {tbl_num}: {e}")
                    continue
            
            return all_metrics
            
        except Exception as e:
            self.logger.error(f"Error extracting metrics from tables: {e}", exc_info=True)
            return {}
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:

        paper_results = task.get('paper_results') or task.get('original_output')
        reproduced_results = task.get('reproduced_results') or task.get('reproduced_output')
        
        if not paper_results or not reproduced_results:
            return {
                'success': False,
                'error': 'Both paper_results/original_output and reproduced_results/reproduced_output are required'
            }
        
        try:
            self.logger.info("🤖 LLM-driven comparison (no hardcoded patterns)...")
            
            # Convert to strings if needed
            paper_str = str(paper_results) if not isinstance(paper_results, str) else paper_results
            repro_str = str(reproduced_results) if not isinstance(reproduced_results, str) else reproduced_results
            
            code_notes = task.get('code_quality_notes', 'Not provided')
            doc_notes = task.get('documentation_notes', 'Not provided')
            output_dir = task.get('output_directory', 'Not provided')

            original_metrics = task.get('original_metrics')
            reproduced_metrics = task.get('reproduced_metrics')
            
            # Use organized reproduced results directory instead of searching
            has_plots = False
            plot_info = []
            reproduced_results_dir_path = task.get('reproduced_results_dir')
            
            if reproduced_results_dir_path and reproduced_results_dir_path != 'Not provided':
                self.logger.info(f"🔍 Reading organized reproduced results from: {reproduced_results_dir_path}")
                # Read metrics from organized folder
                from pathlib import Path
                import json
                
                try:
                    results_path = Path(reproduced_results_dir_path)
                    
                    # Read plot files from organized plots folder
                    plots_dir = results_path / "plots"
                    if plots_dir.exists():
                        plot_info = [
                            {
                                'filename': p.name,
                                'full_path': str(p),
                                'size_kb': p.stat().st_size / 1024
                            }
                            for p in plots_dir.iterdir() if p.is_file()
                        ]
                        has_plots = len(plot_info) > 0
                        if has_plots:
                            self.logger.info(f"✅ Found {len(plot_info)} organized plots")
                    
                    # Read metrics from organized data folder
                    data_dir = results_path / "data"
                    if data_dir.exists() and not isinstance(reproduced_metrics, dict):
                        for json_file in data_dir.glob('*.json'):
                            try:
                                with open(json_file, 'r') as f:
                                    data = json.load(f)
                                    if isinstance(data, dict):
                                        reproduced_metrics = data
                                        self.logger.info(f"✅ Read metrics from: {json_file.name}")
                                        break
                            except Exception as e:
                                self.logger.warning(f"Could not read {json_file.name}: {e}")
                                continue
                
                except Exception as e:
                    self.logger.warning(f"Error reading organized results: {e}")
            
            # LOAD SAVED PAPER METRICS (extracted during paper parsing, not re-extracted here!)
            paper_extracted_metrics = task.get('paper_extracted_metrics', {})
            
            if paper_extracted_metrics:
                self.logger.info(f"📊 Loading {len(paper_extracted_metrics)} SAVED paper metrics (no re-extraction needed)")
            else:
                self.logger.warning("⚠️ No saved paper metrics found - paper parser may have failed to extract them")
            
            # Build structured paper data for LLM comparison
            paper_data = {
                'experimental_results': paper_str,
                'tables': task.get('paper_tables', []),
                'figures': task.get('paper_figures', []),
                'evaluation_metrics': task.get('evaluation_metrics', []),
                'pre_extracted_metrics': paper_extracted_metrics  # Use saved metrics
            }
            
            # Build structured reproduced data
            reproduced_data = {
                'stdout': repro_str,
                'metrics': reproduced_metrics if isinstance(reproduced_metrics, dict) else {},
                'output_files': [],
                'plots': plot_info if has_plots else []
            }
            
            # Output files are now organized in reproduced_results_dir
            # No need to search scattered files - MaterialOrganizer already organized them
            
            # Use LLM-driven comparison (uses saved paper metrics, no re-extraction!)
            self.logger.info("🤖 Comparing SAVED paper metrics with reproduced results...")
            comparison_result = self.comparator.compare_results(paper_data, reproduced_data)
            
            # Extract results
            overall_score = comparison_result.reproducibility_score
            original_metrics = comparison_result.paper_metrics
            reproduced_metrics = comparison_result.reproduced_metrics
            comparisons = comparison_result.metric_comparisons
            
            # Build comparison report for backward compatibility
            comparison_report = {
                'overall_score': overall_score,
                'total_metrics': comparison_result.total_metrics_compared,
                'metrics_matched': comparison_result.metrics_matched,
                'comparisons': comparisons,
                'analysis': comparison_result.analysis,
                'likely_causes': comparison_result.likely_causes,
                'recommendations': comparison_result.recommendations
            }

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
            
            # Visual comparison will be done by LLM using organized folders
            # LLM will use analyze_plot_semantics tool to compare figures from both folders
            image_comparisons: List[Dict[str, Any]] = []
            visual_score: float = 0.0
            figure_mapping: List[Dict[str, Any]] = []
            
            # Build simple mapping for LLM to know what's available
            paper_figures_input = task.get('paper_figures') or []
            for fig in paper_figures_input:
                if isinstance(fig, dict):
                    figure_mapping.append({
                        'paper_figure': f"Figure {fig.get('figure_number', '?')}",
                        'paper_caption': fig.get('caption', ''),
                        'available_for_comparison': True
                    })
            
            # Note: Actual comparison will be done by LLM using the organized folders
            # and the analyze_plot_semantics tool
            
            # Get organized material paths
            paper_materials_dir = task.get('paper_materials_dir', 'Not provided')
            reproduced_results_dir = task.get('reproduced_results_dir', 'Not provided')
            comparison_manifest = task.get('comparison_manifest', 'Not provided')
            
            # Fill in template variables
            agent_task = BaseAgent._render_template(
                task_prompt_template,
                {
                    "paper_str": paper_str[:2000],
                    "repro_str": repro_str[:2000],
                    "output_dir": str(output_dir),
                    "code_notes": code_notes,
                    "doc_notes": doc_notes,
                    "paper_materials_dir": str(paper_materials_dir),
                    "reproduced_results_dir": str(reproduced_results_dir),
                    "comparison_manifest": str(comparison_manifest),
                }
            )
            
            # Append plot information to the task
            agent_task += f"\n\n**Plot Files Detected:**\n{plot_summary}"
            
            # Agent works autonomously
            self.logger.info("Agent is now analyzing and comparing results...")
            agent_response = self.call_llm(agent_task)
            
            self.logger.info("✅ Agent completed autonomous evaluation")
            
            # Use agent's final answer if it's a dict with evaluation results
            if isinstance(agent_response, dict):
                # Agent provided structured evaluation - use it!
                self.logger.info("📊 Using agent's autonomous evaluation results")
                
                # Extract agent's scores and analysis
                agent_score = agent_response.get('score', overall_score)
                agent_visual_score = agent_response.get('visual_score', visual_score)
                agent_explanation = agent_response.get('explanation', comparison_result.analysis)
                agent_causes = agent_response.get('likely_causes', comparison_result.likely_causes)
                agent_recommendations = agent_response.get('recommendations', comparison_result.recommendations)
                
                # Use agent's metrics if provided, otherwise fall back to comparison_result
                agent_paper_metrics = agent_response.get('paper_metrics', original_metrics)
                agent_reproduced_metrics = agent_response.get('reproduced_metrics', reproduced_metrics)
                agent_comparisons = agent_response.get('comparisons', comparisons)
                agent_image_comparisons = agent_response.get('image_comparisons', image_comparisons)
                
                # Use agent's values
                final_score = agent_score
                final_visual_score = agent_visual_score
                analysis_text = agent_explanation
                likely_causes = agent_causes
                recommendations = agent_recommendations
                original_metrics = agent_paper_metrics
                reproduced_metrics = agent_reproduced_metrics
                comparisons = agent_comparisons
                if agent_image_comparisons:
                    image_comparisons = agent_image_comparisons
                
                # Recalculate metrics_matched from agent's comparisons
                if agent_comparisons and isinstance(agent_comparisons, list):
                    metrics_matched = 0
                    total_metrics = 0
                    for comparison in agent_comparisons:
                        if not isinstance(comparison, dict):
                            continue
                        total_metrics += 1
                        diff = comparison.get('difference')
                        if diff is None:
                            original_value = comparison.get('original')
                            reproduced_value = comparison.get('reproduced')
                            if (
                                isinstance(original_value, (int, float))
                                and isinstance(reproduced_value, (int, float))
                            ):
                                diff = reproduced_value - original_value
                        if isinstance(diff, str):
                            try:
                                diff = float(diff)
                            except (TypeError, ValueError):
                                diff = None
                        if isinstance(diff, (int, float)) and abs(diff) < 0.01:
                            metrics_matched += 1
                else:
                    metrics_matched = comparison_report.get('metrics_matched', 0)
                    total_metrics = comparison_report.get('total_metrics', 0)
                
            else:
                # Agent didn't return structured evaluation - use LLM comparison results
                self.logger.warning("⚠️  Agent response not structured, using LLM comparison results")
                final_score = overall_score
                final_visual_score = visual_score
                analysis_text = comparison_result.analysis
                likely_causes = comparison_result.likely_causes
                recommendations = comparison_result.recommendations
                metrics_matched = comparison_report.get('metrics_matched', 0)
                total_metrics = comparison_report.get('total_metrics', 0)

            # Normalize metric structures for downstream reporting
            original_metrics = self._flatten_metric_map(original_metrics)
            reproduced_metrics = self._flatten_metric_map(reproduced_metrics)

            if total_metrics == 0:
                final_score = None

            # Normalize output contract expected by orchestrator
            result = {
                'success': True,
                'final_reproducibility_score': final_score,
                'metrics_matched': metrics_matched,
                'total_metrics': total_metrics,
                'comparison_details': comparison_report,
                'original_metrics': original_metrics,
                'reproduced_metrics': reproduced_metrics,
                'has_plot_metrics': has_plots,
                'plot_files': plot_info if has_plots else [],
                'plot_count': len(plot_info) if has_plots else 0,
                'image_comparisons': image_comparisons,
                'visual_score': final_visual_score,
                'figure_mapping': figure_mapping,  # Structured mapping: Figure X -> reproduced file
                'analysis': analysis_text,
                'likely_causes': likely_causes,
                'recommendations': recommendations,
                'execution_type': 'autonomous',
                'note': 'Deterministic scoring + semantic visual analysis via vision model + Agent autonomous evaluation'
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
