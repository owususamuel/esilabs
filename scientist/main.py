
import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime

from scientist.agents.paper_parser_agent import PaperParserAgent
from scientist.agents.repo_finder_agent import RepoFinderAgent
from scientist.agents.experiment_runner_agent import ExperimentRunnerAgent
from scientist.agents.evaluator_agent import EvaluatorAgent
from scientist.utils.logging_config import setup_logging
from scientist.utils.data_models import PipelineRun, PaperData
from scientist.utils.interactive import InteractiveInputHandler
from scientist.utils.report_generator import ReportGenerator


class ReproducibilityOrchestrator:
    """
    Orchestrates the entire AI Reproducibility Evaluator pipeline.
    
    Pipeline stages:
    1. Parse paper (extract metadata and requirements)
    2. Find repository (locate implementation)
    3. Run experiments (execute with same config)
    4. Evaluate results (compare and assess reproducibility)
    """
    
    def __init__(self):
        
        self.logger = setup_logging()
        self.logger.info("Initializing AI Reproducibility Orchestrator")
        
        # Initialize agents
        self.paper_parser = PaperParserAgent()
        self.repo_finder = RepoFinderAgent()
        self.experiment_runner = ExperimentRunnerAgent()
        self.evaluator = EvaluatorAgent()
        
        # Initialize interactive handler for manual input fallback
        self.interactive_handler = InteractiveInputHandler()
        
        # Initialize report generator
        self.report_generator = ReportGenerator()
        
        self.current_run = None
    
    def run_pipeline(
        self,
        pdf_path: str,
        original_results_output: Optional[str] = None,
        experiment_config: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None
        ) -> Dict[str, Any]:

        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info(f"Starting reproducibility pipeline (run_id: {run_id})")
        self.logger.info(f"Input PDF: {pdf_path}")
        
        # Initialize pipeline run
        self.current_run = PipelineRun(
            run_id=run_id,
            paper_id="",
            parsing_status="pending",
            repo_finding_status="pending",
            experiment_status="pending",
            evaluation_status="pending"
        )
        
        try:
            # Stage 1: Parse Paper
            self.logger.info("\n" + "="*50)
            self.logger.info("STAGE 1: Parsing Research Paper")
            self.logger.info("="*50)
            
            parsing_result = self._stage_parse_paper(pdf_path)
            self.current_run.parsing_status = "completed" if parsing_result['success'] else "failed"
            
            if not parsing_result['success']:
                self.logger.error(f"Paper parsing failed: {parsing_result['error']}")
                self.current_run.error_message = parsing_result['error']
                return self._finalize_pipeline(failed=True)
            
            paper_data_dict = parsing_result['paper_data']
            # Convert dictionary back to PaperData object for proper type handling
            paper_data = PaperData(**paper_data_dict)
            self.current_run.parsed_paper = paper_data
            self.current_run.paper_id = paper_data.title
            
            # Stage 2: Find Repository
            self.logger.info("\n" + "="*50)
            self.logger.info("STAGE 2: Finding Repository")
            self.logger.info("="*50)
            
            repo_url = None
            repo_path = None
            github_url_from_paper = paper_data_dict.get('github_url', '')
            
            is_valid_url = (
                github_url_from_paper and 
                'github.com' in github_url_from_paper.lower() and
                github_url_from_paper.startswith(('http://', 'https://'))
            )
            
            if is_valid_url:
                self.logger.info(f"✓ GitHub URL found in paper: {github_url_from_paper}")
                repo_url = github_url_from_paper
                self.current_run.found_repo_url = repo_url
                self.current_run.repo_finding_status = "completed_from_paper"
            else:
                # Step 2: No URL in paper - show interactive menu
                if github_url_from_paper:
                    self.logger.warning(f"Repository mentioned but no valid URL found: '{github_url_from_paper}'")
                else:
                    self.logger.warning("No GitHub repository link found in the paper")
                repo_url, repo_path = self._handle_interactive_repo_discovery(
                    paper_data.title,
                    paper_data.authors,
                    paper_data
                )
                
                if not repo_url and not repo_path:
                    # User quit or chose to generate code
                    return self._finalize_pipeline(failed=True)
                
                self.current_run.found_repo_url = repo_url or "manual_upload"
                self.current_run.repo_finding_status = "completed_interactive"
            
            # Stage 3: Run Experiments
            self.logger.info("\n" + "="*50)
            self.logger.info("STAGE 3: Running Experiments")
            self.logger.info("="*50)
            
            exp_result = self._stage_run_experiments(
                repo_url,
                paper_data,
                experiment_config,
                repo_path  # Pass manual path if set
            )
            self.current_run.experiment_status = "completed" if exp_result['success'] else "failed"
            
            if not exp_result['success']:
                exp_error = (
                    exp_result.get('error')
                    or exp_result.get('error_message')
                    or "Unknown experiment error"
                )
                self.logger.error(f"Experiment execution failed: {exp_error}")
                self.current_run.error_message = exp_error
                return self._finalize_pipeline(failed=True)
            
            # Stage 4: Evaluate Results
            self.logger.info("\n" + "="*50)
            self.logger.info("STAGE 4: Evaluating Reproducibility")
            self.logger.info("="*50)
            
            eval_result = self._stage_evaluate_results(
                exp_result,
                paper_data,
                original_results_output,
                parsing_result.get('figures')
            )
            self.current_run.evaluation_status = "completed" if eval_result['success'] else "failed"
            self.current_run.evaluation = eval_result
            
            # Success!
            return self._finalize_pipeline(failed=False)
        
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            self.current_run.error_message = str(e)
            return self._finalize_pipeline(failed=True)
    
    def _stage_parse_paper(self, pdf_path: str) -> Dict[str, Any]:
        """Execute paper parsing stage."""
        
        try:
            result = self.paper_parser.execute({'pdf_path': pdf_path})
            
            if result['success']:
                self.logger.info(f"Paper parsed successfully: {result['paper_data']['title']}")
                self.logger.info(f"  Authors: {result['paper_data']['authors']}")
                self.logger.info(f"  Sections found: {result['sections_found']}")
                if result.get('github_urls_found'):
                    self.logger.info(f"  GitHub URLs in paper: {result['github_urls_found']}")
            
            return result
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _stage_find_repository(
        self,
        paper_title: str,
        authors: list
    ) -> Dict[str, Any]:
        """Execute repository finding stage."""
        
        try:
            result = self.repo_finder.execute({
                'paper_title': paper_title,
                'authors': authors
            })
            
            # Check both 'best_repository' and 'selected_repository' keys
            repo = result.get('best_repository') or result.get('selected_repository')
            
            if result['success'] and repo and repo.get('url'):
                self.logger.info(f"Repository found: {repo['url']}")
                if repo.get('stars'):
                    self.logger.info(f"  Stars: {repo['stars']}")
                if repo.get('language'):
                    self.logger.info(f"  Language: {repo['language']}")
                if repo.get('is_official'):
                    self.logger.info("OFFICIAL REPOSITORY")
            
            return result
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _handle_interactive_repo_discovery(
        self,
        paper_title: str,
        authors: list,
        paper_data: Any
    ) -> tuple[Optional[str], Optional[str]]:
        
        # Step 1: Show interactive menu (ZIP, URL, Agent Search)
        repo_info = self.interactive_handler.prompt_for_repository()
        
        if not repo_info:
            self.logger.warning("User cancelled repository input")
            return None, None
        
        # Step 2: Handle user choice
        if repo_info['type'] == 'agent_search':
            # User chose option 3: Let agent search GitHub
            self.logger.info("User selected: Let agent search GitHub")
            
            repo_result = self._stage_find_repository(paper_title, authors)
            
            # Check if agent found anything
            repo_data = repo_result.get('best_repository') or repo_result.get('selected_repository')
            
            if repo_result['success'] and repo_data and repo_data.get('url'):
                # Agent found a repository!
                repo_url = repo_data['url']
                self.logger.info(f"Agent found repository: {repo_url}")
                return repo_url, None
            else:
                # Agent search returned empty - show second menu
                self.logger.warning("Agent search returned no results")
                action = self.interactive_handler.prompt_after_failed_search()
                
                if action == 'generate':
                    # User chose to generate code
                    self.logger.info("Proceeding with code generation from methodology...")
                    return self._handle_code_generation(paper_data)
                else:
                    # User chose to quit
                    self.logger.info("User chose to quit")
                    return None, None
        else:
            # User chose option 1 (ZIP) or 2 (URL) - prepare repository
            repo_path = self.interactive_handler.prepare_repository(repo_info)
            
            if not repo_path:
                self.logger.error("Failed to prepare repository")
                return None, None
            
            # Return both URL (if available) and local path
            repo_url = repo_info.get('url', None)
            
            self.logger.info(f"Repository prepared at: {repo_path}")
            if repo_url:
                self.logger.info(f"  Source URL: {repo_url}")
            
            return repo_url, repo_path
    
    def _handle_code_generation(self, paper_data: Any) -> tuple[Optional[str], Optional[str]]:
        """
        This is a placeholder for the code generation agent.

        """
        self.current_run.error_message = "Code generation not yet implemented"
        return None, None
    
    def _stage_run_experiments(
        self,
        repo_url: str,
        paper_data: PaperData,
        experiment_config: Optional[Dict[str, Any]],
        repo_path: Optional[str] = None
    ) -> Dict[str, Any]:
        
        try:
            # Build experiment config
            config = experiment_config or {}
            
            config['repo_url'] = repo_url
            config['repo_path'] = repo_path
            
            result = self.experiment_runner.execute(
                config,
                interactive_handler=self.interactive_handler
            )
            
            if result['success']:
                self.logger.info("✅ Experiment completed successfully")
                if result.get('duration_seconds'):
                    self.logger.info(f"  Duration: {result['duration_seconds']:.1f}s")
                if result.get('exit_code') is not None:
                    self.logger.info(f"  Exit code: {result['exit_code']}")
                if result.get('metrics'):
                    self.logger.info(f"  Metrics extracted: {list(result['metrics'].keys())}")
            else:
                fail_reason = result.get('error') or result.get('error_message') or result.get('stderr') or "Unknown failure"
                self.logger.error(f"Experiment failed: {fail_reason}")
            
            return result
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _stage_evaluate_results(
        self,
        experiment_result: Dict[str, Any],
        paper_data: PaperData,
        original_output: Optional[str],
        paper_figures: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Execute evaluation stage."""
        
        try:
            # Build an LLM-friendly original results text from the parsed paper
            original_text = original_output or "\n\n".join([
                f"Title: {paper_data.title}",
                f"Abstract: {paper_data.abstract or ''}",
                f"Methodology: {paper_data.methodology or ''}",
                f"Experimental Results: {paper_data.experimental_results or 'Not extracted'}",
                f"Evaluation Metrics: {', '.join(paper_data.evaluation_metrics) if paper_data.evaluation_metrics else 'Not specified'}",
                f"Key Findings: {paper_data.key_findings or 'Not extracted'}"
            ])
            
            # Build evaluation config with both original and reproduced data
            # Prefer passing rich textual outputs so the Evaluator can use LLM tools
            reproduced_stdout = experiment_result.get('stdout', '')
            reproduced_agent_response = experiment_result.get('agent_response', '')
            reproduced_structured = experiment_result.get('metrics_extracted', '')
            reproduced_text = "\n\n".join([
                str(reproduced_stdout or ''),
                str(reproduced_agent_response or ''),
                str(reproduced_structured or '')
            ]).strip()
            
            # Get output directory from experiment result so evaluator can search for files
            output_directory = experiment_result.get('repo_path', 'Not provided')
            
            eval_config = {
                'paper_id': paper_data.title,
                'reproduced_output': reproduced_text,
                'original_output': original_text,
                'output_directory': output_directory,
                'paper_figures': paper_figures or []
            }
            
            result = self.evaluator.execute(eval_config)
            
            if result['success']:
                score = result['final_reproducibility_score']
                self.logger.info(f"Evaluation complete")
                self.logger.info(f"  Reproducibility Score: {score:.2%}")
                self.logger.info(f"  Metrics matched: {result['metrics_matched']}/{result['total_metrics']}")
                
                if result.get('issues_found'):
                    self.logger.warning(f"  Issues found: {len(result['issues_found'])}")
                    for issue in result['issues_found'][:3]:
                        self.logger.warning(f"    - {issue}")
                
                if result.get('recommendations'):
                    self.logger.info(f"  Recommendations: {len(result['recommendations'])}")
            
            return result
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _finalize_pipeline(self, failed: bool = False) -> Dict[str, Any]:
        """Finalize pipeline and generate report."""
        
        self.logger.info("\n" + "="*50)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("="*50)
        
        if failed:
            self.logger.error("Pipeline ended with errors")
            status = "failed"
        else:
            self.logger.info("Pipeline completed successfully")
            status = "completed"
        
        return {
            'success': not failed,
            'status': status,
            'run_id': self.current_run.run_id if self.current_run else None,
            'pipeline': self.current_run.model_dump() if self.current_run else None
        }
    
    def save_report(
        self, 
        output_path: str = None,
        generate_visualizations: bool = True,
        generate_full_report: bool = True
    ) -> Dict[str, str]:
        """
        Save pipeline results with comprehensive reporting.
        
        Args:
            output_path: Optional custom output path for JSON report
            generate_visualizations: Whether to create charts/graphs
            generate_full_report: Whether to generate full report package
            
        Returns:
            Dictionary of generated file paths
        """
        if not output_path:
            output_path = f"./data/outputs/report_{self.current_run.run_id}.json"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            'run_id': self.current_run.run_id,
            'timestamp': datetime.now().isoformat(),
            'pipeline': self.current_run.model_dump()
        }
        
        # Save basic JSON report
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"JSON report saved to: {output_path}")
        
        generated_files = {'json': output_path}
        
        # Generate comprehensive report package if requested
        if generate_full_report:
            try:
                self.logger.info("Generating comprehensive report package...")
                report_files = self.report_generator.generate_full_report(
                    run_data=report_data,
                    run_id=self.current_run.run_id,
                    create_visualizations=generate_visualizations
                )
                generated_files.update(report_files)
                
                self.logger.info(f"✓ Report package complete: {len(generated_files)} files generated")
                self.logger.info(f"  → View dashboard: {report_files.get('dashboard', 'N/A')}")
                
            except Exception as e:
                self.logger.error(f"Error generating full report: {e}", exc_info=True)
                self.logger.info("Continuing with basic JSON report...")
        
        return generated_files


def run_reproducibility_pipeline(
    pdf_path: str,
    original_results: Optional[str] = None,
    output_report: bool = True,
    generate_visualizations: bool = True
) -> Dict[str, Any]:
    """
    Run the full reproducibility pipeline.
    
    Args:
        pdf_path: Path to research paper PDF
        original_results: Optional original results text
        output_report: Whether to save reports
        generate_visualizations: Whether to create charts/graphs
        
    Returns:
        Pipeline execution results with generated file paths
    """
    orchestrator = ReproducibilityOrchestrator()
    result = orchestrator.run_pipeline(pdf_path, original_results)
    
    if output_report and result.get('success'):
        report_files = orchestrator.save_report(
            generate_visualizations=generate_visualizations,
            generate_full_report=True
        )
        result['report_files'] = report_files
    
    return result
