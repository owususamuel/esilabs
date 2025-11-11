"""
Report Generator - Creates visualizations and formatted reports from evaluation results.

Generates:
- Interactive HTML dashboards
- Static PNG charts
- CSV data exports
- PDF reports (optional)
- Journal-ready reproducibility reports
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive reproducibility reports with visualizations."""
    
    def __init__(self, output_base_dir: str = "data/outputs"):
        """
        Initialize report generator.
        
        Args:
            output_base_dir: Base directory for all outputs
        """
        self.output_base_dir = Path(output_base_dir)
        self.logger = logger
    
    def generate_full_report(
        self,
        run_data: Dict[str, Any],
        run_id: str,
        create_visualizations: bool = True
    ) -> Dict[str, str]:
        """
        Generate complete report package with all outputs.
        
        Args:
            run_data: Complete pipeline run data
            run_id: Unique run identifier
            create_visualizations: Whether to create charts/graphs
            
        Returns:
            Dictionary of generated file paths
        """
        self.logger.info(f"Generating comprehensive report for run {run_id}")
        
        # Create run-specific output directory
        run_dir = self.output_base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations subdirectory
        viz_dir = run_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        generated_files = {}
        
        # 1. Save raw JSON report
        json_path = run_dir / f"report_{run_id}.json"
        self._save_json_report(run_data, json_path)
        generated_files['json'] = str(json_path)
        
        # 2. Generate detailed comparison CSV
        csv_path = viz_dir / "detailed_comparison.csv"
        self._generate_comparison_csv(run_data, csv_path)
        generated_files['csv'] = str(csv_path)
        
        # 3. Generate human-readable text report
        text_path = run_dir / f"report_{run_id}.txt"
        self._generate_text_report(run_data, text_path)
        generated_files['text'] = str(text_path)
        
        # 4. Generate journal-ready reproducibility report
        journal_path = run_dir / f"reproducibility_statement_{run_id}.md"
        self._generate_journal_report(run_data, journal_path)
        generated_files['journal'] = str(journal_path)
        
        if create_visualizations:
            try:
                # 5. Generate visualizations
                viz_files = self._generate_visualizations(run_data, viz_dir)
                generated_files.update(viz_files)
            except Exception as e:
                self.logger.warning(f"Visualization generation failed: {e}")
                self.logger.info("Continuing with text-based reports...")
        
        self.logger.info(f"Report generation complete. {len(generated_files)} files created.")
        
        return generated_files
    
    def _save_json_report(self, data: Dict[str, Any], output_path: Path):
        """Save raw JSON report."""
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.logger.debug(f"JSON report saved: {output_path}")
    
    def _generate_comparison_csv(self, run_data: Dict[str, Any], output_path: Path):
        """Generate detailed comparison CSV for meta-analysis."""
        evaluation = run_data.get('pipeline', {}).get('evaluation', {})
        
        if not evaluation or not evaluation.get('success'):
            self.logger.warning("No evaluation data available for CSV export")
            return
        
        original_metrics = evaluation.get('original_metrics', {})
        reproduced_metrics = evaluation.get('reproduced_metrics', {})
        comparison_details = evaluation.get('comparison_details', {})
        
        # Prepare CSV data
        rows = []
        
        # Header
        rows.append([
            'Metric Name',
            'Paper Value',
            'Reproduced Value',
            'Absolute Difference',
            'Relative Difference (%)',
            'Within Threshold',
            'Category'
        ])
        
        # Metric comparisons
        comparisons = comparison_details.get('comparisons', [])
        for comp in comparisons:
            metric_name = comp.get('metric', 'unknown')
            original_val = comp.get('original', 0)
            reproduced_val = comp.get('reproduced', 0)
            diff = comp.get('difference', 0)
            
            # Calculate relative difference
            if original_val != 0:
                rel_diff = (abs(diff) / abs(original_val)) * 100
            else:
                rel_diff = 0 if diff == 0 else 100
            
            # Within threshold (typically 5% for reproducibility)
            within_threshold = rel_diff <= 5.0
            
            rows.append([
                metric_name,
                f"{original_val:.4f}" if isinstance(original_val, float) else str(original_val),
                f"{reproduced_val:.4f}" if isinstance(reproduced_val, float) else str(reproduced_val),
                f"{diff:.4f}" if isinstance(diff, float) else str(diff),
                f"{rel_diff:.2f}",
                'Yes' if within_threshold else 'No',
                'Numerical'
            ])
        
        # Visual comparisons
        image_comparisons = evaluation.get('image_comparisons', [])
        for img_comp in image_comparisons:
            rows.append([
                f"Figure {img_comp.get('paper_figure_index', '?')} Visual Match",
                'N/A',
                'N/A',
                'N/A',
                f"{img_comp.get('combined_similarity', 0) * 100:.2f}",
                'Yes' if img_comp.get('match', False) else 'No',
                'Visual'
            ])
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        self.logger.info(f"Comparison CSV saved: {output_path}")
    
    def _generate_text_report(self, run_data: Dict[str, Any], output_path: Path):
        """Generate human-readable text report."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("REPRODUCIBILITY EVALUATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Run metadata
        lines.append(f"Run ID: {run_data.get('run_id', 'unknown')}")
        lines.append(f"Generated: {run_data.get('timestamp', datetime.now().isoformat())}")
        lines.append("")
        
        pipeline = run_data.get('pipeline', {})
        
        # Paper information
        lines.append("-" * 80)
        lines.append("PAPER INFORMATION")
        lines.append("-" * 80)
        parsed_paper = pipeline.get('parsed_paper', {})
        lines.append(f"Title: {parsed_paper.get('title', 'N/A')}")
        lines.append(f"Authors: {', '.join(parsed_paper.get('authors', []))}")
        lines.append(f"Repository: {pipeline.get('found_repo_url', 'N/A')}")
        lines.append("")
        
        # Evaluation summary
        evaluation = pipeline.get('evaluation', {})
        if evaluation and evaluation.get('success'):
            lines.append("-" * 80)
            lines.append("EVALUATION SUMMARY")
            lines.append("-" * 80)
            
            score = evaluation.get('final_reproducibility_score', 0)
            lines.append(f"Overall Reproducibility Score: {score:.2%}")
            lines.append(f"Metrics Matched: {evaluation.get('metrics_matched', 0)}/{evaluation.get('total_metrics', 0)}")
            
            if evaluation.get('visual_score'):
                lines.append(f"Visual Similarity Score: {evaluation.get('visual_score', 0):.2%}")
            
            lines.append("")
            
            # Detailed metrics comparison
            lines.append("-" * 80)
            lines.append("DETAILED METRICS COMPARISON")
            lines.append("-" * 80)
            
            comparisons = evaluation.get('comparison_details', {}).get('comparisons', [])
            if comparisons:
                lines.append(f"{'Metric':<30} {'Paper':<15} {'Reproduced':<15} {'Diff':<15}")
                lines.append("-" * 75)
                for comp in comparisons:
                    metric = comp.get('metric', 'unknown')[:28]
                    orig = comp.get('original', 0)
                    repro = comp.get('reproduced', 0)
                    diff = comp.get('difference', 0)
                    lines.append(f"{metric:<30} {orig:<15.4f} {repro:<15.4f} {diff:<15.4f}")
            else:
                lines.append("No numerical metrics available")
            
            lines.append("")
            
            # Figure mapping
            figure_mapping = evaluation.get('figure_mapping', [])
            if figure_mapping:
                lines.append("-" * 80)
                lines.append("FIGURE MAPPING")
                lines.append("-" * 80)
                for mapping in figure_mapping:
                    lines.append(f"{mapping.get('paper_figure', 'Figure ?')}: {mapping.get('paper_caption', 'No caption')}")
                    lines.append(f"  → Reproduced: {mapping.get('reproduced_file', 'N/A')}")
                    lines.append(f"  → Similarity: {mapping.get('similarity_score', 0):.2%}")
                    lines.append(f"  → Match: {'✓' if mapping.get('match', False) else '✗'}")
                    lines.append("")
            
            # Analysis and recommendations
            if evaluation.get('analysis'):
                lines.append("-" * 80)
                lines.append("ANALYSIS")
                lines.append("-" * 80)
                lines.append(str(evaluation.get('analysis', '')))
                lines.append("")
            
            if evaluation.get('likely_causes'):
                lines.append("-" * 80)
                lines.append("LIKELY CAUSES OF DIFFERENCES")
                lines.append("-" * 80)
                for i, cause in enumerate(evaluation.get('likely_causes', []), 1):
                    lines.append(f"{i}. {cause}")
                lines.append("")
            
            if evaluation.get('recommendations'):
                lines.append("-" * 80)
                lines.append("RECOMMENDATIONS")
                lines.append("-" * 80)
                for i, rec in enumerate(evaluation.get('recommendations', []), 1):
                    lines.append(f"{i}. {rec}")
                lines.append("")
        else:
            lines.append("Evaluation incomplete or failed")
            lines.append("")
        
        # Footer
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        self.logger.info(f"Text report saved: {output_path}")
    
    def _generate_journal_report(self, run_data: Dict[str, Any], output_path: Path):
        """Generate journal-ready reproducibility statement."""
        lines = []
        
        # Title
        lines.append("# Reproducibility Statement")
        lines.append("")
        
        pipeline = run_data.get('pipeline', {})
        parsed_paper = pipeline.get('parsed_paper', {})
        evaluation = pipeline.get('evaluation', {})
        
        # Paper details
        lines.append(f"**Paper:** {parsed_paper.get('title', 'N/A')}")
        lines.append(f"**Authors:** {', '.join(parsed_paper.get('authors', []))}")
        lines.append(f"**Repository:** {pipeline.get('found_repo_url', 'N/A')}")
        lines.append(f"**Evaluation Date:** {run_data.get('timestamp', 'N/A')[:10]}")
        lines.append("")
        
        # Executive summary
        lines.append("## Executive Summary")
        lines.append("")
        
        if evaluation and evaluation.get('success'):
            score = evaluation.get('final_reproducibility_score', 0)
            
            if score >= 0.9:
                verdict = "**Highly Reproducible** ✓"
                desc = "The results were successfully reproduced with excellent agreement."
            elif score >= 0.7:
                verdict = "**Largely Reproducible** ≈"
                desc = "The results were reproduced with minor discrepancies."
            elif score >= 0.5:
                verdict = "**Partially Reproducible** ~"
                desc = "Some results were reproduced, but significant differences were observed."
            else:
                verdict = "**Not Reproducible** ✗"
                desc = "The results could not be reproduced or showed major discrepancies."
            
            lines.append(f"**Verdict:** {verdict}")
            lines.append(f"**Reproducibility Score:** {score:.1%}")
            lines.append("")
            lines.append(desc)
            lines.append("")
            
            # Key findings
            lines.append("## Key Findings")
            lines.append("")
            lines.append(f"- **Metrics Evaluated:** {evaluation.get('total_metrics', 0)}")
            lines.append(f"- **Metrics Matched:** {evaluation.get('metrics_matched', 0)}")
            
            if evaluation.get('visual_score'):
                lines.append(f"- **Visual Similarity:** {evaluation.get('visual_score', 0):.1%}")
            
            if evaluation.get('has_plot_metrics'):
                lines.append(f"- **Figures Analyzed:** {evaluation.get('plot_count', 0)}")
            
            lines.append("")
            
            # Detailed comparison
            lines.append("## Detailed Comparison")
            lines.append("")
            
            comparisons = evaluation.get('comparison_details', {}).get('comparisons', [])
            if comparisons:
                lines.append("| Metric | Paper | Reproduced | Difference |")
                lines.append("|--------|-------|------------|------------|")
                for comp in comparisons[:10]:  # Show top 10
                    metric = comp.get('metric', 'unknown')
                    orig = comp.get('original', 0)
                    repro = comp.get('reproduced', 0)
                    diff = comp.get('difference', 0)
                    lines.append(f"| {metric} | {orig:.4f} | {repro:.4f} | {diff:+.4f} |")
                lines.append("")
            
            # Reproducibility challenges
            if evaluation.get('likely_causes'):
                lines.append("## Observed Differences and Likely Causes")
                lines.append("")
                for cause in evaluation.get('likely_causes', []):
                    lines.append(f"- {cause}")
                lines.append("")
            
            # Recommendations
            if evaluation.get('recommendations'):
                lines.append("## Recommendations for Improved Reproducibility")
                lines.append("")
                for rec in evaluation.get('recommendations', []):
                    lines.append(f"1. {rec}")
                lines.append("")
        else:
            lines.append("Evaluation could not be completed.")
            lines.append("")
        
        # Methodology
        lines.append("## Evaluation Methodology")
        lines.append("")
        lines.append("This reproducibility assessment was conducted using an automated AI-powered evaluation pipeline:")
        lines.append("")
        lines.append("1. **Paper Parsing:** Extracted experimental setup, metrics, and figures from the paper")
        lines.append("2. **Code Execution:** Cloned and executed the repository code under the same conditions")
        lines.append("3. **Comparison:** Compared numerical metrics and visual outputs")
        lines.append("4. **Analysis:** Used semantic understanding (vision models) for figure comparison")
        lines.append("")
        
        # Citation
        lines.append("## How to Cite This Assessment")
        lines.append("")
        lines.append("```")
        lines.append(f"Reproducibility assessment of '{parsed_paper.get('title', 'N/A')}'")
        lines.append(f"Conducted: {run_data.get('timestamp', 'N/A')[:10]}")
        lines.append(f"Score: {evaluation.get('final_reproducibility_score', 0):.1%}")
        lines.append(f"Tool: AI Reproducibility Evaluator")
        lines.append("```")
        lines.append("")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        self.logger.info(f"Journal report saved: {output_path}")
    
    def _generate_visualizations(self, run_data: Dict[str, Any], viz_dir: Path) -> Dict[str, str]:
        """
        Generate visualization charts.
        
        Returns:
            Dictionary of generated visualization file paths
        """
        viz_files = {}
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            self.logger.warning("Matplotlib not available. Skipping visualizations.")
            return viz_files
        
        evaluation = run_data.get('pipeline', {}).get('evaluation', {})
        
        if not evaluation or not evaluation.get('success'):
            self.logger.warning("No evaluation data for visualizations")
            return viz_files
        
        # 1. Overall Performance Chart
        overall_path = viz_dir / "overall_performance.png"
        self._create_overall_chart(evaluation, overall_path, plt)
        viz_files['overall'] = str(overall_path)
        
        # 2. Baseline vs Reproduced Comparison
        comparison_path = viz_dir / "baseline_vs_reproduced.png"
        self._create_comparison_chart(evaluation, comparison_path, plt)
        viz_files['comparison'] = str(comparison_path)
        
        # 3. Deviation Distribution
        deviation_path = viz_dir / "deviation_distribution.png"
        self._create_deviation_chart(evaluation, deviation_path, plt)
        viz_files['deviation'] = str(deviation_path)
        
        # 4. Interactive HTML Dashboard
        html_path = viz_dir / "visualizations.html"
        self._create_html_dashboard(run_data, html_path)
        viz_files['dashboard'] = str(html_path)
        
        return viz_files
    
    def _create_overall_chart(self, evaluation: Dict[str, Any], output_path: Path, plt):
        """Create overall performance summary chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        score = evaluation.get('final_reproducibility_score', 0)
        metrics_matched = evaluation.get('metrics_matched', 0)
        total_metrics = evaluation.get('total_metrics', 1)
        visual_score = evaluation.get('visual_score', 0)
        
        # Data
        categories = ['Overall\nScore', 'Numerical\nMetrics', 'Visual\nSimilarity']
        scores = [score, metrics_matched / max(total_metrics, 1), visual_score]
        colors = ['#2ecc71' if s >= 0.8 else '#f39c12' if s >= 0.6 else '#e74c3c' for s in scores]
        
        # Bar chart
        bars = ax.bar(categories, scores, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, score_val in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score_val:.1%}',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Styling
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Reproducibility Assessment Summary', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Highly Reproducible (≥80%)')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Acceptable (≥60%)')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Overall chart saved: {output_path}")
    
    def _create_comparison_chart(self, evaluation: Dict[str, Any], output_path: Path, plt):
        """Create baseline vs reproduced comparison chart."""
        comparisons = evaluation.get('comparison_details', {}).get('comparisons', [])
        
        if not comparisons:
            self.logger.warning("No comparison data for chart")
            return
        
        # Extract data (limit to 10 metrics for readability)
        metrics = []
        paper_vals = []
        repro_vals = []
        
        for comp in comparisons[:10]:
            metrics.append(comp.get('metric', 'unknown')[:20])  # Truncate long names
            paper_vals.append(comp.get('original', 0))
            repro_vals.append(comp.get('reproduced', 0))
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(metrics))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], paper_vals, width, label='Paper', color='#3498db', alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], repro_vals, width, label='Reproduced', color='#e74c3c', alpha=0.8)
        
        # Styling
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Values', fontsize=12, fontweight='bold')
        ax.set_title('Paper vs Reproduced Results', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison chart saved: {output_path}")
    
    def _create_deviation_chart(self, evaluation: Dict[str, Any], output_path: Path, plt):
        """Create deviation distribution chart."""
        comparisons = evaluation.get('comparison_details', {}).get('comparisons', [])
        
        if not comparisons:
            self.logger.warning("No comparison data for deviation chart")
            return
        
        # Calculate relative deviations
        deviations = []
        for comp in comparisons:
            orig = comp.get('original', 0)
            diff = comp.get('difference', 0)
            if orig != 0:
                rel_dev = (abs(diff) / abs(orig)) * 100
                deviations.append(min(rel_dev, 100))  # Cap at 100%
        
        if not deviations:
            return
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        n, bins, patches = ax.hist(deviations, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
        
        # Color code bars
        for i, patch in enumerate(patches):
            if bins[i] <= 5:
                patch.set_facecolor('#2ecc71')  # Green for ≤5%
            elif bins[i] <= 10:
                patch.set_facecolor('#f39c12')  # Orange for 5-10%
            else:
                patch.set_facecolor('#e74c3c')  # Red for >10%
        
        # Add threshold lines
        ax.axvline(x=5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='≤5% (Excellent)')
        ax.axvline(x=10, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='≤10% (Acceptable)')
        
        # Styling
        ax.set_xlabel('Relative Deviation (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Metrics', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Metric Deviations', fontsize=14, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Deviation chart saved: {output_path}")
    
    def _create_html_dashboard(self, run_data: Dict[str, Any], output_path: Path):
        """Create interactive HTML dashboard."""
        pipeline = run_data.get('pipeline', {})
        evaluation = pipeline.get('evaluation', {})
        parsed_paper = pipeline.get('parsed_paper', {})
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reproducibility Report - {parsed_paper.get('title', 'Report')[:50]}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .score-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .score-large {{
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }}
        .score-excellent {{ color: #2ecc71; }}
        .score-good {{ color: #f39c12; }}
        .score-poor {{ color: #e74c3c; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            margin-top: 0;
            color: #667eea;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .badge-success {{ background-color: #2ecc71; color: white; }}
        .badge-warning {{ background-color: #f39c12; color: white; }}
        .badge-danger {{ background-color: #e74c3c; color: white; }}
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .image-card {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-card img {{
            width: 100%;
            border-radius: 5px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Reproducibility Evaluation Report</h1>
        <p><strong>Paper:</strong> {parsed_paper.get('title', 'N/A')}</p>
        <p><strong>Authors:</strong> {', '.join(parsed_paper.get('authors', []))}</p>
        <p><strong>Evaluated:</strong> {run_data.get('timestamp', 'N/A')}</p>
    </div>
    
    <div class="score-card">
        <h2>Overall Reproducibility Score</h2>
        <div class="score-large {self._get_score_class(evaluation.get('final_reproducibility_score', 0))}">
            {evaluation.get('final_reproducibility_score', 0):.1%}
        </div>
        <p style="text-align: center; font-size: 1.2em;">
            {self._get_verdict_text(evaluation.get('final_reproducibility_score', 0))}
        </p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <h3>📊 Numerical Metrics</h3>
            <div class="metric-value">{evaluation.get('metrics_matched', 0)}/{evaluation.get('total_metrics', 0)}</div>
            <p>Metrics Successfully Matched</p>
        </div>
        <div class="metric-card">
            <h3>🎨 Visual Similarity</h3>
            <div class="metric-value">{evaluation.get('visual_score', 0):.1%}</div>
            <p>Average Figure Match Score</p>
        </div>
        <div class="metric-card">
            <h3>📈 Figures Analyzed</h3>
            <div class="metric-value">{evaluation.get('plot_count', 0)}</div>
            <p>Plots/Charts Compared</p>
        </div>
    </div>
    
    <div class="score-card">
        <h2>Detailed Metric Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Paper Value</th>
                    <th>Reproduced Value</th>
                    <th>Difference</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Add comparison rows
        comparisons = evaluation.get('comparison_details', {}).get('comparisons', [])
        for comp in comparisons[:15]:  # Limit to 15 for readability
            metric = comp.get('metric', 'unknown')
            orig = comp.get('original', 0)
            repro = comp.get('reproduced', 0)
            diff = comp.get('difference', 0)
            
            # Calculate status
            if orig != 0:
                rel_diff = abs(diff / orig) * 100
            else:
                rel_diff = 0 if diff == 0 else 100
            
            if rel_diff <= 5:
                badge = '<span class="badge badge-success">Excellent</span>'
            elif rel_diff <= 10:
                badge = '<span class="badge badge-warning">Acceptable</span>'
            else:
                badge = '<span class="badge badge-danger">Differs</span>'
            
            html += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{orig:.4f}</td>
                    <td>{repro:.4f}</td>
                    <td>{diff:+.4f}</td>
                    <td>{badge}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
"""
        
        # Figure mapping section
        figure_mapping = evaluation.get('figure_mapping', [])
        if figure_mapping:
            html += """
    <div class="score-card">
        <h2>Figure Mapping</h2>
        <table>
            <thead>
                <tr>
                    <th>Paper Figure</th>
                    <th>Reproduced File</th>
                    <th>Similarity</th>
                    <th>Match</th>
                </tr>
            </thead>
            <tbody>
"""
            for mapping in figure_mapping:
                match_badge = '<span class="badge badge-success">✓</span>' if mapping.get('match') else '<span class="badge badge-danger">✗</span>'
                html += f"""
                <tr>
                    <td>{mapping.get('paper_figure', '?')}: {mapping.get('paper_caption', '')[:40]}</td>
                    <td>{mapping.get('reproduced_file', 'N/A')}</td>
                    <td>{mapping.get('similarity_score', 0):.1%}</td>
                    <td>{match_badge}</td>
                </tr>
"""
            html += """
            </tbody>
        </table>
    </div>
"""
        
        # Analysis section
        if evaluation.get('analysis'):
            html += f"""
    <div class="score-card">
        <h2>Analysis</h2>
        <p>{evaluation.get('analysis', '')}</p>
    </div>
"""
        
        # Recommendations
        if evaluation.get('recommendations'):
            html += """
    <div class="score-card">
        <h2>Recommendations</h2>
        <ul>
"""
            for rec in evaluation.get('recommendations', []):
                html += f"            <li>{rec}</li>\n"
            html += """
        </ul>
    </div>
"""
        
        # Footer
        html += f"""
    <div class="footer">
        <p>Generated by AI Reproducibility Evaluator</p>
        <p>Run ID: {run_data.get('run_id', 'unknown')}</p>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"HTML dashboard saved: {output_path}")
    
    def _get_score_class(self, score: float) -> str:
        """Get CSS class for score."""
        if score >= 0.8:
            return 'score-excellent'
        elif score >= 0.6:
            return 'score-good'
        else:
            return 'score-poor'
    
    def _get_verdict_text(self, score: float) -> str:
        """Get verdict text for score."""
        if score >= 0.9:
            return '✅ Highly Reproducible'
        elif score >= 0.7:
            return '≈ Largely Reproducible'
        elif score >= 0.5:
            return '~ Partially Reproducible'
        else:
            return '❌ Not Reproducible'
