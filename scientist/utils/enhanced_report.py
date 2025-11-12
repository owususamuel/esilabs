"""
Enhanced Report Generator - Creates detailed, structured comparison reports.

Parses paper tables and reproduced results to create meaningful side-by-side comparisons.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EnhancedReportGenerator:
    """Generates detailed, structured reproducibility reports with proper comparisons."""
    
    def __init__(self):
        self.logger = logger
    
    def generate_comprehensive_report(
        self,
        paper_materials_dir: str,
        reproduced_results_dir: str,
        output_dir: str,
        run_id: str
    ) -> Dict[str, str]:
        """
        Generate comprehensive comparison report.
        
        Args:
            paper_materials_dir: Path to paper materials (tables, figures, metrics)
            reproduced_results_dir: Path to reproduced results
            output_dir: Output directory for reports
            run_id: Run identifier
            
        Returns:
            Dictionary of generated file paths
        """
        self.logger.info("🚀 Generating comprehensive comparison report...")
        
        # Parse paper materials
        paper_metrics = self._parse_paper_materials(paper_materials_dir)
        
        # Parse reproduced results
        reproduced_metrics = self._parse_reproduced_results(reproduced_results_dir)
        
        # Create detailed comparison
        comparison = self._create_detailed_comparison(paper_metrics, reproduced_metrics)
        
        # Generate reports
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Markdown report
        md_path = output_path / f"detailed_report_{run_id}.md"
        self._generate_markdown_report(comparison, paper_metrics, reproduced_metrics, md_path)
        
        # 2. HTML report
        html_path = output_path / f"detailed_report_{run_id}.html"
        self._generate_html_report(comparison, paper_metrics, reproduced_metrics, html_path)
        
        # 3. JSON export
        json_path = output_path / f"detailed_comparison_{run_id}.json"
        self._export_json_comparison(comparison, paper_metrics, reproduced_metrics, json_path)
        
        self.logger.info(f"✅ Generated 3 comprehensive reports in {output_dir}")
        
        return {
            'markdown': str(md_path),
            'html': str(html_path),
            'json': str(json_path)
        }
    
    def _parse_paper_materials(self, paper_materials_dir: str) -> Dict[str, Any]:
        """Parse paper tables and extract structured metrics."""
        paper_dir = Path(paper_materials_dir)
        
        if not paper_dir.exists():
            self.logger.warning(f"Paper materials directory not found: {paper_materials_dir}")
            return {}
        
        metrics = {
            'tables': {},
            'extracted_metrics': {},
            'figures': []
        }
        
        # Read tables
        tables_dir = paper_dir / "tables"
        if tables_dir.exists():
            for table_file in sorted(tables_dir.glob("table_*.txt")):
                table_num = re.search(r'table_(\d+)', table_file.name)
                if table_num:
                    table_num = table_num.group(1)
                    table_content = table_file.read_text()
                    parsed = self._parse_table_content(table_content, table_num)
                    if parsed:
                        metrics['tables'][f'table_{table_num}'] = parsed
                        self.logger.info(f"  ✓ Parsed Table {table_num}: {len(parsed)} configurations")
        
        # Read extracted metrics
        metrics_file = paper_dir / "extracted_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics['extracted_metrics'] = json.load(f)
        
        return metrics
    
    def _parse_table_content(self, content: str, table_num: str) -> Dict[str, Dict[str, float]]:
        """
        Parse table content to extract structured metrics.
        
        Table 3 format:
        Sentence-level
        Sentence
        minimal
        0.632
        0.774
        0.891
        0.474
        Sentence
        title_only
        0.629
        ...
        """
        results = {}
        
        try:
            lines = [l.strip() for l in content.strip().split('\n')]
            
            current_granularity = None
            current_template = None
            current_values = []
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Skip empty lines, table title, or separators
                if not line or line.lower().startswith('table') or '====' in line:
                    i += 1
                    continue
                
                # Check for granularity level markers
                if 'sentence-level' in line.lower():
                    current_granularity = 'sentence'
                    i += 1
                    continue
                elif 'paragraph-level' in line.lower():
                    current_granularity = 'paragraph'
                    i += 1
                    continue
                
                # Check if line is "Sentence" or "Paragraph" (type marker)
                if line.lower() in ['sentence', 'paragraph']:
                    current_granularity = line.lower()
                    i += 1
                    continue
                
                # Skip header rows
                if line.lower() in ['granularity', 'template', 'recall@5', 'recall@10', 'recall@20', 'mrr']:
                    i += 1
                    continue
                
                # Try to parse as float (metric value)
                try:
                    val = float(line)
                    # If we have a template set, collect values
                    if current_template and current_granularity:
                        current_values.append(val)
                        
                        # If we've collected 4 values (Recall@5, @10, @20, MRR), save them
                        if len(current_values) >= 4:
                            config_key = f"{current_granularity}/{current_template}"
                            results[config_key] = {
                                'recall@5': current_values[0],
                                'recall@10': current_values[1],
                                'recall@20': current_values[2],
                                'mrr': current_values[3]
                            }
                            # Reset for next configuration
                            current_values = []
                            current_template = None
                    i += 1
                    continue
                except ValueError:
                    # Not a number - could be template name or text
                    pass
                
                # Check if it's a template name
                template_names = ['minimal', 'title_only', 'heading_only', 'title_heading', 'aggressive_title']
                if line.lower() in template_names or '_' in line.lower():
                    # Save previous config if we had one pending
                    if current_template and current_values and len(current_values) >= 4:
                        config_key = f"{current_granularity}/{current_template}"
                        results[config_key] = {
                            'recall@5': current_values[0],
                            'recall@10': current_values[1],
                            'recall@20': current_values[2],
                            'mrr': current_values[3]
                        }
                    
                    # Start new config
                    current_template = line.lower()
                    current_values = []
                else:
                    # Skip text lines (evidence, bullets, etc.)
                    pass
                
                i += 1
            
            # Save last config if pending
            if current_template and current_values and len(current_values) >= 4 and current_granularity:
                config_key = f"{current_granularity}/{current_template}"
                results[config_key] = {
                    'recall@5': current_values[0],
                    'recall@10': current_values[1],
                    'recall@20': current_values[2],
                    'mrr': current_values[3]
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error parsing table {table_num}: {e}", exc_info=True)
            return {}
    
    def _parse_reproduced_results(self, reproduced_results_dir: str) -> Dict[str, Any]:
        """Parse reproduced results JSON files."""
        results_dir = Path(reproduced_results_dir) / "data"
        
        if not results_dir.exists():
            self.logger.warning(f"Reproduced results directory not found: {results_dir}")
            return {}
        
        all_results = {}
        
        # Look for individual result files (e.g., results_sentence_minimal_1.json)
        for result_file in sorted(results_dir.glob("results_*.json")):
            # Skip complete_results files for now
            if 'complete' in result_file.name:
                continue
            
            try:
                with open(result_file) as f:
                    data = json.load(f)
                
                # Extract configuration from filename
                # Format: results_{granularity}_{template}[_{run_number}].json
                # e.g., "results_sentence_minimal_1.json" -> "sentence/minimal"
                # e.g., "results_sentence_aggressive_title.json" -> "sentence/aggressive_title"
                
                # Remove "results_" prefix and ".json" suffix
                name_parts = result_file.stem.replace('results_', '')
                
                # Check if last part is a number (run number)
                parts = name_parts.split('_')
                if parts[-1].isdigit():
                    # Skip run numbers > 0 to avoid duplicates
                    if int(parts[-1]) > 0:
                        continue
                    parts = parts[:-1]  # Remove run number
                
                # First part is granularity (sentence or paragraph)
                if not parts:
                    continue
                    
                granularity = parts[0]
                
                # Rest is template (join with underscores)
                template = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
                
                config_key = f"{granularity}/{template}"
                
                # Extract metrics
                metrics = self._extract_metrics_from_result(data)
                if metrics:
                    all_results[config_key] = metrics
                    self.logger.info(f"  ✓ Parsed {config_key}: {len(metrics)} metrics")
            
            except Exception as e:
                self.logger.warning(f"Error parsing {result_file.name}: {e}")
                continue
        
        return all_results
    
    def _extract_metrics_from_result(self, data: Dict) -> Dict[str, float]:
        """Extract key metrics from reproduced result JSON."""
        metrics = {}
        
        try:
            # Navigate nested structure
            if 'retrieval' in data:
                retrieval = data['retrieval']
                
                # BM25 metrics (most relevant for comparison)
                if 'bm25' in retrieval:
                    bm25 = retrieval['bm25']
                    
                    if 'recall' in bm25:
                        recall = bm25['recall']
                        if isinstance(recall, dict):
                            for k, v in recall.items():
                                metrics[f'recall@{k}'] = float(v)
                    
                    if 'mrr' in bm25:
                        metrics['mrr'] = float(bm25['mrr'])
                    
                    if 'ndcg' in bm25:
                        ndcg = bm25['ndcg']
                        if isinstance(ndcg, dict):
                            for k, v in ndcg.items():
                                metrics[f'ndcg@{k}'] = float(v)
            
            # Downstream metrics
            if 'downstream' in data and 'bm25' in data['downstream']:
                downstream = data['downstream']['bm25']
                
                if 'answerability' in downstream:
                    ans = downstream['answerability']
                    for key in ['accuracy', 'precision', 'recall', 'f1_score']:
                        if key in ans:
                            metrics[f'answerability_{key}'] = float(ans[key])
        
        except Exception as e:
            self.logger.error(f"Error extracting metrics: {e}", exc_info=True)
        
        return metrics
    
    def _create_detailed_comparison(
        self,
        paper_metrics: Dict[str, Any],
        reproduced_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create detailed comparison between paper and reproduced metrics."""
        
        comparison = {
            'configurations': {},
            'summary': {
                'total_configs': 0,
                'configs_matched': 0,
                'total_metrics': 0,
                'metrics_matched': 0,
                'metrics_close': 0,  # within 5%
                'metrics_different': 0
            }
        }
        
        # Get paper table metrics (focus on Table 3 which has the main results)
        paper_table_metrics = {}
        if 'tables' in paper_metrics and 'table_3' in paper_metrics['tables']:
            paper_table_metrics = paper_metrics['tables']['table_3']
        
        # Compare each configuration
        all_configs = set(paper_table_metrics.keys()) | set(reproduced_metrics.keys())
        
        for config in sorted(all_configs):
            paper_config = paper_table_metrics.get(config, {})
            repro_config = reproduced_metrics.get(config, {})
            
            config_comparison = {
                'paper_metrics': paper_config,
                'reproduced_metrics': repro_config,
                'metric_comparisons': [],
                'has_paper_data': bool(paper_config),
                'has_repro_data': bool(repro_config)
            }
            
            if paper_config and repro_config:
                comparison['summary']['configs_matched'] += 1
                
                # Compare individual metrics
                all_metrics = set(paper_config.keys()) | set(repro_config.keys())
                
                for metric in sorted(all_metrics):
                    paper_val = paper_config.get(metric)
                    repro_val = repro_config.get(metric)
                    
                    if paper_val is not None and repro_val is not None:
                        diff = repro_val - paper_val
                        rel_diff = abs(diff / paper_val) if paper_val != 0 else float('inf')
                        
                        match_status = 'exact' if abs(diff) < 0.001 else \
                                     'close' if rel_diff < 0.05 else \
                                     'different'
                        
                        config_comparison['metric_comparisons'].append({
                            'metric': metric,
                            'paper': paper_val,
                            'reproduced': repro_val,
                            'difference': diff,
                            'relative_diff': rel_diff,
                            'match_status': match_status
                        })
                        
                        comparison['summary']['total_metrics'] += 1
                        if match_status == 'exact':
                            comparison['summary']['metrics_matched'] += 1
                        elif match_status == 'close':
                            comparison['summary']['metrics_close'] += 1
                        else:
                            comparison['summary']['metrics_different'] += 1
            
            comparison['configurations'][config] = config_comparison
            comparison['summary']['total_configs'] = len(all_configs)
        
        # Calculate overall score
        total = comparison['summary']['total_metrics']
        matched = comparison['summary']['metrics_matched']
        close = comparison['summary']['metrics_close']
        
        if total > 0:
            comparison['summary']['reproducibility_score'] = (matched + 0.5 * close) / total
        else:
            comparison['summary']['reproducibility_score'] = 0.0
        
        return comparison
    
    def _generate_markdown_report(
        self,
        comparison: Dict[str, Any],
        paper_metrics: Dict[str, Any],
        reproduced_metrics: Dict[str, Any],
        output_path: Path
    ):
        """Generate detailed markdown report."""
        lines = []
        
        # Header
        lines.append("# Detailed Reproducibility Report\n")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        lines.append("---\n\n")
        
        # Executive Summary
        summary = comparison['summary']
        score = summary['reproducibility_score']
        
        lines.append("## 📊 Executive Summary\n\n")
        lines.append(f"**Reproducibility Score:** {score:.1%}\n\n")
        lines.append(f"- **Configurations Compared:** {summary['configs_matched']}/{summary['total_configs']}\n")
        lines.append(f"- **Total Metrics:** {summary['total_metrics']}\n")
        lines.append(f"- ✅ **Exact Matches:** {summary['metrics_matched']}\n")
        lines.append(f"- 🟡 **Close Matches (within 5%):** {summary['metrics_close']}\n")
        lines.append(f"- ❌ **Different:** {summary['metrics_different']}\n\n")
        
        # Overall verdict
        if score >= 0.9:
            verdict = "✅ **HIGHLY REPRODUCIBLE** - Results closely match the paper"
        elif score >= 0.75:
            verdict = "✓ **REPRODUCIBLE** - Minor discrepancies within acceptable range"
        elif score >= 0.5:
            verdict = "⚠️ **PARTIALLY REPRODUCIBLE** - Significant differences found"
        else:
            verdict = "❌ **NOT REPRODUCIBLE** - Major discrepancies in results"
        
        lines.append(f"{verdict}\n\n")
        lines.append("---\n\n")
        
        # Configuration-by-Configuration Comparison
        lines.append("## 🔬 Detailed Configuration Comparison\n\n")
        
        for config, comp in sorted(comparison['configurations'].items()):
            if not comp['has_paper_data'] or not comp['has_repro_data']:
                continue
            
            lines.append(f"### {config.replace('/', ' / ').title()}\n\n")
            
            metric_comps = comp['metric_comparisons']
            if metric_comps:
                lines.append("| Metric | Paper | Reproduced | Difference | Status |\n")
                lines.append("|--------|-------|------------|------------|--------|\n")
                
                for mc in metric_comps:
                    metric = mc['metric'].replace('@', '@')
                    paper_val = mc['paper']
                    repro_val = mc['reproduced']
                    diff = mc['difference']
                    status = mc['match_status']
                    
                    status_icon = '✅' if status == 'exact' else '🟡' if status == 'close' else '❌'
                    diff_str = f"{diff:+.4f}" if abs(diff) < 10 else f"{diff:+.2e}"
                    
                    lines.append(f"| {metric} | {paper_val:.4f} | {repro_val:.4f} | {diff_str} | {status_icon} |\n")
                
                lines.append("\n")
        
        # Missing Configurations
        lines.append("## ⚠️ Missing Configurations\n\n")
        
        missing_paper = []
        missing_repro = []
        
        for config, comp in comparison['configurations'].items():
            if comp['has_paper_data'] and not comp['has_repro_data']:
                missing_repro.append(config)
            elif not comp['has_paper_data'] and comp['has_repro_data']:
                missing_paper.append(config)
        
        if missing_repro:
            lines.append("**In paper but not reproduced:**\n")
            for config in missing_repro:
                lines.append(f"- {config}\n")
            lines.append("\n")
        
        if missing_paper:
            lines.append("**Reproduced but not in paper:**\n")
            for config in missing_paper:
                lines.append(f"- {config}\n")
            lines.append("\n")
        
        if not missing_paper and not missing_repro:
            lines.append("✅ All configurations present in both paper and reproduced results.\n\n")
        
        # Write file
        output_path.write_text(''.join(lines))
        self.logger.info(f"  ✓ Markdown report: {output_path}")
    
    def _generate_html_report(
        self,
        comparison: Dict[str, Any],
        paper_metrics: Dict[str, Any],
        reproduced_metrics: Dict[str, Any],
        output_path: Path
    ):
        """Generate interactive HTML report."""
        summary = comparison['summary']
        score = summary['reproducibility_score']
        
        # Determine status styling
        if score >= 0.9:
            status_class = "highly-reproducible"
            status_label = "Highly Reproducible"
        elif score >= 0.75:
            status_class = "reproducible"
            status_label = "Reproducible"
        elif score >= 0.5:
            status_class = "partial"
            status_label = "Partially Reproducible"
        else:
            status_class = "not-reproducible"
            status_label = "Not Reproducible"
        
        # Generate configuration tables
        config_tables = []
        for config, comp in sorted(comparison['configurations'].items()):
            if not comp['has_paper_data'] or not comp['has_repro_data']:
                continue
            
            config_title = config.replace('/', ' / ').title()
            
            rows = []
            for mc in comp['metric_comparisons']:
                status_class_row = 'exact' if mc['match_status'] == 'exact' else \
                                  'close' if mc['match_status'] == 'close' else 'different'
                status_icon = '✅' if mc['match_status'] == 'exact' else \
                             '🟡' if mc['match_status'] == 'close' else '❌'
                
                rows.append(f"""
                    <tr class="{status_class_row}">
                        <td>{mc['metric'].upper()}</td>
                        <td>{mc['paper']:.4f}</td>
                        <td>{mc['reproduced']:.4f}</td>
                        <td>{mc['difference']:+.4f}</td>
                        <td>{status_icon}</td>
                    </tr>
                """)
            
            config_tables.append(f"""
                <div class="config-section">
                    <h3>{config_title}</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Paper</th>
                                <th>Reproduced</th>
                                <th>Difference</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(rows)}
                        </tbody>
                    </table>
                </div>
            """)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detailed Reproducibility Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #1f2937;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 12px 32px;
            border-radius: 50px;
            font-weight: bold;
            font-size: 1.2em;
            margin-top: 20px;
        }}
        
        .highly-reproducible {{ background: #10b981; }}
        .reproducible {{ background: #3b82f6; }}
        .partial {{ background: #f59e0b; }}
        .not-reproducible {{ background: #ef4444; }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f9fafb;
        }}
        
        .summary-card {{
            text-align: center;
            padding: 20px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .summary-card .number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
        }}
        
        .summary-card .label {{
            color: #6b7280;
            font-size: 0.9em;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .config-section {{
            margin: 30px 0;
            padding: 20px;
            background: #f9fafb;
            border-radius: 12px;
        }}
        
        .config-section h3 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        thead {{
            background: #667eea;
            color: white;
        }}
        
        th, td {{
            padding: 12px 16px;
            text-align: left;
        }}
        
        th {{
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        
        tbody tr {{
            border-bottom: 1px solid #e5e7eb;
        }}
        
        tbody tr:hover {{
            background: #f3f4f6;
        }}
        
        tr.exact {{
            background: #d1fae5;
        }}
        
        tr.close {{
            background: #fef3c7;
        }}
        
        tr.different {{
            background: #fee2e2;
        }}
        
        .footer {{
            background: #1f2937;
            color: #9ca3af;
            padding: 20px;
            text-align: center;
            font-size: 0.9em;
        }}
        
        h2 {{
            color: #667eea;
            margin: 30px 0 20px 0;
            font-size: 1.8em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Detailed Reproducibility Report</h1>
            <div class="status-badge {status_class}">
                {score:.0%} - {status_label}
            </div>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <div class="number">{score:.0%}</div>
                <div class="label">Reproducibility Score</div>
            </div>
            <div class="summary-card">
                <div class="number">{summary['configs_matched']}/{summary['total_configs']}</div>
                <div class="label">Configurations</div>
            </div>
            <div class="summary-card">
                <div class="number">{summary['metrics_matched']}</div>
                <div class="label">Exact Matches ✅</div>
            </div>
            <div class="summary-card">
                <div class="number">{summary['metrics_close']}</div>
                <div class="label">Close Matches 🟡</div>
            </div>
            <div class="summary-card">
                <div class="number">{summary['metrics_different']}</div>
                <div class="label">Different ❌</div>
            </div>
        </div>
        
        <div class="content">
            <h2>Configuration Comparisons</h2>
            {''.join(config_tables)}
        </div>
        
        <div class="footer">
            Generated by AI Reproducibility Evaluator • {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>
</body>
</html>"""
        
        output_path.write_text(html)
        self.logger.info(f"  ✓ HTML report: {output_path}")
    
    def _export_json_comparison(
        self,
        comparison: Dict[str, Any],
        paper_metrics: Dict[str, Any],
        reproduced_metrics: Dict[str, Any],
        output_path: Path
    ):
        """Export structured comparison as JSON."""
        export_data = {
            'summary': comparison['summary'],
            'configurations': comparison['configurations'],
            'paper_metrics': paper_metrics,
            'reproduced_metrics': reproduced_metrics,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"  ✓ JSON export: {output_path}")

