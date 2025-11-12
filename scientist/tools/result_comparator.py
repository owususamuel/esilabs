"""
LLM-Driven Result Comparator - Intelligent comparison without hardcoded patterns.

The LLM handles all comparison logic given paper tables/results and reproduced results.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of LLM-driven comparison."""
    
    # Overall scores
    reproducibility_score: float  # 0-1
    metrics_matched: int
    total_metrics_compared: int
    
    # Detailed comparisons
    metric_comparisons: List[Dict[str, Any]]  # Each: {metric, paper_value, reproduced_value, match, difference}
    
    # Analysis
    analysis: str
    likely_causes: List[str]
    recommendations: List[str]
    
    # Raw data for transparency
    paper_metrics: Dict[str, float]
    reproduced_metrics: Dict[str, float]


class LLMResultComparator:
    """
    LLM-driven result comparator with no hardcoded patterns.
    
    The LLM:
    1. Receives paper results (tables, figures, text) and reproduced results
    2. Intelligently identifies which metrics correspond to each other
    3. Compares them and calculates reproducibility score
    4. Provides detailed analysis
    """
    
    def __init__(self, llm_client):
        """
        Initialize with LLM client.
        
        Args:
            llm_client: LLM client that can be called with messages
        """
        self.llm = llm_client
        self.logger = logger
    
    def compare_results(
        self,
        paper_data: Dict[str, Any],
        reproduced_data: Dict[str, Any]
    ) -> ComparisonResult:
        """
        Compare paper results with reproduced results using LLM intelligence.
        
        Args:
            paper_data: {
                'tables': [{'caption': '...', 'content': '...'}, ...],
                'figures': [{'caption': '...', 'image_base64': '...'}, ...],
                'experimental_results': 'text description of results',
                'evaluation_metrics': ['accuracy', 'f1', ...]
            }
            reproduced_data: {
                'output_files': [{'path': '...', 'content': '...'}, ...],
                'metrics': {'metric_name': value, ...},
                'stdout': 'command output',
                'plots': [{'path': '...'}, ...]
            }
        
        Returns:
            ComparisonResult with scores, analysis, and recommendations
        """
        self.logger.info("🤖 LLM comparing paper results with reproduced results...")
        
        # Step 1: Extract metrics from paper using LLM
        paper_metrics = self._extract_paper_metrics(paper_data)
        self.logger.info(f"  📊 Extracted {len(paper_metrics)} metrics from paper")
        
        # Step 2: Extract metrics from reproduced results using LLM
        reproduced_metrics = self._extract_reproduced_metrics(reproduced_data)
        self.logger.info(f"  📊 Extracted {len(reproduced_metrics)} metrics from reproduced results")
        
        # Step 3: Let LLM compare and analyze
        comparison = self._llm_compare(paper_metrics, reproduced_metrics, paper_data, reproduced_data)
        
        self.logger.info(f"  ✅ Reproducibility score: {comparison.reproducibility_score:.1%}")
        
        return comparison
    
    def _extract_paper_metrics(self, paper_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Use LLM to extract all metrics from paper tables, figures, and text.
        
        If pre_extracted_metrics are provided (from paper parsing stage), use those instead.
        """
        # CHECK FOR SAVED METRICS FIRST (from paper parsing stage)
        pre_extracted = paper_data.get('pre_extracted_metrics', {})
        if pre_extracted:
            self.logger.info(f"✅ Using {len(pre_extracted)} pre-extracted paper metrics (no re-extraction)")
            return pre_extracted
        
        # FALLBACK: Extract metrics if not pre-extracted
        self.logger.info("📊 No pre-extracted metrics found, extracting from paper data...")
        
        # Build context from all available paper data
        context_parts = []
        
        # Collect deterministic metrics from tables if present
        combined_metrics: Dict[str, float] = {}
        table_metrics = self._collect_table_metrics(paper_data.get('tables'))
        if table_metrics:
            self.logger.info(f"  📥 Collected {len(table_metrics)} metrics directly from tables")
            combined_metrics.update(table_metrics)

        # Add experimental results text
        if paper_data.get('experimental_results'):
            context_parts.append(f"EXPERIMENTAL RESULTS TEXT:\n{paper_data['experimental_results'][:3000]}")
        
        # Add table contents
        if paper_data.get('tables'):
            context_parts.append("\nTABLES:")
            for i, table in enumerate(paper_data['tables'][:5]):  # Limit to 5 tables
                caption = table.get('caption', '')
                content = table.get('content', '')
                context_parts.append(f"\nTable {i+1} - {caption}\n{content[:1000]}")
        
        # Add evaluation metrics mentioned
        if paper_data.get('evaluation_metrics'):
            context_parts.append(f"\nMETRICS USED: {', '.join(paper_data['evaluation_metrics'])}")
        
        context = '\n\n'.join(context_parts)
        
        # Prompt for extraction
        prompt = f"""Extract ALL numerical performance metrics from this research paper's results.

{context}

Extract every metric with its numerical value. Look for:
- Performance metrics: Accuracy, Precision, Recall, F1, MRR, NDCG, AUC
- Recall@k values: Recall@1, Recall@10, Recall@50
- Any numerical scores, percentages, or measurements

Return ONLY a JSON object with metric names as keys and numerical values:
{{
  "Recall@10": 0.85,
  "MRR": 0.72,
  "F1_Score": 0.68
}}

If no metrics found, return empty object: {{}}
"""
        
        try:
            response = self._call_llm(prompt)
            metrics = self._parse_json_response(response)
            normalized = self._normalize_metric_values(metrics)
            if normalized:
                self.logger.info(f"  🤖 LLM extracted {len(normalized)} metrics from paper text")
            combined_metrics.update(normalized)
            return combined_metrics
        except Exception as e:
            self.logger.error(f"Error extracting paper metrics: {e}")
            return combined_metrics
    
    def _extract_reproduced_metrics(self, reproduced_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Use LLM to extract all metrics from reproduced experiment outputs.
        """
        # Build context from reproduced data
        context_parts = []
        
        # Add pre-extracted metrics if available
        if reproduced_data.get('metrics'):
            context_parts.append(f"EXTRACTED METRICS:\n{json.dumps(reproduced_data['metrics'], indent=2)}")
        
        # Add stdout/stderr
        if reproduced_data.get('stdout'):
            context_parts.append(f"\nCOMMAND OUTPUT:\n{reproduced_data['stdout'][:2000]}")
        
        # Add output file contents
        if reproduced_data.get('output_files'):
            context_parts.append("\nOUTPUT FILES:")
            for i, file_info in enumerate(reproduced_data['output_files'][:3]):
                path = file_info.get('path', '')
                content = file_info.get('content', '')
                context_parts.append(f"\nFile: {path}\n{content[:1000]}")
        
        context = '\n\n'.join(context_parts)
        
        prompt = f"""Extract ALL numerical performance metrics from these reproduced experiment results.

{context}

Extract every metric with its numerical value. Return ONLY a JSON object:
{{
  "Recall@10": 0.83,
  "MRR": 0.70,
  "F1_Score": 0.65
}}

If no metrics found, return empty object: {{}}
"""
        
        try:
            response = self._call_llm(prompt)
            metrics = self._parse_json_response(response)
            return self._normalize_metric_values(metrics)
        except Exception as e:
            self.logger.error(f"Error extracting reproduced metrics: {e}")
            return reproduced_data.get('metrics', {})
    
    def _normalize_metric_values(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert metric values returned by the LLM into floats, handling string formats.
        """
        normalized: Dict[str, float] = {}
        
        for raw_key, raw_value in (metrics or {}).items():
            if raw_key is None:
                continue
            
            key = str(raw_key).strip()
            if not key:
                continue
            
            value = self._to_float(raw_value)
            if value is None:
                continue
            
            normalized[key] = value
        
        return normalized
    
    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """
        Attempt to coerce a metric value to float.
        Supports numeric types and common string representations (percentages, +/- ranges).
        """
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            
            # Detect percentages (e.g., "45.8%")
            is_percent = '%' in text
            
            # Remove common decorations
            cleaned = text.replace('%', '').replace(',', '')
            
            # Look for first numeric token
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
    
    def _collect_table_metrics(self, tables: Optional[List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Gather metrics that were deterministically extracted from tables.
        """
        collected: Dict[str, float] = {}
        
        if not tables:
            return collected
        
        for table in tables:
            if not isinstance(table, dict):
                continue
            
            table_number = table.get('table_number')
            prefix = f"Table_{table_number}_" if table_number else ""
            
            extracted_metrics = table.get('extracted_metrics') or {}
            if not isinstance(extracted_metrics, dict):
                continue
            
            for raw_name, raw_value in extracted_metrics.items():
                if raw_name is None:
                    continue
                metric_name = prefix + self._sanitize_metric_key(str(raw_name))
                value = self._to_float(raw_value)
                if value is None:
                    continue
                collected[metric_name] = value
        
        return collected
    
    @staticmethod
    def _sanitize_metric_key(name: str) -> str:
        """
        Normalize metric names into filesystem/JSON-friendly keys.
        """
        sanitized = re.sub(r'[^A-Za-z0-9@]+', '_', name.strip())
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        return sanitized or "metric"
    
    def _llm_compare(
        self,
        paper_metrics: Dict[str, float],
        reproduced_metrics: Dict[str, float],
        paper_data: Dict[str, Any],
        reproduced_data: Dict[str, Any]
    ) -> ComparisonResult:
        """
        Let LLM intelligently compare paper and reproduced metrics.
        """
        prompt = f"""Compare these research paper metrics with reproduced experiment results.

PAPER'S REPORTED METRICS:
{json.dumps(paper_metrics, indent=2)}

REPRODUCED EXPERIMENT METRICS:
{json.dumps(reproduced_metrics, indent=2)}

Your task:
1. Match metrics between paper and reproduced results (they may have different names)
2. For each matched metric, calculate if they're close (within 5-10% is good for most metrics)
3. Calculate overall reproducibility score (0.0 to 1.0)
4. Identify likely causes of discrepancies
5. Provide recommendations

Return ONLY this JSON structure:
{{
  "reproducibility_score": 0.85,
  "metric_comparisons": [
    {{
      "metric": "Recall@10",
      "paper_value": 0.85,
      "reproduced_value": 0.83,
      "difference": -0.02,
      "relative_difference_percent": -2.35,
      "match": true,
      "note": "Within acceptable range"
    }}
  ],
  "metrics_matched": 5,
  "total_metrics_compared": 6,
  "analysis": "The reproduced results closely match the paper's reported values...",
  "likely_causes": [
    "Minor differences due to random seed variation",
    "Possible differences in library versions"
  ],
  "recommendations": [
    "Results are highly reproducible",
    "Consider documenting random seed for exact replication"
  ]
}}

Guidelines for scoring:
- 1.0 = Perfect match (all metrics within 1%)
- 0.9-1.0 = Highly reproducible (within 5%)
- 0.7-0.9 = Reproducible (within 10%)
- 0.5-0.7 = Partially reproducible (within 20%)
- <0.5 = Not reproducible (>20% difference or missing metrics)

Be intelligent about matching:
- "Recall@10" matches "recall_at_10", "Recall_10", "R@10"
- "F1" matches "F1-score", "f1_score"
- Look for semantic similarity, not just exact string matches
"""
        
        try:
            response = self._call_llm(prompt)
            comparison_data = self._parse_json_response(response)
            
            return ComparisonResult(
                reproducibility_score=float(comparison_data.get('reproducibility_score', 0.0)),
                metrics_matched=int(comparison_data.get('metrics_matched', 0)),
                total_metrics_compared=int(comparison_data.get('total_metrics_compared', 0)),
                metric_comparisons=comparison_data.get('metric_comparisons', []),
                analysis=comparison_data.get('analysis', ''),
                likely_causes=comparison_data.get('likely_causes', []),
                recommendations=comparison_data.get('recommendations', []),
                paper_metrics=paper_metrics,
                reproduced_metrics=reproduced_metrics
            )
        except Exception as e:
            self.logger.error(f"Error in LLM comparison: {e}", exc_info=True)
            # Fallback to simple comparison
            return self._simple_fallback_comparison(paper_metrics, reproduced_metrics)
    
    def _simple_fallback_comparison(
        self,
        paper_metrics: Dict[str, float],
        reproduced_metrics: Dict[str, float]
    ) -> ComparisonResult:
        """
        Simple fallback comparison if LLM fails.
        """
        # Try exact key matches first
        comparisons = []
        for key in paper_metrics:
            if key in reproduced_metrics:
                paper_val = paper_metrics[key]
                repro_val = reproduced_metrics[key]
                diff = repro_val - paper_val
                rel_diff = (abs(diff) / paper_val * 100) if paper_val != 0 else float('inf')
                
                comparisons.append({
                    'metric': key,
                    'paper_value': paper_val,
                    'reproduced_value': repro_val,
                    'difference': diff,
                    'relative_difference_percent': rel_diff,
                    'match': rel_diff < 10,
                    'note': 'Exact key match'
                })
        
        matched = sum(1 for c in comparisons if c['match'])
        total = len(comparisons) if comparisons else max(len(paper_metrics), len(reproduced_metrics))
        score = matched / total if total > 0 else 0.0
        
        return ComparisonResult(
            reproducibility_score=score,
            metrics_matched=matched,
            total_metrics_compared=total,
            metric_comparisons=comparisons,
            analysis="Fallback comparison used (LLM comparison failed)",
            likely_causes=["Unable to perform detailed analysis"],
            recommendations=["Review metrics manually"],
            paper_metrics=paper_metrics,
            reproduced_metrics=reproduced_metrics
        )
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM with prompt and return response text.
        """
        if hasattr(self.llm, '__call__'):
            messages = [{"role": "user", "content": prompt}]
            response = self.llm(messages)
        elif hasattr(self.llm, 'invoke'):
            response = self.llm.invoke(prompt)
        else:
            response = str(self.llm)
        
        return response if isinstance(response, str) else str(response)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from LLM response.
        """
        # Try to find JSON in response
        json_match = re.search(r'\{[\s\S]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON: {e}")
                # Try to fix common JSON issues
                json_str = json_match.group(0)
                # Remove trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return {}
        return {}


# Legacy API for backward compatibility
class ResultComparator:
    """Legacy wrapper for backward compatibility."""
    
    def __init__(self, thresholds=None):
        self.logger = logger
        self.thresholds = thresholds or {}
    
    def extract_metrics(self, output_text: str, llm_client=None) -> Dict[str, float]:
        """Extract metrics using LLM if available."""
        if not llm_client:
            return self._basic_extract(output_text)
        
        comparator = LLMResultComparator(llm_client)
        result = comparator._extract_reproduced_metrics({'stdout': output_text})
        return result
    
    def _basic_extract(self, text: str) -> Dict[str, float]:
        """Basic regex extraction as fallback."""
        metrics = {}
        
        # Try JSON first
        try:
            data = json.loads(text)
            return self._flatten_json(data)
        except:
            pass
        
        # Regex patterns for common metrics
        patterns = {
            'recall@10': r'recall[@_]10[:\s=]+([0-9.]+)',
            'recall@50': r'recall[@_]50[:\s=]+([0-9.]+)',
            'mrr': r'mrr[:\s=]+([0-9.]+)',
            'accuracy': r'accuracy[:\s=]+([0-9.]+)',
            'f1': r'f1[:\s=]+([0-9.]+)',
        }
        
        text_lower = text.lower()
        for metric, pattern in patterns.items():
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    metrics[metric] = float(match.group(1))
                except ValueError:
                    pass
        
        return metrics
    
    def _flatten_json(self, obj, prefix='', max_depth=5):
        """Flatten nested JSON to extract numerical values."""
        metrics = {}
        
        if max_depth == 0:
            return metrics
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}/{key}" if prefix else key
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[new_key] = float(value)
                elif isinstance(value, dict):
                    metrics.update(self._flatten_json(value, new_key, max_depth - 1))
        
        return metrics
    
    def compare_metrics(
        self,
        original_metrics: Dict[str, float],
        reproduced_metrics: Dict[str, float],
        llm_client=None
    ) -> List[Dict[str, Any]]:
        """Compare metrics (legacy API)."""
        if not llm_client:
            # Simple exact match comparison
            comparisons = []
            for key in original_metrics:
                if key in reproduced_metrics:
                    orig = original_metrics[key]
                    repro = reproduced_metrics[key]
                    diff = repro - orig
                    comparisons.append({
                        'metric': key,
                        'original': orig,
                        'reproduced': repro,
                        'difference': diff,
                        'match': abs(diff / orig) < 0.1 if orig != 0 else diff == 0
                    })
            return comparisons
        
        # Use LLM comparator
        comparator = LLMResultComparator(llm_client)
        paper_data = {'experimental_results': json.dumps(original_metrics)}
        reproduced_data = {'metrics': reproduced_metrics}
        result = comparator.compare_results(paper_data, reproduced_data)
        return result.metric_comparisons
    
    def calculate_reproducibility_score(self, comparisons: List[Dict[str, Any]]) -> float:
        """Calculate score from comparisons."""
        if not comparisons:
            return 0.0
        matched = sum(1 for c in comparisons if c.get('match', False))
        return matched / len(comparisons)
    
    def generate_comparison_report(
        self,
        comparisons: List[Dict[str, Any]],
        overall_score: float,
        additional_info=None
    ) -> Dict[str, Any]:
        """Generate report (legacy API)."""
        return {
            'overall_score': overall_score,
            'total_metrics': len(comparisons),
            'metrics_matched': sum(1 for c in comparisons if c.get('match', False)),
            'comparisons': comparisons,
            'additional_info': additional_info or {}
        }
