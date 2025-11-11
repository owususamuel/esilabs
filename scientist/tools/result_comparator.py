"""
Result Comparator Tool - Compares original vs reproduced experimental results.
Calculates similarity metrics and identifies discrepancies.
"""

import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class MetricComparison:
    """Comparison for a single metric."""
    
    metric_name: str
    original_value: float
    reproduced_value: float
    
    # Metrics
    absolute_difference: float
    relative_difference: float  # Percentage
    match_score: float  # 0-1: how closely they match
    
    is_close: bool  # Within acceptable threshold


class ResultComparator:
    
    # Acceptable thresholds for metric matching
    DEFAULT_THRESHOLDS = {
        'accuracy': 0.01,  # 1% tolerance for accuracy
        'loss': 0.05,      # 5% tolerance for loss
        'f1': 0.01,        # 1% tolerance for F1
        'auc': 0.02,       # 2% tolerance for AUC
        'default': 0.1     # 10% tolerance for unknown metrics
    }
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        
        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)
        
        self.logger = logger
    
    # --- Metric key normalization helpers ---
    def _normalize_metric_key(self, key: str) -> str:
        """
        Normalize heterogeneous metric keys into canonical names so we can match
        paper vs reproduced metrics even when their paths differ.
        
        Canonicalization examples:
        - "sentence/minimal.retrieval.bm25.metrics.recall.10" -> "bm25_recall@10"
        - "bm25.metrics.mrr" -> "bm25_mrr"
        - "Recall@10" -> "recall@10"
        """
        k = (key or "").strip().lower()
        if not k:
            return k
        
        # Standardize separators
        k = k.replace('\\', '/')
        for sep in [' ', '\t']:
            k = k.replace(sep, '')
        # Keep '/' for context detection but we'll also search raw text
        
        has_bm25 = 'bm25' in k
        
        base_name = None
        
        # Detect MRR
        if 'mrr' in k:
            base_name = 'mrr'
        
        # Detect recall@K
        if base_name is None and 'recall' in k:
            # Match recall followed by @, ., _, /, or nothing then a number
            m = re.search(r'recall(?:@|[._/])?(\d+)', k)
            if m:
                base_name = f"recall@{m.group(1)}"
            else:
                # Generic recall without K
                base_name = 'recall'
        
        # Other common IR metrics (expandable)
        if base_name is None and 'ndcg' in k:
            m = re.search(r'ndcg(?:@|[._/])?(\d+)', k)
            base_name = f"ndcg@{m.group(1)}" if m else 'ndcg'
        if base_name is None and 'precision' in k:
            m = re.search(r'precision(?:@|[._/])?(\d+)', k)
            base_name = f"precision@{m.group(1)}" if m else 'precision'
        if base_name is None and 'f1' in k:
            base_name = 'f1'
        if base_name is None and 'accuracy' in k:
            base_name = 'accuracy'
        if base_name is None and 'auc' in k:
            base_name = 'auc'
        if base_name is None and 'loss' in k:
            base_name = 'loss'
        
        if base_name:
            return f"bm25_{base_name}" if has_bm25 else base_name
        
        # Fallback: strip path-like structure to the last token
        tail = k.split('/')[-1]
        # Replace remaining dots with underscores
        tail = tail.replace('.', '_')
        return tail
    
    def _canonicalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Convert a dict of metrics into canonical names. If multiple raw keys map to the
        same canonical name, prefer ones that mention 'bm25' explicitly.
        """
        if not isinstance(metrics, dict):
            return {}
        
        canonical: Dict[str, float] = {}
        chosen_has_bm25: Dict[str, bool] = {}
        
        for raw_key, value in metrics.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            
            norm = self._normalize_metric_key(str(raw_key))
            has_bm25 = 'bm25' in str(raw_key).lower()
            
            if norm not in canonical:
                canonical[norm] = numeric_value
                chosen_has_bm25[norm] = has_bm25
            else:
                # Prefer bm25-specific over generic if conflict
                if has_bm25 and not chosen_has_bm25.get(norm, False):
                    canonical[norm] = numeric_value
                    chosen_has_bm25[norm] = True
        
        return canonical
    
    def _llm_align_metrics(
        self,
        original: Dict[str, float],
        reproduced: Dict[str, float],
        llm_client
    ) -> Dict[str, str]:
        """
        Ask the LLM to align metric keys between original and reproduced dicts.

        """
        # Limit to avoid overly long prompts
        def sample_dict(d: Dict[str, float], max_items: int = 50) -> Dict[str, float]:
            items = list(d.items())[:max_items]
            return {k: v for k, v in items}
        
        orig_sample = sample_dict(original)
        repro_sample = sample_dict(reproduced)
        
        import json as _json
        prompt = (
            "You are aligning metric keys between two result dictionaries from experiments.\n"
            "Given ORIGINAL and REPRODUCED metric dicts (keys with floats), produce a JSON mapping from\n"
            "original metric keys to the best matching reproduced metric keys. Only map comparable metrics\n"
            "like recall@K, mrr, ndcg@K, accuracy, loss, etc. Prefer BM25 metrics when relevant.\n\n"
            "Return ONLY valid JSON in this exact format:\n"
            "{\n"
            '  "alignments": {"original_key": "reproduced_key"}\n'
            "}\n\n"
            f"ORIGINAL: {_json.dumps(orig_sample, ensure_ascii=False)}\n\n"
            f"REPRODUCED: {_json.dumps(repro_sample, ensure_ascii=False)}\n"
        )
        
        # Call LLM
        if hasattr(llm_client, '__call__'):
            response = llm_client(prompt)
        elif hasattr(llm_client, 'invoke'):
            response = llm_client.invoke(prompt)
        else:
            response = str(llm_client)
        
        text = response if isinstance(response, str) else str(response)
        
        # Extract JSON
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            return {}
        
        try:
            data = _json.loads(match.group(0))
            alignments = data.get('alignments', {})
            # Filter to keys that exist
            filtered = {
                ok: rk for ok, rk in alignments.items()
                if ok in original and rk in reproduced
            }
            if filtered:
                self.logger.info(f"✅ LLM aligned {len(filtered)} metrics")
            return filtered
        except Exception:
            return {}
    
    @staticmethod
    def _load_extraction_prompt() -> str:
        import yaml
        from pathlib import Path
        
        # Find config file relative to this file
        config_path = Path(__file__).resolve().parents[2] / "config" / "agent_instructions.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('metric_extraction', {}).get('llm_prompt', '')
        except Exception as e:
            # Fallback prompt if config can't be loaded
            return """Extract all numerical metrics from the output below.
Return ONLY valid JSON: {{"metric_name": value, ...}}

Output:
{output_text}"""
    
    def extract_metrics(self, output_text: str, llm_client=None) -> Dict[str, float]:
        # If LLM is available, use intelligent extraction
        if llm_client:
            return self._llm_extract_metrics(output_text, llm_client)
        
        # Fallback: Use basic parsing (JSON first, then regex)
        return self._basic_extract_metrics(output_text)
    
    def _llm_extract_metrics(self, output_text: str, llm_client) -> Dict[str, float]:
        """
        Use LLM to intelligently extract metrics from any format.
        
        This is much more robust than regex patterns - it can handle:
        - JSON files (any structure)
        - Text logs with metrics
        - Tables (markdown, LaTeX, CSV)
        - Mixed formats
        - Abbreviated metric names
        """
        
        # Load prompt template from configuration
        prompt_template = self._load_extraction_prompt()
        prompt = prompt_template.format(output_text=output_text[:5000])
        
        try:
            # Use the correct method for smolagents models
            # The model object is an AzureOpenAIServerModel which expects messages array
            if hasattr(llm_client, '__call__'):
                # OpenAI/Azure models expect messages in array format
                messages = [{"role": "user", "content": prompt}]
                response = llm_client(messages)
            elif hasattr(llm_client, 'invoke'):
                response = llm_client.invoke(prompt)
            else:
                # Fallback: try calling directly
                response = str(llm_client)
            
            # Parse the JSON response
            import json
            # Try to extract JSON from the response
            response_text = response if isinstance(response, str) else str(response)
            
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                metrics = json.loads(json_match.group(0))
                
                # Ensure all values are floats
                metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                
                self.logger.info(f"✅ LLM extracted {len(metrics)} metrics")
                return metrics
            else:
                self.logger.warning("LLM response didn't contain valid JSON, falling back to basic extraction")
                return self._basic_extract_metrics(output_text)
                
        except Exception as e:
            self.logger.warning(f"LLM extraction failed: {e}, falling back to basic extraction")
            return self._basic_extract_metrics(output_text)
    
    def _basic_extract_metrics(self, output_text: str) -> Dict[str, float]:
        """
        Fallback: Basic metric extraction using JSON parsing and regex patterns.
        Used when LLM is not available or fails.
        """
        
        metrics = {}
        
        # Try JSON parsing first
        try:
            import json
            data = json.loads(output_text)
            
            # Recursively extract all numerical values from JSON with their paths
            def extract_from_json(obj, prefix=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        new_prefix = f"{prefix}{key}." if prefix else f"{key}."
                        extract_from_json(value, new_prefix)
                elif isinstance(obj, list):
                    # Only extract from first few items to avoid huge lists
                    for i, item in enumerate(obj[:3]):
                        extract_from_json(item, f"{prefix}[{i}].")
                elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                    metric_key = prefix.rstrip('.')
                    metrics[metric_key] = float(obj)
            
            extract_from_json(data)
            self.logger.info(f"Extracted {len(metrics)} metrics from JSON (basic parsing)")
            return metrics
            
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback to regex patterns for common metrics
        patterns = {
            'accuracy': r'accuracy[:\s]*([0-9.]+)',
            'loss': r'loss[:\s]*([0-9.]+)',
            'f1': r'f1[:\s]*([0-9.]+)',
            'precision': r'precision[:\s]*([0-9.]+)',
            'recall': r'recall[:\s]*([0-9.]+)',
            'recall@10': r'recall@10[:\s]*([0-9.]+)',
            'mrr': r'mrr[:\s]*([0-9.]+)',
        }
        
        output_lower = output_text.lower()
        
        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, output_lower, re.IGNORECASE)
            if matches:
                try:
                    metrics[metric_name] = float(matches[-1])
                except ValueError:
                    continue
        
        self.logger.info(f"Extracted {len(metrics)} metrics from text (basic regex)")
        return metrics
    
    def compare_metrics(
        self,
        original_metrics: Dict[str, float],
        reproduced_metrics: Dict[str, float],
        llm_client=None
    ) -> List[MetricComparison]:
        """
        Compare two sets of metrics and assess similarity.
        
        1. Match metrics between two runs
        2. Calculate different similarity measures
        3. Apply domain-specific thresholds
        4. Fall back to LLM-driven alignment when keys differ
        
        Args:
            original_metrics: Metrics from original paper
            reproduced_metrics: Metrics from reproduced run
            llm_client: Optional LLM for semantic alignment of metric keys
            
        Returns:
            List of MetricComparison objects
        """
        
        comparisons = []
        
        # Canonicalize metric keys to maximize matches
        canonical_original = self._canonicalize_metrics(original_metrics or {})
        canonical_reproduced = self._canonicalize_metrics(reproduced_metrics or {})
        
        # Find common metrics on canonical names
        common_metrics = set(canonical_original.keys()) & set(canonical_reproduced.keys())
        
        # If no overlap, try LLM-driven alignment
        aligned_pairs: List[Tuple[str, str]] = []
        if not common_metrics and llm_client:
            try:
                alignment = self._llm_align_metrics(canonical_original, canonical_reproduced, llm_client)
                for orig_key, repro_key in alignment.items():
                    aligned_pairs.append((orig_key, repro_key))
            except Exception as e:
                self.logger.warning(f"LLM alignment failed: {e}")
        
        if not common_metrics:
            self.logger.warning("No common metrics found between original and reproduced")
        
        # Build comparisons for direct matches
        for metric_name in common_metrics:
            original_value = canonical_original[metric_name]
            reproduced_value = canonical_reproduced[metric_name]
            
            # Calculate differences
            if original_value == 0:
                relative_diff = float('inf') if reproduced_value != 0 else 0
            else:
                relative_diff = abs(reproduced_value - original_value) / abs(original_value)
            
            absolute_diff = abs(reproduced_value - original_value)
            
            # Determine if values are close enough
            threshold = self.thresholds.get(metric_name, self.thresholds['default'])
            is_close = relative_diff <= threshold
            
            # Calculate match score (0-1, where 1 is perfect match)
            match_score = max(0, 1 - (relative_diff / (2 * threshold)))
            
            comparison = MetricComparison(
                metric_name=metric_name,
                original_value=original_value,
                reproduced_value=reproduced_value,
                absolute_difference=absolute_diff,
                relative_difference=relative_diff * 100,
                match_score=match_score,
                is_close=is_close
            )
            
            comparisons.append(comparison)
        
        # Build comparisons for aligned (different) keys
        for orig_key, repro_key in aligned_pairs:
            original_value = canonical_original[orig_key]
            reproduced_value = canonical_reproduced[repro_key]
            
            # Calculate differences
            if original_value == 0:
                relative_diff = float('inf') if reproduced_value != 0 else 0
            else:
                relative_diff = abs(reproduced_value - original_value) / abs(original_value)
            
            absolute_diff = abs(reproduced_value - original_value)
            
            # Determine if values are close enough
            threshold = self.thresholds.get(orig_key, self.thresholds['default'])
            is_close = relative_diff <= threshold
            
            # Calculate match score (0-1, where 1 is perfect match)
            match_score = max(0, 1 - (relative_diff / (2 * threshold)))
            
            comparison = MetricComparison(
                metric_name=orig_key,
                original_value=original_value,
                reproduced_value=reproduced_value,
                absolute_difference=absolute_diff,
                relative_difference=relative_diff * 100,
                match_score=match_score,
                is_close=is_close
            )
            
            comparisons.append(comparison)
        
        return comparisons
    
    def calculate_reproducibility_score(
        self,
        comparisons: List[MetricComparison]
    ) -> float:
        
        if not comparisons:
            return 0.0
        
        # Average the match scores
        total_score = sum(c.match_score for c in comparisons)
        average_score = total_score / len(comparisons)
        
        # Apply bonus for all metrics being close
        all_close = all(c.is_close for c in comparisons)
        if all_close:
            average_score = min(1.0, average_score * 1.1)
        
        return average_score
    
    def generate_comparison_report(
        self,
        comparisons: List[MetricComparison],
        overall_score: float,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        
        report = {
            'overall_score': overall_score,
            'total_metrics': len(comparisons),
            'metrics_matched': sum(1 for c in comparisons if c.is_close),
            'metrics_details': []
        }
        
        for comparison in comparisons:
            report['metrics_details'].append({
                'metric': comparison.metric_name,
                'original': comparison.original_value,
                'reproduced': comparison.reproduced_value,
                'absolute_diff': comparison.absolute_difference,
                'relative_diff_percent': comparison.relative_difference,
                'match_score': comparison.match_score,
                'is_close': comparison.is_close
            })
        
        if additional_info:
            report['additional_info'] = additional_info
        
        return report


def compare_results(
    original_output: str,
    reproduced_output: str,
    custom_thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    
    comparator = ResultComparator(thresholds=custom_thresholds)
    
    # Extract metrics
    original_metrics = comparator.extract_metrics(original_output)
    reproduced_metrics = comparator.extract_metrics(reproduced_output)
    
    # Compare
    comparisons = comparator.compare_metrics(original_metrics, reproduced_metrics)
    
    # Calculate score
    score = comparator.calculate_reproducibility_score(comparisons)
    
    # Generate report
    report = comparator.generate_comparison_report(comparisons, score)
    
    return report
