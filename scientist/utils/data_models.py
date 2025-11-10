"""
Data models for the reproducibility agent system.
Defines structures for papers, experiments, results, and evaluations.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class PaperData(BaseModel):
    
    title: str
    authors: List[str]
    abstract: str
    publication_year: Optional[int] = None
    github_url: Optional[str] = None
    
    # Key sections extracted from paper
    methodology: Optional[str] = None
    datasets_used: List[str] = Field(default_factory=list)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    environment_requirements: Optional[str] = None
    
    experimental_results: Optional[str] = None  # Quantitative results: accuracy, F1, BLEU, etc.
    evaluation_metrics: List[str] = Field(default_factory=list)  # Metric names: ["accuracy", "F1", ...]
    key_findings: Optional[str] = None 
    
    # Metadata
    extracted_at: datetime = Field(default_factory=datetime.now)
    pdf_path: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class ExperimentConfig(BaseModel):
    
    repo_url: str
    repo_branch: Optional[str] = "main"
    working_directory: str
    
    # Execution details
    python_version: Optional[str] = "3.9"
    requirements_file: Optional[str] = "requirements.txt"
    setup_commands: List[str] = Field(default_factory=list)
    
    # Main experiment command
    run_command: str
    
    # Expected parameters from paper
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Timeouts and limits
    timeout_seconds: int = 300
    max_memory_mb: Optional[int] = None


class ExperimentResult(BaseModel):
    
    experiment_id: str
    status: str  # "success", "failure", "timeout"
    
    # Output metrics
    metrics: Dict[str, float] = Field(default_factory=dict)
    output_logs: str
    error_logs: Optional[str] = None
    
    # Timing
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # Artifacts
    artifacts_path: Optional[str] = None
    generated_files: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class ReproducibilityEvaluation(BaseModel):
    
    paper_id: str
    evaluation_id: str
    
    # Original vs Reproduced
    original_results: Dict[str, Any]
    reproduced_results: Dict[str, Any]
    
    # Evaluation scores
    metric_match_score: float  # 0-1: how closely metrics match
    success_rate: float  # 0-1: what % of experiments succeeded
    code_availability_score: float  # 0-1: code quality and availability
    documentation_score: float  # 0-1: documentation quality
    overall_reproducibility_score: float  # 0-1: overall score
    
    # Details
    differences: Dict[str, str] = Field(default_factory=dict)
    issues_found: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Metadata
    evaluated_at: datetime = Field(default_factory=datetime.now)
    evaluator_model: str
    
    class Config:
        arbitrary_types_allowed = True


class PipelineRun(BaseModel):
    
    run_id: str
    paper_id: str
    
    # Steps status
    parsing_status: str  # "pending", "in_progress", "completed", "failed"
    repo_finding_status: str
    experiment_status: str
    evaluation_status: str
    
    # Results
    parsed_paper: Optional[PaperData] = None
    found_repo_url: Optional[str] = None
    experiment_results: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation: Optional[Dict[str, Any]] = None
    
    # Tracking
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
