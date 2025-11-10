# Research Paper Reproducibility Agent

An intelligent multi-agent system that automatically reproduces research paper experiments and evaluates their reproducibility. This project uses smolagents (HuggingFace's lightweight agent framework) to orchestrate specialized agents that parse papers, find code repositories, run experiments, and evaluate results.

## 🏗️ Architecture

### LLM-Powered AI Agent System

This system uses **LLM intelligence** for decision-making, not procedural code:

```
┌─────────────────────────────────────────────────────────────┐
│                 Reproducibility Orchestrator                 │
│                    (Pipeline Coordinator)                    │
└──────────┬──────────────┬──────────────┬──────────────┬──────┘
           │              │              │              │
    ┌──────▼──────┐  ┌────▼────┐  ┌──────▼──────┐  ┌────▼────┐
    │ Paper       │  │  Repo   │  │ Experiment  │  │Evaluator│
    │ Parser      │  │ Finder  │  │  Runner     │  │  Agent  │
    │ Agent       │  │ Agent   │  │  Agent      │  │         │
    │             │  │         │  │             │  │         │
    │ LLM: Extract│  │LLM: Pick│  │LLM: Analyze │  │LLM: Eval│
    │ structured  │  │best repo│  │& build cmd  │  │results  │
    │ info        │  │         │  │             │  │         │
    └──────┬──────┘  └────┬────┘  └──────┬──────┘  └────┬────┘
           │              │              │              │
    ┌──────▼──┐    ┌──────▼──┐    ┌──────▼──┐    ┌──────▼──┐
    │  PDF    │    │GitHub    │   │  Code   │   │ Result  │
    │ Parser  │    │          │   │Executor │   │Comparator
    │ Tool    │    │          │   │  Tool   │   │ Tool    │
    └─────────┘    └──────────┘   └─────────┘   └─────────┘
```

**Key Principle**: Agents **THINK** (using LLM), Tools **DO** (using code)

### How Each Agent Uses LLM

1. **Paper Parser Agent**
   - Tools: Read PDF, extract text
   - **LLM**: Intelligently extract datasets, hyperparameters, methodology

2. **Repo Finder Agent**  
   - Tools: Query GitHub API, get repo metadata
   - **LLM**: Analyze and select the most likely official repository

3. **Experiment Runner Agent**
   - Tools: Read files, run commands
   - **LLM**: Understand repository, determine how to run code, construct command with all required arguments

4. **Evaluator Agent**
   - Tools: Calculate metric differences
   - **LLM**: Analyze WHY results differ, assess quality, provide recommendations

### Installation

1. **Clone and setup the environment:**

```bash
cd ~/esilabs
python -m venv .venv
source .venv/bin/activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Configure environment variables:**

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Quick Start

```bash
python launch_agent.py

# or with uv
uv run launch_agent.py
```

## 📚 Understanding the Code

### 1. Base Agent Framework (`scientist/agents/base_agent.py`)

**Key Concepts:**
- Abstract base class pattern
- Tool registration system
- LLM client integration
- Execution history tracking

**Example:**
```python
class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="my_agent",
            system_prompt="Your system prompt here"
        )
        self.register_tool("my_tool", my_tool_function)
    
    def execute(self, task):
        result = self.call_llm(messages=[...])
        self.log_execution("step_name", result)
        return result
```


### 2. Specialized Agents

Each agent works **autonomously** using tools:

#### Paper Parser Agent
- **Tools**: `parse_pdf` (reads PDF content)
- **Autonomous Behavior**: Agent reads PDF, analyzes content, extracts structured information (title, authors, datasets, hyperparameters, methodology, experiment results, metrics)
- **Result**: Structured JSON with all paper information

#### Repo Finder Agent
- **Tools**: `search_github` (queries GitHub)
- **Autonomous Behavior**: Agent searches GitHub with multiple queries, analyzes results, selects best match based on authors/stars/description
- **Result**: Selected repository with confidence score and reasoning

#### Interactive Mode 🎯

When automated repository search fails, the system can ask you to provide one manually:

```python
from scientist.main import run_reproducibility_pipeline

# Run with interactive fallback (enabled by default)
result = run_reproducibility_pipeline(
    pdf_path="paper.pdf"
)

# Terminal prompts you:
# NO REPOSITORY FOUND - INTERACTIVE MODE
# Choose [1/2] or [q]uit: 
#   1. GitHub Repository URL
#   2. Local ZIP file containing code
```


#### Experiment Runner Agent
- **Tools**: `read_file_contents`, `list_directory_files`, `run_command_in_repo`, `create_file_or_directory`, `extract_metrics`
- **Autonomous Behavior**: Agent explores repo, reads README, runs scripts with --help, determines required arguments, creates needed files, installs dependencies, executes experiments, extracts results
- **Result**: Successfully executed experiments with metrics
- **Example**: Agent sees `usage: script.py [-h] --text-path TEXT_PATH --out-dir OUT_DIR`, creates input.txt and output/, then runs with both args
- **Smart Caching**: Uses hybrid venv strategy for fast, isolated environments

#### Evaluator Agent
- **Tools**: `extract_metrics` (extracts numerical values)
- **Autonomous Behavior**: Agent compares original and reproduced results, analyzes significance of differences, identifies likely causes, provides recommendations
- **Result**: Comprehensive reproducibility report with scores, analysis, and actionable insights

### 4. Tools

Each tool encapsulates a specific capability:

#### PDF Parser Tool
```python
from scientist.tools.pdf_parser import PDFParser

parser = PDFParser()
content = parser.parse_pdf("paper.pdf")
print(content.title, content.abstract)
```

#### Code Executor Tool
```python
from scientist.tools.code_executor import CodeExecutor

executor = CodeExecutor(sandbox_mode=True, max_timeout=300)
result = executor.execute_command("python train.py")
print(result.stdout, result.duration_seconds)
```

### Modifying Agent Behavior

Agents dynamically load their system prompts from `config/agent_instructions.yaml`. You can customize agent behavior by editing this file:

```yaml
my_agent:
  system_prompt: |
    You are an expert at...
    Your task is to...
    Be careful to...
```

### Adding New Tools

1. Create tool in `src/tools/`
2. Register in agent: `self.register_tool("tool_name", tool_function)`
3. Use in agent: `result = self.tools["tool_name"](...)`

## 📊 Output

The pipeline generates a comprehensive report in `data/outputs/`:

```json
{
  "run_id": "20251107_120530",
  "pipeline": {
    "paper_id": "My Paper Title",
    "parsed_paper": {...},
    "found_repo_url": "https://github.com/...",
    "experiment_results": [...],
    "evaluation": {
      "final_reproducibility_score": 0.85,
      "issues_found": [...],
      "recommendations": [...]
    }
  }
}
```

## 🎓 Learning Objectives

1. **Agent Design Patterns**: How to structure autonomous agents
2. **Tool Integration**: Creating and managing agent capabilities
3. **LLM Integration**: Using modern language models in applications
4. **Error Handling**: Robust error management across pipeline stages
5. **Configuration Management**: Environment-based configuration
6. **Testing Strategies**: Testing multi-agent systems
7. **DevOps Concepts**: Environment management, logging, monitoring

## 🔐 Security Considerations

This project includes security features:

- **Sandbox Execution**: Code runs in isolated environments
- **Command Validation**: Forbidden commands are blocked
- **Timeout Protection**: Execution limits prevent infinite loops
- **Environment Isolation**: Virtual environments for each experiment using smart caching

### Virtual Environment Strategy

The system uses a **hybrid approach** for managing Python environments:

- ✅ **Isolation**: Each experiment gets its own `.venv` directory
- ♻️  **Caching**: Identical `requirements.txt` → reuse cached venv (fast!)
- 🚀 **Performance**: First run ~30s, cached runs ~0.1s
- 💾 **Efficiency**: Disk space saved via symlinks

