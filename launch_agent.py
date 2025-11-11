import os
from pathlib import Path

# Find PDF files in inputs directory
inputs_dir = Path("./data/inputs")
pdf_files = list(inputs_dir.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF files found in ./data/inputs/")

# Use the first PDF found
pdf_path = str(pdf_files[0])

print(f"Using PDF: {pdf_path}")

from scientist.main import run_reproducibility_pipeline

result = run_reproducibility_pipeline(pdf_path=pdf_path, output_report=True)
