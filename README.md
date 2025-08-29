

# Resume Parser - Technical Assignment

## Overview
As part of a technical assessment, this project implements a resume parsing system that extracts structured information (name, email, skills) from PDF and DOCX resume files.

## Technical Approach


### Hybrid Architecture
The solution uses a hybrid approach with zero-shot capabilities that combines:
- GLiNER large-v2.1: General entity recognition
- NuNER Zero: Superior multi-word entity handling with entity merging
- LLM-as-a-Judge: For evaluation only (not primary extraction). OpenAI GPT-4o-mini for structured evaluation
- Traditional Validation: Regex patterns and rule-based validation
- Email format validation
- Skills cleaning and deduplication

### Why Not Standalone LLMs?
Based on recent research evidence, standalone LLM approaches have significant limitations for resume parsing:
- Performance Deficits: Traditional NER models can outperform LLMs
- Speed Issues: Traditional methods are faster 
- Cost Concerns: GPT-4o costs per inference vs no API costs for local models
- Behavioral Inconsistency: LLMs show significant performance variations across contexts

## Installation & Usage

```python 
#Installation Python 3.11+
pip install -r requirements.txt

#Usage
#Option 1: Jupyter Notebook (Recommended for Demo)
#Open main.ipynb

#Option 2: Command Line
# Run complete benchmark
python benchmark.py --data_dir ./data --output_dir ./results
```



## Performance Results
- Average Success Rate: 96.7%
- Files Processed: 10
- Average Processing Time: 4.5s per resume
- File Types: PDF (5), DOCX (5), Other (0)
- Detailed results saved to: ./results

## Technical Decisions
- GLiNER over spaCy: Zero-shot capabilities, no training data required
- NuNER Zero addition: Superior multi-word entity extraction ("Machine Learning" vs "Machine" + "Learning")
- Dual model approach: Combines strengths of both models

## Processing Pipeline
1. Document Processing: PyMuPDF (PDFs), python-docx (Word documents)
2. Entity Extraction: Separate passes for basic fields vs skills with different thresholds
3. Entity Merging: Handle multi-word entities from NuNER Zero
4. Validation: Email regex validation, skills cleaning
5. Evaluation: LLM-based quality assessment

## Future Improvements
- Model Optimization: Fine-tune thresholds based on larger dataset
- Additional Fields: Extend to extract education, experience, certifications
- More robust skill extraction pipeline

## Data Sources
- Microsoft Create: https://create.microsoft.com/en-us/search?query=Resume%20health
- Kaggle Dataset was not used because of lack of emails and names. 

## References
- https://arxiv.org/abs/2507.08019
- https://arxiv.org/pdf/2407.19816
- https://arxiv.org/abs/2402.15343
- https://arxiv.org/abs/2311.08526
- https://arxiv.org/abs/2411.15594

## Note
This project is developed for educational and demonstration purposes as part of a technical assignment. 
Perplexity and Claude were used to develop this code. 

