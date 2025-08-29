"""
Configuration file for Resume Parser System.
All parameters, thresholds, prompts, and settings are defined here.
"""

import os
from pathlib import Path

# =============================================================================
# DIRECTORY SETTINGS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# =============================================================================
# MODEL SETTINGS
# =============================================================================
MODELS = {
    "gliner": {
        "model_name": "urchade/gliner_large-v2.1",
        "enabled": True
    },
    "nuner": {
        "model_name": "numind/NuNerZero",
        "enabled": True,
        "requires_lowercase": True  # NuNER Zero requires lowercase labels
    }
}

# =============================================================================
# EXTRACTION SETTINGS
# =============================================================================
# Whether to run separate extractions for different entity types
USE_SEPARATE_EXTRACTIONS = True

# Entity labels for different extraction passes
ENTITY_LABELS = {
    "basic": ["person", "email"],
    "skills": ["skill", "technology", "tool", "programming language"]
}

# Thresholds for different entity types
ENTITY_THRESHOLDS = {
    "person": 0.4,
    "email": 0.3,
    "skill": 0.2,
    "technology": 0.2,
    "tool": 0.2,
    "programming language": 0.2
}

# Fallback settings
FALLBACK_SETTINGS = {
    "min_skills_threshold": 2,  # Minimum skills to consider successful
    "lower_threshold_fallback": 0.1,  # Emergency lower threshold
    "max_text_length": 5000  # Max characters to process
}

# =============================================================================
# ENTITY NORMALIZATION MAPPING
# =============================================================================
LABEL_NORMALIZATION = {
    # Name entities
    "person": "name",

    # Email entities
    "email": "email",

    # Skill entities
    "programming language": "skill",
    "tool": "skill",
    "technology": "skill",
    "skill": "skill"
}

# =============================================================================
# EMAIL VALIDATION SETTINGS
# =============================================================================
EMAIL_VALIDATION = {
    "must_contain_at": True,
    "must_contain_dot": True,
    "min_length": 5
}

# =============================================================================
# LLM EVALUATION SETTINGS
# =============================================================================
LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "api_key_env": "",  # Environment variable name
    "temperature": 0.1,
    "max_tokens": 1000,
    "timeout": 30
}

# Structured schema for consistent evaluation output
EVALUATION_SCHEMA = {
    "name": "resume_parsing_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "name_status": {
                "type": "string",
                "enum": ["correct", "incorrect", "partial"],
                "description": "Status of name extraction"
            },
            "email_status": {
                "type": "string",
                "enum": ["correct", "incorrect", "partial"],
                "description": "Status of email extraction"
            },
            "skills_status": {
                "type": "string",
                "enum": ["correct", "incorrect", "partial"],
                "description": "Status of skills extraction"
            },
            "overall_reasoning": {
                "type": "string",
                "description": "Brief explanation of assessment for all fields"
            },
            "extraction_quality": {
                "type": "string",
                "enum": ["excellent", "good", "fair", "poor"],
                "description": "Overall quality assessment"
            }
        },
        "required": [
            "name_status", "email_status", "skills_status",
            "overall_reasoning", "extraction_quality"
        ],
        "additionalProperties": False
    }
}

# System prompt for LLM evaluation
EVALUATION_SYSTEM_PROMPT = """You are an expert resume parsing evaluator. Your job is to assess how well a resume parser extracted key information from a resume.

Evaluate based on these criteria:
- NAME: Is the full name correctly identified? Check for completeness and accuracy.
- EMAIL: Is the email address correctly extracted? Check for format and completeness.
- SKILLS: Are the technical and professional skills comprehensively identified? 
  Look for programming languages, frameworks, tools, certifications, soft skills.

For each field, determine if the extraction is:
- "correct": Perfectly extracted and accurate
- "incorrect": Missing, wrong, or completely inaccurate  
- "partial": Extracted but incomplete or has minor issues

Be thorough but fair in your assessment. Consider that some resumes may have formatting challenges."""

# User prompt template for evaluation
EVALUATION_USER_PROMPT = """Evaluate this resume parsing extraction:

ORIGINAL RESUME TEXT (first 1000 chars):
{resume_text}

EXTRACTED DATA:
{extracted_data_json}

Provide your assessment focusing on accuracy and completeness."""

# =============================================================================
# FILE PROCESSING SETTINGS
# =============================================================================
SUPPORTED_FORMATS = [".pdf", ".docx", ".doc"]

# =============================================================================
# LOGGING SETTINGS
# =============================================================================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_handler": True,
    "console_handler": True,
    "log_file": LOGS_DIR / "resume_parser.log"
}

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================
OUTPUT_CONFIG = {
    "save_individual_results": True,
    "save_aggregate_results": True,
    "save_detailed_logs": True,
    "output_format": "json",  # json, csv, both
    "include_metadata": True
}

# =============================================================================
# BENCHMARK SETTINGS
# =============================================================================
BENCHMARK_CONFIG = {
    "run_individual_evaluations": True,
    "run_combined_evaluation": True,
    "compare_models": True,
    "generate_summary_report": True,
    "include_performance_metrics": True
}


# =============================================================================
# ENTITY MERGING FUNCTION (for NuNER Zero)
# =============================================================================
def merge_entities_config():
    """Configuration for entity merging function used with NuNER Zero"""
    return {
        "merge_adjacent": True,
        "merge_same_label": True,
        "max_gap": 1  # Maximum character gap to merge entities
    }
