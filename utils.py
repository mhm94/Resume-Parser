"""
Utility functions for Resume Parser System.
Contains helper functions for file I/O, validation, entity processing, and text manipulation.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from collections import defaultdict

# Configure module logger
logger = logging.getLogger(__name__)


def setup_logging(config: Dict) -> None:
    """
    Configure logging based on config settings.

    Parameters
    ----------
    config : Dict
        Logging configuration dictionary
    """
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_level = getattr(logging, config.get("level", "INFO").upper())

    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handlers = []

    # Console handler
    if config.get("console_handler", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)

    # File handler
    if config.get("file_handler", False) and config.get("log_file"):
        file_handler = logging.FileHandler(config["log_file"])
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True
    )


def list_resume_files(data_dir: Union[str, Path], supported_formats: List[str]) -> List[Path]:
    """
    Find all resume files in data directory with supported formats.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Path to directory containing resume files
    supported_formats : List[str]
        List of supported file extensions (e.g., ['.pdf', '.docx'])

    Returns
    -------
    List[Path]
        List of Path objects for found resume files, sorted alphabetically
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_path}")
        return []

    if not data_path.is_dir():
        logger.warning(f"Data path is not a directory: {data_path}")
        return []

    resume_files = []
    for file_path in data_path.iterdir():
        if (file_path.is_file() and
                file_path.suffix.lower() in supported_formats):
            resume_files.append(file_path)

    resume_files.sort()  # Sort alphabetically for consistent processing order

    logger.info(f"Found {len(resume_files)} resume files in {data_path}")
    for file_path in resume_files:
        logger.debug(f"  - {file_path.name}")

    return resume_files


def validate_email(email: str, validation_rules: Dict) -> bool:
    """
    Validate email address using basic sanity checks.

    Parameters
    ----------
    email : str
        Email address to validate
    validation_rules : Dict
        Validation configuration with rules

    Returns
    -------
    bool
        True if email passes validation, False otherwise
    """
    if not email or not isinstance(email, str):
        return False

    email = email.strip()

    # Check minimum length
    min_length = validation_rules.get("min_length", 5)
    if len(email) < min_length:
        return False

    # Check for @ symbol
    if validation_rules.get("must_contain_at", True):
        if email.count("@") != 1:
            return False

    # Check for dot after @
    if validation_rules.get("must_contain_dot", True):
        if "@" in email:
            domain_part = email.split("@")[1]
            if "." not in domain_part:
                return False
        else:
            # If no @, check if . exists at all
            if "." not in email:
                return False

    return True


def merge_adjacent_entities(entities: List[Dict], text: str, max_gap: int = 1) -> List[Dict]:
    """
    Merge adjacent entities of the same label to handle multi-word entities.

    This is particularly useful for NuNER Zero which may split multi-word entities
    like "Machine Learning" into separate entities.

    Parameters
    ----------
    entities : List[Dict]
        List of entity dictionaries with 'start', 'end', 'label', 'text' keys
    text : str
        Original text used to reconstruct merged entity text
    max_gap : int, default=1
        Maximum character gap between entities to consider for merging

    Returns
    -------
    List[Dict]
        List of merged entities
    """
    if not entities:
        return []

    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda x: x.get('start', 0))

    merged = []
    current = sorted_entities[0].copy()

    for next_entity in sorted_entities[1:]:
        # Check if entities can be merged
        same_label = next_entity['label'] == current['label']
        adjacent = (next_entity['start'] - current['end']) <= max_gap

        if same_label and adjacent:
            # Merge entities
            current['end'] = next_entity['end']
            current['text'] = text[current['start']:current['end']].strip()

            # Update confidence score (take average)
            if 'score' in current and 'score' in next_entity:
                current['score'] = (current['score'] + next_entity['score']) / 2
        else:
            # Cannot merge, append current and move to next
            merged.append(current)
            current = next_entity.copy()

    # Append the last entity
    merged.append(current)

    return merged


def normalize_entities(entities: List[Dict], label_mapping: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Normalize entity labels and group by canonical label.

    Parameters
    ----------
    entities : List[Dict]
        List of entity dictionaries with 'label' and 'text' keys
    label_mapping : Dict[str, str]
        Mapping from original labels to canonical labels

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping canonical labels to lists of entity texts
    """
    normalized = defaultdict(list)

    for entity in entities:
        original_label = entity.get('label', '').lower()
        entity_text = entity.get('text', '').strip()

        if not entity_text:
            continue

        # Map to canonical label
        canonical_label = label_mapping.get(original_label, original_label)
        normalized[canonical_label].append(entity_text)

    # Convert to regular dict and remove duplicates
    result = {}
    for label, texts in normalized.items():
        # Remove duplicates while preserving order
        seen = set()
        unique_texts = []
        for text in texts:
            text_lower = text.lower()
            if text_lower not in seen:
                seen.add(text_lower)
                unique_texts.append(text)
        result[label] = unique_texts

    return result


def clean_skill_text(skill: str, exclude_words: List[str] = None) -> Optional[str]:
    """
    Clean and validate skill text.

    Parameters
    ----------
    skill : str
        Raw skill text
    exclude_words : List[str], optional
        Words that invalidate a skill if present

    Returns
    -------
    Optional[str]
        Cleaned skill text or None if invalid
    """
    if not skill or not isinstance(skill, str):
        return None

    # Basic cleaning
    skill = skill.strip()
    skill = re.sub(r'[^\w\s\+\-\.\#]', '', skill)  # Keep common tech characters
    skill = re.sub(r'\s+', ' ', skill)  # Normalize whitespace

    if len(skill) < 2 or len(skill) > 30:
        return None

    # Check for exclude words
    if exclude_words:
        skill_lower = skill.lower()
        for exclude_word in exclude_words:
            if exclude_word in skill_lower:
                return None

    return skill


def deduplicate_skills(skills: List[str]) -> List[str]:
    """
    Remove duplicate skills while preserving order.

    Parameters
    ----------
    skills : List[str]
        List of skill strings (may contain duplicates)

    Returns
    -------
    List[str]
        Deduplicated list of skills
    """
    seen = set()
    unique_skills = []

    for skill in skills:
        skill_lower = skill.lower().strip()
        if skill_lower and skill_lower not in seen:
            seen.add(skill_lower)
            unique_skills.append(skill.strip())

    return unique_skills


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with optional suffix.

    Parameters
    ----------
    text : str
        Text to truncate
    max_length : int
        Maximum length including suffix
    suffix : str, default="..."
        Suffix to add to truncated text

    Returns
    -------
    str
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    truncate_length = max_length - len(suffix)
    if truncate_length < 0:
        return text[:max_length]

    return text[:truncate_length] + suffix


def save_results(results: Any, output_path: Union[str, Path],
                 format_type: str = "json") -> bool:
    """
    Save results to file in specified format.

    Parameters
    ----------
    results : Any
        Results data to save
    output_path : Union[str, Path]
        Path where to save results
    format_type : str, default="json"
        Output format: "json" or "csv"

    Returns
    -------
    bool
        True if saved successfully, False otherwise
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format_type.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        elif format_type.lower() == "csv":
            # For CSV, results should be a list of dictionaries
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False, encoding='utf-8')

        else:
            logger.error(f"Unsupported format: {format_type}")
            return False

        logger.info(f"Results saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")
        return False

