"""
LLM-based evaluation module for resume parsing assessment.
Uses structured output to provide consistent correct/incorrect/partial evaluations.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from openai import OpenAI

# Import configuration
import config
from utils import truncate_text

# Configure logger
logger = logging.getLogger(__name__)


class ResumeEvaluationJudge:
    """
    LLM-as-a-judge evaluator for resume parsing quality assessment.

    This evaluator uses OpenAI's structured output to provide consistent,
    explainable evaluation of resume parsing results with correct/incorrect/partial
    status for each field plus overall reasoning.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the evaluation judge.

        Parameters
        ----------
        api_key : Optional[str]
            OpenAI API key. If None, tries to get from environment variable.
        """
        # Get API key from parameter or environment
        if api_key is None:
            # api_key = os.getenv(config.LLM_CONFIG["api_key_env"])
            api_key = config.LLM_CONFIG['api_key_env']

        if not api_key:
            raise ValueError(
                f"API key required. Set {config.LLM_CONFIG['api_key_env']} or pass api_key parameter.")

        self.client = OpenAI(api_key=api_key)
        self.model = config.LLM_CONFIG["model"]
        self.temperature = config.LLM_CONFIG["temperature"]
        self.max_tokens = config.LLM_CONFIG["max_tokens"]
        self.timeout = config.LLM_CONFIG.get("timeout", 30)

        logger.info(f"Initialized LLM evaluator with model: {self.model}")

    def evaluate_extraction(self, resume_text: str, extracted_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate the quality of resume parsing extraction.

        Parameters
        ----------
        resume_text : str
            Original resume text content
        extracted_data : Dict[str, Any]
            Parsed resume data with keys: name, email, skills, metadata

        Returns
        -------
        Optional[Dict[str, Any]]
            Structured evaluation results or None if evaluation fails
        """
        try:
            # Truncate resume text to fit within token limits
            truncated_text = truncate_text(resume_text, max_length=1000, suffix="...")

            # Format extracted data as clean JSON
            extracted_json = json.dumps({
                "name": extracted_data.get("name", ""),
                "email": extracted_data.get("email", ""),
                "skills": extracted_data.get("skills", [])
            }, indent=2, ensure_ascii=False)

            # Build user prompt using template from config
            user_prompt = config.EVALUATION_USER_PROMPT.format(
                resume_text=truncated_text,
                extracted_data_json=extracted_json
            )

            logger.debug(f"Sending evaluation request to {self.model}")

            # Make API call with structured output
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": config.EVALUATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": config.EVALUATION_SCHEMA
                },
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )

            # Parse response
            if response.choices and response.choices[0].message.content:
                evaluation_result = json.loads(response.choices[0].message.content)

                # Add metadata about the evaluation
                evaluation_result["evaluation_metadata"] = {
                    "model_used": self.model,
                    "text_length": len(resume_text),
                    "truncated": len(resume_text) > 1000,
                    "extracted_fields_count": len([v for v in [
                        extracted_data.get("name"),
                        extracted_data.get("email"),
                        extracted_data.get("skills")
                    ] if v])
                }

                logger.info(f"Evaluation completed: {evaluation_result.get('extraction_quality', 'unknown')} quality")
                return evaluation_result

            else:
                logger.error("No content in LLM response")
                return None

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return None

    def evaluate_batch(self, resume_data_list: list) -> list:
        """
        Evaluate multiple resume parsing results in batch.

        Parameters
        ----------
        resume_data_list : list
            List of dictionaries, each containing 'resume_text' and 'extracted_data'

        Returns
        -------
        list
            List of evaluation results
        """
        results = []

        for i, resume_data in enumerate(resume_data_list):
            logger.info(f"Evaluating resume {i + 1}/{len(resume_data_list)}")

            try:
                resume_text = resume_data["resume_text"]
                extracted_data = resume_data["extracted_data"]

                evaluation = self.evaluate_extraction(resume_text, extracted_data)

                if evaluation:
                    results.append({
                        "resume_index": i,
                        "file_path": extracted_data.get("metadata", {}).get("file_path", "unknown"),
                        "evaluation": evaluation,
                        "success": True
                    })
                else:
                    results.append({
                        "resume_index": i,
                        "file_path": extracted_data.get("metadata", {}).get("file_path", "unknown"),
                        "evaluation": None,
                        "success": False,
                        "error": "Evaluation failed"
                    })

            except Exception as e:
                logger.error(f"Error evaluating resume {i}: {e}")
                results.append({
                    "resume_index": i,
                    "evaluation": None,
                    "success": False,
                    "error": str(e)
                })

        logger.info(
            f"Batch evaluation completed: {len([r for r in results if r['success']])}/{len(results)} successful")
        return results


def calculate_evaluation_metrics(evaluations: list) -> Dict[str, Any]:
    """
    Calculate performance metrics from evaluation results.

    Parameters
    ----------
    evaluations : list
        List of evaluation results with status fields

    Returns
    -------
    Dict[str, Any]
        Performance metrics including success rates and distributions
    """
    if not evaluations:
        return {"error": "No evaluations provided"}

    # Filter successful evaluations
    successful_evals = [e for e in evaluations if e.get("success") and e.get("evaluation")]

    if not successful_evals:
        return {"error": "No successful evaluations"}

    total = len(successful_evals)

    # Count status occurrences
    name_stats = {"correct": 0, "partial": 0, "incorrect": 0}
    email_stats = {"correct": 0, "partial": 0, "incorrect": 0}
    skills_stats = {"correct": 0, "partial": 0, "incorrect": 0}
    quality_stats = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}

    for eval_data in successful_evals:
        evaluation = eval_data["evaluation"]

        name_stats[evaluation.get("name_status", "incorrect")] += 1
        email_stats[evaluation.get("email_status", "incorrect")] += 1
        skills_stats[evaluation.get("skills_status", "incorrect")] += 1
        quality_stats[evaluation.get("extraction_quality", "poor")] += 1

    # Calculate success rates (correct + partial = success)
    name_success_rate = (name_stats["correct"] + name_stats["partial"]) / total * 100
    email_success_rate = (email_stats["correct"] + email_stats["partial"]) / total * 100
    skills_success_rate = (skills_stats["correct"] + skills_stats["partial"]) / total * 100

    return {
        "total_evaluations": len(evaluations),
        "successful_evaluations": total,
        "success_rate": total / len(evaluations) * 100,

        "name_performance": {
            **name_stats,
            "success_rate": name_success_rate
        },

        "email_performance": {
            **email_stats,
            "success_rate": email_success_rate
        },

        "skills_performance": {
            **skills_stats,
            "success_rate": skills_success_rate
        },

        "overall_quality_distribution": quality_stats,

        "summary": {
            "avg_success_rate": (name_success_rate + email_success_rate + skills_success_rate) / 3,
            "best_field": max(
                [("name", name_success_rate), ("email", email_success_rate), ("skills", skills_success_rate)],
                key=lambda x: x[1]
            )[0],
            "worst_field": min(
                [("name", name_success_rate), ("email", email_success_rate), ("skills", skills_success_rate)],
                key=lambda x: x[1]
            )[0]
        }
    }


# Factory function for easy evaluator creation
def create_evaluator(api_key: Optional[str] = None) -> ResumeEvaluationJudge:
    """
    Factory function to create and return a configured ResumeEvaluationJudge.

    Parameters
    ----------
    api_key : Optional[str]
        OpenAI API key. If None, uses environment variable.

    Returns
    -------
    ResumeEvaluationJudge
        Initialized evaluator instance
    """
    return ResumeEvaluationJudge(api_key=api_key)

