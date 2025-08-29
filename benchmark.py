"""
Resume Parser Benchmark Runner
Main script that processes resume files, evaluates results, and generates comprehensive reports.
"""

import logging
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import config
from utils import (
    setup_logging,
    list_resume_files,
    save_results,
)
from parser import create_parser
from evaluator import create_evaluator, calculate_evaluation_metrics

logger = logging.getLogger(__name__)


class ResumeParserBenchmark:
    """
    Comprehensive benchmark runner for resume parsing system.

    This class orchestrates the entire pipeline:
    1. Load and process resume files
    2. Extract structured data using GLiNER + NuNER Zero
    3. Evaluate results using LLM-as-a-judge
    4. Generate performance metrics and reports
    """

    def __init__(self, api_key: Optional[str] = None, output_dir: Optional[Path] = None):
        """
        Initialize benchmark runner.

        Parameters
        ----------
        api_key : Optional[str]
            OpenAI API key for evaluation
        output_dir : Optional[Path]
            Output directory for results
        """
        # Setup logging
        setup_logging(config.LOGGING_CONFIG)

        # Initialize components
        self.parser = create_parser()
        self.evaluator = create_evaluator(api_key) if api_key else None

        # Setup output directory
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Benchmark initialized - Output: {self.output_dir}")

    def process_resume_files(self, data_dir: Path) -> List[Dict[str, Any]]:
        """
        Process all resume files in the data directory.

        Parameters
        ----------
        data_dir : Path
            Directory containing resume files

        Returns
        -------
        List[Dict[str, Any]]
            List of processing results
        """
        # Find resume files
        resume_files = list_resume_files(data_dir, config.SUPPORTED_FORMATS)

        if not resume_files:
            logger.warning(f"No resume files found in {data_dir}")
            return []

        logger.info(f"Found {len(resume_files)} resume files to process")

        # Process each file
        results = []
        for i, file_path in enumerate(resume_files, 1):
            logger.info(f"Processing {i}/{len(resume_files)}: {file_path.name}")

            try:
                # Parse resume
                start_time = time.time()
                parsed_data = self.parser.parse_resume_file(file_path)
                processing_time = time.time() - start_time

                # Extract original text for evaluation
                original_text = self.parser.extract_text_from_file(file_path)

                result = {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_format": file_path.suffix.lower(),
                    "parsed_data": parsed_data,
                    "original_text": original_text,
                    "processing_time": processing_time,
                    "success": parsed_data["metadata"].get("processing_successful", False)
                }

                results.append(result)

                # Log parsing results
                name_status = "y" if parsed_data["name"] else "x"
                email_status = "y" if parsed_data["email"] else "x"
                skills_count = len(parsed_data["skills"])

                logger.info(
                    f" Name: {name_status}, Email: {email_status}, Skills: {skills_count}, Time: {processing_time:.2f}s")

                # Save individual result if configured
                if config.OUTPUT_CONFIG["save_individual_results"]:
                    individual_output = self.output_dir / f"{file_path.stem}_parsed_{self.timestamp}.json"
                    save_results(parsed_data, individual_output, config.OUTPUT_CONFIG["output_format"])

            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                results.append({
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "error": str(e),
                    "success": False
                })

        return results

    def evaluate_results(self, processing_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate parsing results using LLM-as-a-judge.

        Parameters
        ----------
        processing_results : List[Dict[str, Any]]
            Results from resume processing

        Returns
        -------
        List[Dict[str, Any]]
            Evaluation results
        """
        if not self.evaluator:
            logger.warning("No evaluator available - skipping evaluation")
            return []

        # Prepare evaluation inputs
        eval_inputs = []
        for result in processing_results:
            if result.get("success") and result.get("original_text"):
                eval_inputs.append({
                    "resume_text": result["original_text"],
                    "extracted_data": result["parsed_data"],
                    "file_info": {
                        "file_name": result["file_name"],
                        "file_path": result["file_path"]
                    }
                })

        if not eval_inputs:
            logger.warning("No successful parsing results to evaluate")
            return []

        logger.info(f"Evaluating {len(eval_inputs)} successfully parsed resumes")

        # Run batch evaluation
        evaluation_results = []
        for i, eval_input in enumerate(eval_inputs, 1):
            logger.info(f"Evaluating {i}/{len(eval_inputs)}: {eval_input['file_info']['file_name']}")

            try:
                evaluation = self.evaluator.evaluate_extraction(
                    eval_input["resume_text"],
                    eval_input["extracted_data"]
                )

                if evaluation:
                    evaluation_results.append({
                        "file_info": eval_input["file_info"],
                        "evaluation": evaluation,
                        "success": True
                    })

                    # Log evaluation summary
                    quality = evaluation.get("extraction_quality", "unknown")
                    name_status = evaluation.get("name_status", "unknown")
                    email_status = evaluation.get("email_status", "unknown")
                    skills_status = evaluation.get("skills_status", "unknown")

                    logger.info(
                        f" Quality: {quality}, Name: {name_status}, Email: {email_status}, Skills: {skills_status}")

                else:
                    evaluation_results.append({
                        "file_info": eval_input["file_info"],
                        "evaluation": None,
                        "success": False,
                        "error": "Evaluation failed"
                    })

            except Exception as e:
                logger.error(f"Evaluation failed for {eval_input['file_info']['file_name']}: {e}")
                evaluation_results.append({
                    "file_info": eval_input["file_info"],
                    "evaluation": None,
                    "success": False,
                    "error": str(e)
                })

        return evaluation_results

    def generate_reports(self, processing_results: List[Dict[str, Any]],
                         evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark reports.

        Parameters
        ----------
        processing_results : List[Dict[str, Any]]
            Processing results
        evaluation_results : List[Dict[str, Any]]
            Evaluation results

        Returns
        -------
        Dict[str, Any]
            Comprehensive report data
        """
        logger.info("Generating benchmark reports")

        # Calculate processing metrics
        total_files = len(processing_results)
        successful_parsing = len([r for r in processing_results if r.get("success")])
        avg_processing_time = sum([r.get("processing_time", 0) for r in processing_results if
                                   r.get("processing_time")]) / total_files if total_files > 0 else 0

        # Calculate evaluation metrics
        evaluation_metrics = calculate_evaluation_metrics(evaluation_results) if evaluation_results else {}

        # Build comprehensive report
        report = {
            "benchmark_info": {
                "timestamp": self.timestamp,
                "total_files_processed": total_files,
                "successful_parsing": successful_parsing,
                "parsing_success_rate": (successful_parsing / total_files * 100) if total_files > 0 else 0,
                "avg_processing_time_seconds": round(avg_processing_time, 3)
            },
            "file_breakdown": {
                "pdf_files": len([r for r in processing_results if r.get("file_format") == ".pdf"]),
                "docx_files": len([r for r in processing_results if r.get("file_format") == ".docx"]),
                "other_files": len([r for r in processing_results if r.get("file_format") not in [".pdf", ".docx"]])
            },
            "parsing_performance": {
                "files_with_names": len([r for r in processing_results if r.get("parsed_data", {}).get("name")]),
                "files_with_emails": len([r for r in processing_results if r.get("parsed_data", {}).get("email")]),
                "files_with_skills": len([r for r in processing_results if r.get("parsed_data", {}).get("skills")]),
                "avg_skills_per_resume": sum([len(r.get("parsed_data", {}).get("skills", [])) for r in
                                              processing_results]) / total_files if total_files > 0 else 0
            },
            "evaluation_metrics": evaluation_metrics,
            "individual_results": []
        }

        # Add individual file results summary
        for proc_result in processing_results:
            file_summary = {
                "file_name": proc_result.get("file_name"),
                "file_format": proc_result.get("file_format"),
                "parsing_success": proc_result.get("success", False),
                "processing_time": proc_result.get("processing_time"),
                "extracted_fields": {}
            }

            if proc_result.get("parsed_data"):
                parsed_data = proc_result["parsed_data"]
                file_summary["extracted_fields"] = {
                    "name": parsed_data.get("name", ""),
                    "email": parsed_data.get("email", ""),
                    "skills_count": len(parsed_data.get("skills", [])),
                    "skills": parsed_data.get("skills", [])
                }

            # Add evaluation data if available
            eval_result = next(
                (er for er in evaluation_results if
                 er.get("file_info", {}).get("file_name") == proc_result.get("file_name")),
                None
            )

            if eval_result and eval_result.get("evaluation"):
                file_summary["evaluation"] = eval_result["evaluation"]

            report["individual_results"].append(file_summary)

        return report

    def run_benchmark(self, data_dir: Path) -> Dict[str, Any]:
        """
        Run complete benchmark pipeline.

        Parameters
        ----------
        data_dir : Path
            Directory containing resume files

        Returns
        -------
        Dict[str, Any]
            Complete benchmark results
        """
        logger.info(f"Starting resume parsing benchmark on {data_dir}")
        start_time = time.time()

        try:
            # Process resume files
            logger.info("=== STEP 1: Processing Resume Files ===")
            processing_results = self.process_resume_files(data_dir)

            # Evaluate results
            evaluation_results = []
            if self.evaluator and config.BENCHMARK_CONFIG["run_individual_evaluations"]:
                logger.info("=== STEP 2: Evaluating Results ===")
                evaluation_results = self.evaluate_results(processing_results)
            else:
                logger.info("=== STEP 2: Skipping Evaluation (no API key) ===")

            # Generate reports
            logger.info("=== STEP 3: Generating Reports ===")
            report = self.generate_reports(processing_results, evaluation_results)

            # Add total benchmark time
            total_time = time.time() - start_time
            report["benchmark_info"]["total_runtime_seconds"] = round(total_time, 3)

            # Save results
            if config.OUTPUT_CONFIG["save_aggregate_results"]:
                logger.info("=== STEP 4: Saving Results ===")

                # Save comprehensive report
                report_path = self.output_dir / f"benchmark_report_{self.timestamp}.json"
                save_results(report, report_path, "json")

                # Save evaluation results separately if available
                if evaluation_results:
                    eval_path = self.output_dir / f"evaluation_results_{self.timestamp}.json"
                    save_results(evaluation_results, eval_path, "json")

                logger.info(f"Results saved to {self.output_dir}")

            logger.info(f" Benchmark completed successfully in {total_time:.2f} seconds")
            return report

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise

    def print_summary(self, report: Dict[str, Any]) -> None:
        """
        Print human-readable benchmark summary.

        Parameters
        ----------
        report : Dict[str, Any]
            Benchmark report data
        """
        print("\n" + "=" * 60)
        print("RESUME PARSER BENCHMARK SUMMARY")
        print("=" * 60)

        # Basic info
        benchmark_info = report.get("benchmark_info", {})
        print(f" Files Processed: {benchmark_info.get('total_files_processed', 0)}")
        print(
            f" Successful Parsing: {benchmark_info.get('successful_parsing', 0)} ({benchmark_info.get('parsing_success_rate', 0):.1f}%)")
        print(f"Average Processing Time: {benchmark_info.get('avg_processing_time_seconds', 0):.3f}s")
        print(f" Total Runtime: {benchmark_info.get('total_runtime_seconds', 0):.2f}s")

        # File breakdown
        file_breakdown = report.get("file_breakdown", {})
        print(f"\n File Types:")
        print(f"   PDF: {file_breakdown.get('pdf_files', 0)}")
        print(f"   DOCX: {file_breakdown.get('docx_files', 0)}")
        print(f"   Other: {file_breakdown.get('other_files', 0)}")

        # Parsing performance
        parsing_perf = report.get("parsing_performance", {})
        print(f"\n Parsing Performance:")
        print(f"   Names Extracted: {parsing_perf.get('files_with_names', 0)}")
        print(f"   Emails Extracted: {parsing_perf.get('files_with_emails', 0)}")
        print(f"   Files with Skills: {parsing_perf.get('files_with_skills', 0)}")
        print(f"   Avg Skills per Resume: {parsing_perf.get('avg_skills_per_resume', 0):.1f}")

        # Evaluation metrics
        eval_metrics = report.get("evaluation_metrics", {})
        if eval_metrics and not eval_metrics.get("error"):
            print(f"\nEvaluation Results:")
            print(f"   Total Evaluations: {eval_metrics.get('total_evaluations', 0)}")
            print(f"   Successful Evaluations: {eval_metrics.get('successful_evaluations', 0)}")

            name_perf = eval_metrics.get("name_performance", {})
            email_perf = eval_metrics.get("email_performance", {})
            skills_perf = eval_metrics.get("skills_performance", {})

            print(f"   Name Success Rate: {name_perf.get('success_rate', 0):.1f}%")
            print(f"   Email Success Rate: {email_perf.get('success_rate', 0):.1f}%")
            print(f"   Skills Success Rate: {skills_perf.get('success_rate', 0):.1f}%")

            summary = eval_metrics.get("summary", {})
            print(f"   Average Success Rate: {summary.get('avg_success_rate', 0):.1f}%")
            print(f"   Best Performing Field: {summary.get('best_field', 'N/A')}")
            print(f"   Needs Improvement: {summary.get('worst_field', 'N/A')}")

        print("=" * 60)


def main():
    """Main function for command-line execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Resume Parser Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data_dir ./data --api_key sk-xxx
  python main.py --data_dir ./resumes --output_dir ./results
  python main.py --help
        """
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(config.DATA_DIR),
        help="Directory containing resume files (PDF/DOCX)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Directory to save benchmark results"
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default=config.LLM_CONFIG['api_key_env'],
        help="OpenAI API key for evaluation (or set OPENAI_API_KEY env var)"
    )

    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip LLM evaluation (parsing only)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get API key

    api_key = args.api_key
    if args.skip_evaluation:
        api_key = None

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory does not exist: {data_dir}")
        sys.exit(1)

    # Run benchmark
    try:
        benchmark = ResumeParserBenchmark(
            api_key=api_key,
            output_dir=Path(args.output_dir)
        )

        report = benchmark.run_benchmark(data_dir)
        benchmark.print_summary(report)

        print(f"Detailed results saved to: {benchmark.output_dir}")

    except KeyboardInterrupt:
        print("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
