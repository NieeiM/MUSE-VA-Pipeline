"""Command line entry point for the MUSE-VA text distillation pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **_: Any):
        class _Progress:
            def __init__(self, values=None):
                self.values = values or []

            def __iter__(self):
                return iter(self.values)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def update(self, *_args, **_kwargs):
                return None

            def set_postfix(self, *_args, **_kwargs):
                return None

        if iterable is None:
            return _Progress()
        return _Progress(iterable)

from .pipeline.llm_client import LLMConfig
from .pipeline.stage_1_affective_mapping import run_stage_1
from .pipeline.stage_2_music_association import run_stage_2
from .pipeline.stage_3_suno_caption import run_stage_3
from .pipeline.stage_4_narrative_imagery import run_stage_4
from .pipeline.stage_5_consistency_check import run_stage_5


def sample_innovation() -> str:
    value = random.random()
    if value < 0.7:
        return "low"
    if value < 0.9:
        return "medium"
    return "high"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def run_pipeline_instance(
    valence: float,
    arousal: float,
    llm_config: LLMConfig,
    language: str = "en",
    output_dir: Path | str = "logs",
    max_stage_retries: int = 3,
    sleep_time: float = 0,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Run one complete five-stage text distillation instance."""
    run_id = str(uuid.uuid4())
    log_dir = Path(output_dir) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "valence": valence,
        "arousal": arousal,
        "llm_provider": llm_config.provider,
        "llm_model": llm_config.model,
        "language": language,
        "start_time": start_time.isoformat(),
        "status": "RUNNING",
    }
    write_json(log_dir / "metadata.json", metadata)

    try:
        innovation = sample_innovation()

        stage_1 = run_stage_1(
            valence=valence,
            arousal=arousal,
            model_name=llm_config.provider,
            log_dir=str(log_dir),
            language=language,
            random_seed=random_seed,
        )
        sleep(sleep_time)

        stage_2 = run_stage_2(
            theme=stage_1["theme"],
            emotion=stage_1["emotion"],
            valence=valence,
            arousal=arousal,
            model_name=llm_config.provider,
            log_dir=str(log_dir),
            language=language,
            innovation=innovation,
            max_retries=max_stage_retries,
            llm_config=llm_config,
        )
        sleep(sleep_time)

        enhanced_stage_2 = {
            **stage_2,
            **stage_1,
            "valence": valence,
            "arousal": arousal,
            "innovation": innovation,
            "vocal": "no",
        }
        stage_3 = run_stage_3(
            stage_2_output=enhanced_stage_2,
            model_name=llm_config.provider,
            log_dir=str(log_dir),
            language=language,
            max_retries=max_stage_retries,
            llm_config=llm_config,
        )
        sleep(sleep_time)

        enhanced_stage_3 = {**enhanced_stage_2, **stage_3}
        stage_4 = run_stage_4(
            stage_3_output=enhanced_stage_3,
            model_name=llm_config.provider,
            log_dir=str(log_dir),
            language=language,
            max_retries=max_stage_retries,
            llm_config=llm_config,
        )
        sleep(sleep_time)

        enhanced_stage_4 = {**enhanced_stage_3, **stage_4}
        stage_5 = run_stage_5(
            stage_4_output=enhanced_stage_4,
            model_name=llm_config.provider,
            log_dir=str(log_dir),
            language=language,
            max_retries=max_stage_retries,
            llm_config=llm_config,
        )

        consistency = stage_5["consistency"]
        status = "SUCCESS" if consistency["result"] else "INCONSISTENT"
        summary = {
            "run_id": run_id,
            **enhanced_stage_4,
            "consistency_result": consistency["result"],
            "consistency_reason": consistency["reason"],
            "prompt_language": language,
            "llm_provider": llm_config.provider,
            "llm_model": llm_config.model or "",
        }
        if status == "SUCCESS":
            write_csv(log_dir / "summary.csv", summary)
        else:
            write_csv(log_dir / "discarded_summary.csv", summary)

        metadata["status"] = status
        metadata["message"] = consistency["reason"]
    except Exception as exc:
        metadata["status"] = "FAILURE"
        metadata["message"] = str(exc)
    finally:
        end_time = datetime.now()
        metadata["end_time"] = end_time.isoformat()
        metadata["duration_seconds"] = (end_time - start_time).total_seconds()
        write_json(log_dir / "metadata.json", metadata)

    return metadata


def aggregate_summaries(output_dir: Path) -> Path | None:
    rows: list[dict[str, str]] = []
    for summary_path in sorted(output_dir.glob("*/summary.csv")):
        with summary_path.open("r", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))

    if not rows:
        return None

    output_path = output_dir / "global_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def parse_va_pairs(args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.valence is not None or args.arousal is not None:
        if args.valence is None or args.arousal is None:
            raise ValueError("--valence and --arousal must be provided together.")
        return [(round(args.valence, 1), round(args.arousal, 1))]

    rng = random.Random(args.seed)
    return [
        (round(rng.uniform(1.0, 9.0), 1), round(rng.uniform(1.0, 9.0), 1))
        for _ in range(args.num_samples)
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the MUSE-VA five-stage text distillation pipeline."
    )
    parser.add_argument("--num-samples", "-n", type=int, default=1)
    parser.add_argument("--valence", type=float, default=None)
    parser.add_argument("--arousal", type=float, default=None)
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    parser.add_argument(
        "--llm-provider",
        choices=["mock", "gemini", "openai-compatible"],
        default="mock",
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--max-workers", "-w", type=int, default=1)
    parser.add_argument("--stage-retries", type=int, default=3)
    parser.add_argument("--sleep-time", type=float, default=0)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    va_pairs = parse_va_pairs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_config = LLMConfig.from_env(
        provider=args.llm_provider,
        model=args.llm_model,
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
    )

    worker_count = max(1, args.max_workers)
    results: list[dict[str, Any]] = []
    if worker_count == 1:
        iterator = tqdm(va_pairs, desc="Processing", unit="sample")
        for valence, arousal in iterator:
            result = run_pipeline_instance(
                valence=valence,
                arousal=arousal,
                llm_config=llm_config,
                language=args.language,
                output_dir=output_dir,
                max_stage_retries=args.stage_retries,
                sleep_time=args.sleep_time,
                random_seed=args.seed,
            )
            results.append(result)
            iterator.set_postfix(status=result["status"])
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    run_pipeline_instance,
                    valence,
                    arousal,
                    llm_config,
                    args.language,
                    output_dir,
                    args.stage_retries,
                    args.sleep_time,
                    args.seed,
                )
                for valence, arousal in va_pairs
            ]
            with tqdm(total=len(futures), desc="Processing", unit="sample") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.set_postfix(status=result["status"])
                    pbar.update(1)

    aggregate_path = aggregate_summaries(output_dir)
    success = sum(1 for result in results if result["status"] == "SUCCESS")
    inconsistent = sum(1 for result in results if result["status"] == "INCONSISTENT")
    failure = sum(1 for result in results if result["status"] == "FAILURE")
    print(
        f"Done. success={success}, inconsistent={inconsistent}, failure={failure}, "
        f"global_summary={aggregate_path or 'none'}"
    )


if __name__ == "__main__":
    main()
