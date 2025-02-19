# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import dataclasses
import json
import logging
import multiprocessing
from functools import partial
from pathlib import Path
from typing import NamedTuple, Optional, Union

import typer
from tqdm.auto import tqdm
from transformers import WEIGHTS_NAME
from transformers.trainer_utils import set_seed

from dem.combinator import CombinatorConfig, TaskOperand
from dem.eq_parser import MissingWeightsInit, TaskVectorArithmetic

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = typer.Typer()

TASK_VECTORS_DIR = "task_vectors"


class ArithmeticRunParams(NamedTuple):
    config: CombinatorConfig
    output_root: str


def _joinpath(path: Union[str, Path], *folders: str) -> Union[str, Path]:
    # We treat this as an s3 path.
    if isinstance(path, str):
        return "/".join([path.rstrip("/"), *folders])

    # This is an OS path
    for folder in folders:
        path /= folder

    return path


def path_exists(path: Union[str, Path]) -> bool:
    return Path(path).exists()


def export_config(config: CombinatorConfig, output_dir: str) -> None:
    export_path = _joinpath(output_dir, "arithmetic_config.json")
    config_json = json.dumps(dataclasses.asdict(config), ensure_ascii=False, indent=2)

    export_path = Path(export_path)
    export_path.parent.mkdir(exist_ok=True, parents=True)
    export_path.write_text(config_json)


def export_vectors(config: CombinatorConfig, task_vector_model: TaskOperand, output_dir: str) -> None:
    task_vectors_path = _joinpath(output_dir, TASK_VECTORS_DIR)
    logger.info("Exporting the new task vectors to %s", task_vectors_path)
    task_vector_model.export(task_vectors_path)
    export_config(config, output_dir)


def extract_vectors(
    config: CombinatorConfig,
    missing_weights_init: MissingWeightsInit = MissingWeightsInit.NO,
    cache_dir: Optional[Path] = None,
) -> TaskOperand:
    logger.debug("Loaded arithmetic config: \n%s", json.dumps(dataclasses.asdict(config), indent=2, ensure_ascii=False))
    arithmetic = TaskVectorArithmetic(config, missing_weights_init=missing_weights_init, cache_dir=cache_dir)
    return arithmetic.evaluate_formula()


def run_arithmetic(
    run_params: ArithmeticRunParams,
    missing_weights_init: MissingWeightsInit = MissingWeightsInit.NO,
    cache_dir: Optional[Path] = None,
) -> str:
    task_vector_model = extract_vectors(run_params.config, missing_weights_init, cache_dir)
    export_vectors(run_params.config, task_vector_model, run_params.output_root)

    return run_params.output_root


@app.command()
def main(
    config_path: Path = typer.Option(...),
    output_dir: str = typer.Option(...),
    cache_dir: Optional[Path] = Path("transformers_cache"),
    missing_weights_init: MissingWeightsInit = MissingWeightsInit.NO,
    seed: int = 42,
    n_processes: int = 1,
) -> None:
    assert n_processes > 0, "Number of processes should be at least 1."
    set_seed(seed)

    # Read the json, if there are multiple rows (i.e., it's a list),
    # then we have multiple configurations to run. Both formats are allowed for legacy purposes.
    config_text_dict = json.loads(config_path.read_text())
    if isinstance(config_text_dict, list):
        arithmetic_runs = config_text_dict
    else:
        arithmetic_runs = [config_text_dict]

    arithmetic_operations = []
    combination_ids = set()
    for run_config in arithmetic_runs:
        # Legacy configs do not have the combination_id.
        base_config = CombinatorConfig.from_dict(run_config)
        base_root = _joinpath(output_dir, base_config.combination_id)
        export_config(base_config, base_root)
        assert (
            base_config.combination_id not in combination_ids
        ), f"More than one formula detected with `combination_id`: '{base_config.combination_id}'."
        combination_ids.add(base_config.combination_id)

        for conf_idx, (experiment_name, config) in enumerate(base_config.next_formula(), start=1):
            output_root = _joinpath(base_root, experiment_name)
            model_path = _joinpath(output_root, "task_vectors", WEIGHTS_NAME)
            if path_exists(model_path):
                logger.info(
                    "Vectors for config #%d: %s already exist. Skipping....",
                    conf_idx,
                    config_path.name + " " + experiment_name,
                )
                continue
            logger.info(
                "Scheduling the vectors for config #%d: %s/%s", conf_idx, base_config.combination_id, experiment_name
            )
            arithmetic_operations.append(ArithmeticRunParams(output_root=output_root, config=config))

    f_arithmetic = partial(run_arithmetic, missing_weights_init=missing_weights_init, cache_dir=cache_dir)
    with multiprocessing.Pool(processes=n_processes) as pool:
        for experiment_root_dir in tqdm(
            pool.imap_unordered(f_arithmetic, arithmetic_operations),
            leave=True,
            position=0,
            total=len(arithmetic_operations),
        ):
            logger.info("Processing vectors: %s", experiment_root_dir)


if __name__ == "__main__":
    app()
