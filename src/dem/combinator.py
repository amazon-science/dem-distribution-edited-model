# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import copy
import itertools
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from itertools import chain, cycle, product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

from dem.enums import MissingWeightsInit

MAX_SHARD_SIZE_SAFETENSORS = "5GB"

logger = logging.getLogger(__name__)


class ConstantType(str, Enum):
    FIXED = "fixed"
    SAMPLED = "sampled"
    RANGE = "range"
    NORM_SAMPLE = "norm_sample"


@dataclass
class TaskOperand:
    state_dict: Dict[str, torch.Tensor]

    config: Optional[dict] = None

    def config_to_json(self) -> str:
        if self.config is None:
            return "{}"

        return json.dumps(self.config, ensure_ascii=False, indent=2)

    def _export_local_fs(self, output_dir: Path) -> None:
        output_dir.mkdir(exist_ok=True, parents=True)
        output_dir.joinpath("config.json").write_text(self.config_to_json())
        state_dict_split = split_torch_state_dict_into_shards(
            self.state_dict,
            filename_pattern=SAFE_WEIGHTS_NAME.replace(".safetensors", "{suffix}.safetensors"),
            max_shard_size=MAX_SHARD_SIZE_SAFETENSORS,
        )

        # Save index if sharded
        index = None
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }

        # Save the model
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        for shard_file, tensors in filename_to_tensors:
            shard = {}
            for tensor in tensors:
                shard[tensor] = self.state_dict[tensor].contiguous()
                # delete reference, see https://github.com/huggingface/transformers/pull/34890
                del self.state_dict[tensor]

            save_file(shard, output_dir / shard_file, metadata={"format": "pt"})

        if index is not None:
            save_index_file = output_dir / SAFE_WEIGHTS_INDEX_NAME
            save_index_file.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({MAX_SHARD_SIZE_SAFETENSORS}) and is going to be "
                f"split in {len(state_dict_split.filename_to_tensors)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

    def export(self, output_dir: str) -> None:
        self._export_local_fs(Path(output_dir))


@dataclass
class Operand:
    operand_id: str


@dataclass
class ModelDefinition(Operand):
    path: Union[str, Path]

    weight: float = 1.0

    def _read_shard_files_from_weight_index(self, weight_index_path: Path) -> List[str]:
        shard_files = list(set(json.loads(weight_index_path.read_text())["weight_map"].values()))

        return shard_files

    def load_model(self: ModelDefinition, cache_dir: Optional[Path] = None) -> TaskOperand:
        model_name_or_path = Path(self.path)
        model_checkpoint = model_name_or_path

        shard_files = None
        if model_checkpoint.joinpath(WEIGHTS_INDEX_NAME).exists():
            shard_files = self._read_shard_files_from_weight_index(model_checkpoint.joinpath(WEIGHTS_INDEX_NAME))
        elif model_checkpoint.joinpath(SAFE_WEIGHTS_INDEX_NAME).exists():
            shard_files = self._read_shard_files_from_weight_index(model_checkpoint.joinpath(SAFE_WEIGHTS_INDEX_NAME))
        elif model_checkpoint.joinpath(WEIGHTS_NAME).exists():
            shard_files = [WEIGHTS_NAME]

        elif model_checkpoint.joinpath(SAFE_WEIGHTS_NAME).exists():
            shard_files = [SAFE_WEIGHTS_NAME]

        if shard_files:
            state_dict = {}
            for shard in shard_files:
                shard_path = model_name_or_path.joinpath(shard)
                if shard.endswith(".safetensors"):
                    state_dict.update(load_file(shard_path, device="cpu"))
                else:
                    state_dict.update(torch.load(shard_path, map_location="cpu"))
        else:
            # Load the model from HF Hub otherwise
            state_dict = (
                AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir).cpu().state_dict()
            )

        local_config_path = model_name_or_path / "config.json"
        config = (
            AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir).to_dict()
            if not local_config_path.exists()
            else json.loads(local_config_path.read_text())
        )
        model = TaskOperand(config=config, state_dict=state_dict)

        model.config["operand_id"] = self.operand_id
        return model


@dataclass
class ConstantDefinition(Operand):
    constant_type: ConstantType

    values: Optional[List[float]] = None

    min_value: float = 0.0

    max_value: float = 0.0

    n_values: int = 0

    def __post_init__(self):
        assert self.min_value <= self.max_value, (
            f"Failed to initialize the constant. "
            f"Min value ({self.min_value}) is greater than the max value ({self.max_value})"
        )
        if self.constant_type == ConstantType.FIXED:
            self.n_values = len(self.values)
            self.min_value = min(self.values)
            self.max_value = max(self.values)
        elif self.constant_type == ConstantType.RANGE:
            self.values = list(np.linspace(self.min_value, self.max_value, num=self.n_values).astype(float))
        elif self.constant_type == ConstantType.NORM_SAMPLE:
            sample_f = partial(np.random.choice, a=self.values, size=self.n_values, replace=True)

            def _norm_sample() -> Iterable[float]:
                arr = sample_f().astype(float)
                arr = arr / arr.sum()
                return arr.tolist()

            # Otherwise we sample on the fly
            self.generator = iter(_norm_sample, 1.0)
            # [() for _ in range(self.num_samples)])
            self.values = None

        elif self.constant_type == ConstantType.SAMPLED:
            # Sample from pre-defined values
            sampler = (
                partial(np.random.choice, a=self.values, replace=False)
                if self.values
                else partial(np.random.uniform, low=self.min_value, high=self.max_value)
            )

            # If `n_values` is set we sample all the values
            if self.n_values > 0:
                self.values = list(sampler(size=self.n_values).astype(float))
            else:
                self.values = None
                # Otherwise we sample on the fly
                self.generator = iter(lambda: float(sampler()), 1.0)

        # We make a cyclic iterator if there are some values set
        if self.values:
            self.generator = cycle(map(float, self.values))

    def next_value(self) -> Union[float, Iterable[float]]:
        return next(self.generator)


@dataclass
class CombinatorConfig:
    formula: str

    models: List[ModelDefinition]

    constants: List[ConstantDefinition] = field(default_factory=list)

    missing_weights_init: MissingWeightsInit = MissingWeightsInit.NO

    combination_id: str = ""

    def __post_init__(self):
        self.model_vars: Dict[str, Operand] = {}

        for model in chain(self.models):
            id_ = model.operand_id
            if id_ in self.model_vars:
                raise KeyError(f"Model with duplicate operand_id {id_} ({model.path}) {self.model_vars}.")
            self.model_vars[id_] = model

    def next_formula(self) -> Iterable[Tuple[str, CombinatorConfig]]:
        random_constants = {}
        keys = []
        values_to_combine = []
        for constant in self.constants:
            if not constant.values:
                random_constants[constant.operand_id] = constant
                continue
            keys.append(constant.operand_id)
            values_to_combine.append(constant.values)

        def _flatten_constants(
            constant_id: str, constant_definition: ConstantDefinition
        ) -> Iterable[Tuple[str, float]]:
            constant_values = constant_definition.next_value()
            if isinstance(constant_values, list):
                for idx, constant_value in enumerate(constant_values):
                    yield constant_id + str(idx + 1), constant_value
            else:
                yield constant_id, constant_values

        for combination in product(*values_to_combine):
            new_config = copy.deepcopy(self)
            new_config.constants.clear()
            name = []

            rnd_const_values = itertools.chain(*[_flatten_constants(k, v) for k, v in random_constants.items()])
            for key, value in chain(zip(keys, combination), rnd_const_values):
                new_config.constants.append(
                    ConstantDefinition(constant_type=ConstantType.FIXED, operand_id=key, values=[value])
                )
                new_config.model_vars[key] = new_config.constants[-1]
                name += [key, f"{value:.3f}"]
            yield "_".join(name), new_config

    @classmethod
    def from_dict(cls, dictionary: dict) -> CombinatorConfig:
        models = []
        for model in dictionary["models"]:
            models.append(ModelDefinition(**model))

        constants = []
        for constant in dictionary.get("constants", []):
            constants.append(ConstantDefinition(**constant))

        return CombinatorConfig(
            formula=dictionary["formula"],
            combination_id=dictionary.get("combination_id", ""),
            models=models,
            constants=constants,
            missing_weights_init=MissingWeightsInit(
                dictionary.get("missing_weights_init", MissingWeightsInit.NO.value)
            ),
        )
