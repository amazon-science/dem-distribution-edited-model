# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import ast
import logging
import operator
from _ast import AST
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Set, Union

import torch
from dem.combinator import CombinatorConfig, ConstantDefinition, ModelDefinition, TaskOperand
from dem.enums import MissingWeightsInit

logger = logging.getLogger(__name__)


class OperationDefinition(NamedTuple):
    operator_f: operator
    symbol: str


OPERATIONS = {
    ast.Add: OperationDefinition(operator.add, "+"),
    ast.Sub: OperationDefinition(operator.sub, "-"),
    ast.Mult: OperationDefinition(operator.mul, "*"),
    ast.Div: OperationDefinition(operator.truediv, "/"),
}


def byte_offset_to_char_offset(source: str, byte_offset: int) -> int:
    pre_source = ""
    while True:
        try:
            pre_source = source.encode()[:byte_offset].decode()
            break
        except UnicodeDecodeError:
            byte_offset -= 1
            continue
    return len(pre_source)


class FormulaError(Exception):
    pass


class FormulaSyntaxError(FormulaError):
    def __init__(self, msg: str, lineno: int, offset: int):
        self.msg = msg
        self.lineno = lineno
        self.offset = offset

    @classmethod
    def from_ast_node(cls, source: str, node: AST, msg: str) -> FormulaSyntaxError:
        lineno = node.lineno
        col_offset = node.col_offset
        offset = byte_offset_to_char_offset(source, col_offset)
        return cls(msg=msg, lineno=lineno, offset=offset + 1)

    @classmethod
    def from_syntax_error(cls, error: SyntaxError, msg: str) -> FormulaSyntaxError:
        return cls(msg=f"{msg}: {error.msg}", lineno=error.lineno, offset=error.offset)

    def __str__(self):
        return f"{self.lineno}:{self.offset}: {self.msg}"


class FormulaRuntimeError(FormulaError):
    pass


def unify_missing_weights(first: TaskOperand, second: TaskOperand, missing_weights_init: MissingWeightsInit) -> None:
    for name, weight in first.state_dict.items():
        if missing_weights_init == MissingWeightsInit.COPY:
            second.state_dict[name] = first.state_dict[name]
        elif missing_weights_init == MissingWeightsInit.ZEROS:
            second.state_dict[name] = torch.zeros_like(weight)
        elif missing_weights_init == MissingWeightsInit.ONES:
            second.state_dict[name] = torch.ones_like(weight)
        else:
            raise NotImplementedError(f"Option {missing_weights_init} not supported.")
        logger.info("%s is missing from the model weights.", name)


def get_shared_weight(state_dict: Dict[str, torch.Tensor]) -> Set[str]:
    # We want to resolve the list of parameter that are tied in order to avoid applying
    # the operation multiple times on the shared parameters.
    keys = list(state_dict.keys())
    shared_weights_list = {
        k
        for i, k in enumerate(keys)
        if any(state_dict[k].data_ptr() == state_dict[k2].data_ptr() for k2 in keys[i + 1 :])
    }
    logger.info("Shared keys found: %s", shared_weights_list)
    return shared_weights_list


def combine_weights(
    first: Union[float, TaskOperand],
    second: [float, TaskOperand],
    op: operator,
    missing_weights_init: MissingWeightsInit = MissingWeightsInit.NO,
) -> Union[TaskOperand, float]:
    operation = OPERATIONS[type(op)]
    is_second_scalar = isinstance(second, float)
    # We always put the model first, and the constant second, so we can iterate later on the model_State and update it.
    if isinstance(first, float):
        if is_second_scalar:
            # Apply the operation directly if both are floats
            return operation.operator_f(first, second)

        # Swap the two to make sure the first is an TaskOperand, and set the flag to true
        first, second = second, first
        is_second_scalar = True
        logger.info("Rescaling the model's weights using a constant (%.3f).", second)

    second_op_id = f"{second:.3f}" if is_second_scalar else second.config["operand_id"]
    first.config["operand_id"] = f"({first.config['operand_id']} {operation.symbol} {second_op_id})"
    logger.info("Executing: `%s`", first.config["operand_id"])

    if not is_second_scalar and missing_weights_init != MissingWeightsInit.NO:
        unify_missing_weights(first, second, missing_weights_init)
        unify_missing_weights(second, first, missing_weights_init)

    shared_weights_list = get_shared_weight(first.state_dict)
    for name, weight in first.state_dict.items():
        if (
            weight.dtype in [torch.int64, torch.uint8, torch.bool]
            or name in shared_weights_list
            # Attn.mask_bias is set to a constant value in the model, we don't need to interpolate it.
            # For some model checkpoints it can be missing in the weights file.
            or name.endswith("attn.masked_bias")
        ):
            logger.info("skipping weight: %s", name)
            continue

        if is_second_scalar:
            scaler_value = second
        else:
            resolved_name = name
            if resolved_name not in second.state_dict:
                if name.startswith("lm_head"):
                    resolved_name = name.replace("lm_head", "wte")
                resolved_names = [
                    x for x in second.state_dict.keys() if x.endswith(resolved_name.replace("transformer.", ""))
                ]
                assert len(resolved_names) == 1, (
                    f"Cannot resolve the key for `{resolved_name}` (`{name}`) in found keys: {resolved_names}."
                    f"\n {' '.join(second.state_dict.keys())}"
                )
                resolved_name = resolved_names[0]
                logger.debug("Renamed weight `%s` to`%s`", name, resolved_name)
            scaler_value = second.state_dict[resolved_name]
            assert (
                scaler_value.size() == weight.size()
            ), f"Incompatible sizes of the `{name}` layer: {list(scaler_value.shape)} vs {list(weight.shape)}"

        if isinstance(op, ast.Add):
            weight += scaler_value
        elif isinstance(op, ast.Sub):
            weight -= scaler_value
        # These operations are only supported for scalars
        elif isinstance(op, ast.Mult) and is_second_scalar:
            weight *= scaler_value
        elif isinstance(op, ast.Div) and is_second_scalar:
            weight /= scaler_value
        else:
            raise NotImplementedError(
                f"Unsupported operation: `{operation.symbol}` between `{type(first)}` and `{type(second)}`"
            )

    return first


class TaskVectorArithmetic:
    def __init__(
        self,
        config: CombinatorConfig,
        missing_weights_init: MissingWeightsInit = MissingWeightsInit.NO,
        cache_dir: Optional[Path] = None,
        s3_profile: Optional[str] = None,
    ):
        self.config = config
        self.model_vars = self.config.model_vars
        self.missing_weights_init = missing_weights_init
        self.cache_dir = cache_dir
        self.s3_profile = s3_profile

    def _eval_name(self, node: ast.Name) -> [float, TaskOperand]:
        try:
            var_definition = self.model_vars[node.id]
        except KeyError:
            raise FormulaSyntaxError.from_ast_node(
                self.config.formula, node, f"Undefined operand id: {node.id} in [{', '.join(self.model_vars.keys())}]"
            )

        if isinstance(var_definition, ModelDefinition):
            model = var_definition.load_model(self.cache_dir)
            logger.info("Loaded model weights from `%s`", var_definition.path)

            # Scale the model's parameters with the weight
            if var_definition.weight != 1.0:
                model = combine_weights(
                    first=model,
                    second=var_definition.weight,
                    op=ast.Mult(),
                    missing_weights_init=self.missing_weights_init,
                )
            return model
        elif isinstance(var_definition, ConstantDefinition):
            return var_definition.next_value()
        elif isinstance(var_definition, float):
            return var_definition
        else:
            raise NotImplementedError(f"Unsupported model variable {type(var_definition)}")

    def _eval_binop(self, node: ast.BinOp) -> TaskOperand:
        left_value = self._eval_node(node.left)
        right_value = self._eval_node(node.right)

        try:
            return combine_weights(
                first=left_value, second=right_value, op=node.op, missing_weights_init=self.missing_weights_init
            )
        except KeyError:
            raise FormulaSyntaxError.from_ast_node(
                self.config.formula, node, "Operations of this type are not supported"
            )

    def _eval_unaryop(self, node: ast.UnaryOp) -> Any:
        if isinstance(node.op, ast.USub):
            operand_value = self._eval_node(node.operand)
            return combine_weights(
                first=operand_value, second=-1.0, op=ast.Mult(), missing_weights_init=self.missing_weights_init
            )

        raise FormulaSyntaxError.from_ast_node(self.config.formula, node, "Operations of this type are not supported")

    def _eval_node(self, node: AST) -> [float, TaskOperand]:
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)

            raise FormulaSyntaxError.from_ast_node(self.config.formula, node, "Literals of this type are not supported")
        elif isinstance(node, ast.Name):
            return self._eval_name(node)
        elif isinstance(node, ast.BinOp):
            return self._eval_binop(node)
        elif isinstance(node, ast.UnaryOp):
            return self._eval_unaryop(node)
        else:
            raise FormulaSyntaxError.from_ast_node(self.config.formula, node, "This syntax is not supported")

    @torch.no_grad()
    def evaluate_formula(self: TaskVectorArithmetic) -> TaskOperand:
        logger.info("Formula for the model arithmetic: `%s`", self.config.formula)
        try:
            node = ast.parse(self.config.formula, "<string>", mode="eval")
        except SyntaxError as e:
            raise FormulaSyntaxError.from_syntax_error(e, "Could not parse")

        result = self._eval_node(node)
        if isinstance(result, float):
            raise ValueError(f"There are no models to combine in the formula: {self.config.formula}")

        return result
