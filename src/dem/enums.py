# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from enum import Enum


class MissingWeightsInit(str, Enum):
    NO = "no"
    COPY = "copy"
    ZEROS = "zeros"
    ONES = "ones"
