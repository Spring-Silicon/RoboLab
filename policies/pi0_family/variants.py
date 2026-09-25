# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

PI0_VARIANTS = (
    "pi0",
    "pi0_fast",
    "pi05",
    "pi05_compiled_regular",
    "pi05_compiled_optimized",
    "paligemma",
    "paligemma_fast",
)

DEFAULT_HORIZONS: dict[str, int] = {
    "pi0": 10,
    "pi0_fast": 10,
    "pi05": 15,
    "pi05_compiled_regular": 15,
    "pi05_compiled_optimized": 15,
    "paligemma": 15,
    "paligemma_fast": 15,
}

COMPILED_VARIANTS = frozenset({"pi05_compiled_regular", "pi05_compiled_optimized"})
VELOCITY_VARIANTS = frozenset({"paligemma_fast"})
