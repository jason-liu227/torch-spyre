# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pytest


def _is_spyre_hardware_available() -> bool:
    """
    Detect whether Spyre hardware is available.

    Returns True if the torch_spyre runtime and device can be initialized.
    This function is defensive and returns False if any step fails.
    """
    try:
        import torch
        import torch_spyre  # noqa: F401

        # Attempt to create a Spyre device. If the backend is not registered,
        # this will raise a RuntimeError.
        torch.device("spyre")
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def spyre_profiler_available() -> bool:
    """
    Fixture that returns True only when:
      1. The environment variable USE_SPYRE_PROFILER is set to "1".
      2. Spyre hardware is available.
    """
    use_profiler = os.environ.get("USE_SPYRE_PROFILER") == "1"
    hardware_available = _is_spyre_hardware_available()
    return use_profiler and hardware_available


def pytest_configure(config: pytest.Config) -> None:
    """
    Register the custom pytest marker for Spyre profiler tests.
    """
    config.addinivalue_line(
        "markers",
        "requires_spyre_profiler: mark test as requiring Spyre profiler "
        "(USE_SPYRE_PROFILER=1 and Spyre hardware present)",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """
    Automatically skip tests marked with @pytest.mark.requires_spyre_profiler
    when the Spyre profiler is not available.
    """
    if "requires_spyre_profiler" in item.keywords:
        use_profiler = os.environ.get("USE_SPYRE_PROFILER") == "1"
        hardware_available = _is_spyre_hardware_available()

        if not (use_profiler and hardware_available):
            pytest.skip(
                "Skipping test: requires Spyre profiler "
                "(set USE_SPYRE_PROFILER=1 and ensure Spyre hardware is available)"
            )
