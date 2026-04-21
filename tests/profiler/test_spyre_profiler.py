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

import json
import pytest


def test_package_importable():
    """
    Verify that the torch_spyre.profiler package can be imported
    without requiring Spyre hardware.
    """
    import torch_spyre.profiler  # noqa: F401


"TO BE WORKED ON"


def test_supported_activities():
    """
    Ensure that supported_activities() returns at least CPU activity.
    """
    from torch.profiler import ProfilerActivity
    from torch_spyre.profiler import supported_activities

    activities = supported_activities()
    assert isinstance(activities, (list, tuple, set)), (
        "supported_activities() must return a collection"
    )
    assert ProfilerActivity.CPU in activities, "ProfilerActivity.CPU must be supported"


"TO BE WORKED ON"


def test_profile_spyre_context_runs():
    """
    Verify that profile_spyre() can be entered and exited without error.
    This test should run even on machines without Spyre hardware.
    """
    import torch
    from torch_spyre.profiler import profile_spyre, supported_activities

    with profile_spyre(activities=supported_activities()) as prof:
        x = torch.randn(10, 10)
        y = torch.matmul(x, x)
        assert y is not None

    # Ensure profiler object exposes expected API
    assert hasattr(prof, "key_averages")


"TO BE WORKED ON"


def test_profile_spyre_twice():
    """
    Ensure that running profile_spyre() twice in sequence does not raise errors.
    """
    import torch
    from torch_spyre.profiler import profile_spyre, supported_activities

    for _ in range(2):
        with profile_spyre(activities=supported_activities()):
            x = torch.randn(10, 10)
            _ = torch.matmul(x, x)


"TO BE WORKED ON"


def test_chrome_trace_is_valid_json(tmp_path):
    """
    Verify that export_chrome_trace() produces valid JSON with at least one event.
    """
    import torch
    from torch_spyre.profiler import profile_spyre, supported_activities

    trace_file = tmp_path / "spyre_trace.json"

    with profile_spyre(activities=supported_activities()) as prof:
        x = torch.randn(10, 10)
        _ = torch.matmul(x, x)

    prof.export_chrome_trace(str(trace_file))

    # Ensure the file exists and contains valid JSON
    assert trace_file.exists(), "Chrome trace file was not created"

    with open(trace_file, "r") as f:
        data = json.load(f)

    # Chrome traces typically contain a "traceEvents" list
    assert isinstance(data, dict), "Trace JSON must be a dictionary"
    assert "traceEvents" in data, "Trace JSON must contain 'traceEvents'"
    assert len(data["traceEvents"]) > 0, "Trace JSON must contain at least one event"


@pytest.mark.requires_spyre_profiler
def test_synchronize_callable():
    """
    Ensure that torch.spyre.synchronize() is callable without error.
    This test requires Spyre hardware and USE_SPYRE_PROFILER=1.
    """
    import torch

    # Verify the attribute exists
    assert hasattr(torch, "spyre"), "torch.spyre namespace is missing"
    assert hasattr(torch.spyre, "synchronize"), "torch.spyre.synchronize() is missing"

    # Call the function; it should not raise an exception
    torch.spyre.synchronize()
