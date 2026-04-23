# Architecture Overview

```
+-----------------------------------------------------------------------+
|                         Visualization Layer                           |
|-----------------------------------------------------------------------|
|          Chrome Trace Viewer / TensorBoard Profiler Plugin            |
+----------------------------------▲------------------------------------+
                                   |
                                   | Unified Traces (JSON)
                                   |
+-----------------------------------------------------------------------+
|                          PyTorch Process                              |
|-----------------------------------------------------------------------|
|  +---------------------------------------------------------------+    |
|  |                    torch.profiler (User API)                  |    |
|  +-------------------------------▲-------------------------------+    |
|                                  |                                    |
|                                  | Python Events & Traces             |
|  +-------------------------------+-------------------------------+    |
|  |                      torch-spyre Wiring                       |    |
|  |        (Integration with PyTorch ATen & Kineto backend)       |    |
|  +-------------------------------▲-------------------------------+    |
|                                  |                                    |
|                                  |                                    |
|  +-------------------------------+-------------------------------+    |
|  |                  torch_spyre.profiler (Python)                |    |
|  |---------------------------------------------------------------|    |
|  |  __init__.py              → profile_spyre(),                  |    |
|  |  _spyre_activity.py       → Activity registration (future)    |    |
|  |                                                       |    |
|  +-------------------------------▲-------------------------------+    |
|                                  | Python ↔ C++ bridge                |
|  +-------------------------------+-------------------------------+    |
|  |               torch_spyre/csrc/profiler (C++)                 |    |
|  |---------------------------------------------------------------|    |
|  |                          Populate Later                       |    |
|  +-------------------------------▲-------------------------------+    |
|                                  |                                    |
|  +-------------------------------+-------------------------------+    |
|  |                            libkineto                          |    |
|  |          (Activity collection and trace aggregation)          |    |
|  +-------------------------------▲-------------------------------+    |
|                                  | AIU Activities                     |
|  +-------------------------------+-------------------------------+    |
|  |               AiuptiActivityApi (kineto-spyre)                |    |
|  |         (CUPTI-like interface for Kineto integration)         |    |
|  +-------------------------------▲-------------------------------+    |
|                                  | Hardware Counters & Events         |
|  +-------------------------------+-------------------------------+    |
|  |                           libAIUPTI                           |    |
|  |         (Low-level AIU profiling and instrumentation)         |    |
|  +-------------------------------▲-------------------------------+    |
+----------------------------------|------------------------------------+
                                   |
                                   | Device Operations & Metrics
                                   ▼
+-----------------------------------------------------------------------+
|                              OS Layer                                 |
|-----------------------------------------------------------------------|
|  +---------------------------------------------------------------+    |
|  |                       AIU Runtime (Flex)                      |    |
|  |             (Queues and manages AIU operation)                |    |
|  +-------------------------------▲-------------------------------+    |
|                                  |                                    |
|  +-------------------------------+-------------------------------+    |
|  |                      AIU Driver (VF/PF)                       |    |
|  |               (Kernel-level device management)                |    |
|  +-------------------------------▲-------------------------------+    |
+----------------------------------|------------------------------------+
                                   |
                                   | Hardware Execution
                                   ▼
+-----------------------------------------------------------------------+
|                             AIU Hardware                              |
|-----------------------------------------------------------------------|
|                 +--------+   +--------+         +--------+            |
|                 | AIU 1  |   | AIU 2  |   ...   | AIU n  |            |
|                 +--------+   +--------+         +--------+            |
+-----------------------------------------------------------------------+
```

## Installation

### 1. Install the kineto-spyre Wheel

Profiling support requires the kineto-spyre wheel (version matching the PyTorch install):

```bash
pip install torch-2.9.1+aiu.kineto.1.1-*.whl
```

### 2. Enable Profiling During Build

Set the USE_SPYRE_PROFILER flag when building the project:

```bash
export USE_SPYRE_PROFILER=1
pip install -e .
```

Alternatively, for a CMake-based build:

```bash
USE_SPYRE_PROFILER=1 python setup.py develop
```

## Quick Start

## API Reference

## Kineto Dependency

Spyre profiling depends on a specialized Kineto build. Two integration paths are supported, a Short-Term Path and a Long-Term Path:

### Short-Term Path (Prebuilt Wheel)

Install the prebuilt kineto-spyre wheel:

```bash
pip install torch-2.9.1+aiu.kineto.1.1-*.whl
```

This is the recommended approach for most users and CI environments.

### Long-Term Path (Upstream Integration)

To be added.
Please see [Pytorch #166205](https://github.com/torch-spyre/torch-spyre/issues/926?issue=pytorch%7Cpytorch%7C166205) for known limitations.

## Known Limitations

Please see [Pytorch #166205](https://github.com/torch-spyre/torch-spyre/issues/926?issue=pytorch%7Cpytorch%7C166205) for known limitations.
Currently being worked on.

## Advanced Usage

Please see [IBM Kineto-Spyre Benchmark](https://github.com/IBM/kineto-spyre/tree/main/benchmarks) for Perfetto trace examples.
