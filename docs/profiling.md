#Architecture Overview

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

##Installation

### 1. Install the kineto-spyre Wheel

Profiling support requires the kineto-spyre wheel (version matching the PyTorch install):

```bash
pip install torch-(PyTorch Version)+aiu.kineto.1.1-*.whl
```

##Quick Start

##API Reference

##Kineto Dependency

##Known Limitations

Please see [Pytorch #166205](https://github.com/torch-spyre/torch-spyre/issues/926?issue=pytorch%7Cpytorch%7C166205) for known limitations.

##Advanced Usage

Please see [IBM Kineto-Spyre Benchmark](https://github.com/IBM/kineto-spyre/tree/main/benchmarks) for Perfetto trace examples.
