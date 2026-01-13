# Runtime Module

## Responsibility
Implements the execution backends for shaders and AI models.

## Why it exists
Isolates hardware and library dependencies. Contains the concrete implementations (e.g., `VulkanShaderRunner`, `OnnxModelExecutor`) that fulfill the contracts defined in `core/`.
