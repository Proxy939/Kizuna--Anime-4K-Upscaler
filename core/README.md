# Core Module

## Responsibility
Contains the abstract interfaces, data contracts, and pipeline definitions.

## Why it exists
To decouple the system logic from specific implementations. Code here defines *what* the system does (e.g., `IUpscaler`, `FrameDescriptor`), not *how* it does it.
