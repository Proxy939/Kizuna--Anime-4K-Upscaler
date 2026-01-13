# Design Principles

## Purpose
Guiding philosophy for code contribution and architectural decisions in KizunaSR.

## Key Principles (Draft)
1.  **Latency First**: The pipeline must never block the render thread longer than 16ms in "Realtime" profile.
2.  **Backend Agnostic**: The Core logic should never import `vulkan` or `torch` directly.
3.  **Config-Driven**: Behavior should be defined in `profiles/`, not hardcoded in C++/Python.
