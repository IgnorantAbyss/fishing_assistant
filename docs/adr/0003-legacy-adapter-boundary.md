# ADR 0003: Legacy Adapter Boundary

Status: accepted

Only `src/fishing_v2/legacy_adapters` may import legacy specialized detectors and replay storage. Adapters translate dictionaries/exceptions into typed v2 observations. Core domain/fusion/runtime packages cannot import legacy state detection, smoothing, prompt templates, specialized implementation modules, or `src.ml`.
