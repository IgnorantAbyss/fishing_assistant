# ADR 0006: Dataset Lineage

Status: accepted

`datasets/fishing_v2` is frozen. Prompt Observation data is versioned separately under `datasets/prompt_observation_v1` and labelled only by `prompt_ground_truth.yaml`. Manifests retain global state only as analysis metadata and record source/frame/annotation/ROI hashes. Previously inspected v1 test sessions are development/diagnostic; v2 final test remains uncollected.
