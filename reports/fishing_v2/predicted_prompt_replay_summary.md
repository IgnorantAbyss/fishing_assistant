# Predicted Prompt Replay Summary

| session | scripted final | predicted final | READY | HOOK hint | PRESS hint | false intents | sync | result |
|---|---|---|---:|---:|---:|---:|---:|---|
| session_20260709_192315 | WAITING | WAITING | 1/1 | 1/1 | 1/1 | 0 | 0 | PASS |
| session_20260710_061220 | WAITING | WAITING | 1/1 | 1/1 | 1/1 | 0 | 0 | PASS |
| session_20260710_123210 | IDLE | IDLE | 1/1 | 1/1 | 1/1 | 0 | 0 | PASS |
| session_20260710_124419 | WAITING | WAITING | 1/1 | 1/1 | 1/1 | 0 | 0 | PASS |
| session_20260710_125441 | WAITING | WAITING | 1/1 | 1/1 | 1/1 | 0 | 0 | PASS |
| session_20260710_130308 | WAITING | WAITING | 2/2 | 2/2 | 1/1 | 0 | 0 | PASS |
| session_20260710_131254 | IDLE | IDLE | 2/2 | 2/2 | 2/2 | 0 | 0 | PASS |

- Complete replays: **7/7**
- FAIL frames: **0**
- Actions applied: **0**
- False / missed intents: **0 / 0**
- SYNC_REQUIRED frames: **0**
- Mean Runtime state agreement: **0.9960**
- Mean transition agreement: **0.8485**
- Gate `prompt_observer_ready_for_live_detect_only`: **true**

## First divergences

- session_20260709_192315: `{'frame': 502, 'scripted_state': 'IDLE', 'predicted_state': 'COLLECT_PENDING', 'scripted_reason': 'stable_highest_supported_candidate', 'predicted_reason': 'candidate_not_stable', 'observer': {'predicted_label': 'IDLE_CAST', 'raw_predicted_label': 'IDLE_CAST', 'similarity': 0.8970315456390381, 'second_label': 'PRESS_INSTRUCTION', 'second_similarity': 0.5876941680908203, 'ambiguity_margin': 0.3093373775482178, 'rejection_reason': None, 'prototype_id': 'IDLE_CAST:session_20260710_061220:000551', 'observer_version': 'prototype_v1', 'approved_roi': [940, 36, 1620, 100], 'idle_streak': 4, 'idle_stability_frames': 4}}`
- session_20260710_061220: `{'frame': 560, 'scripted_state': 'WAITING', 'predicted_state': 'IDLE', 'scripted_reason': 'stable_highest_supported_candidate', 'predicted_reason': 'idle_waiting_for_prompt_or_get_guard', 'observer': {'predicted_label': 'UNKNOWN', 'raw_predicted_label': 'UNKNOWN', 'similarity': 0.8967551589012146, 'second_label': 'IDLE_CAST', 'second_similarity': 0.6164915561676025, 'ambiguity_margin': 0.28026360273361206, 'rejection_reason': 'ambiguous_top_two', 'prototype_id': 'WAITING_IN_PROGRESS:session_20260710_124419:000573', 'observer_version': 'prototype_v1', 'approved_roi': [940, 36, 1620, 100], 'idle_streak': 0, 'idle_stability_frames': 4}}`
- session_20260710_123210: `None`
- session_20260710_124419: `{'frame': 46, 'scripted_state': 'WAITING', 'predicted_state': 'IDLE', 'scripted_reason': 'stable_highest_supported_candidate', 'predicted_reason': 'idle_waiting_for_prompt_or_get_guard', 'observer': {'predicted_label': 'UNKNOWN', 'raw_predicted_label': 'UNKNOWN', 'similarity': 0.9072011113166809, 'second_label': 'IDLE_CAST', 'second_similarity': 0.6242858171463013, 'ambiguity_margin': 0.28291529417037964, 'rejection_reason': 'ambiguous_top_two', 'prototype_id': 'WAITING_IN_PROGRESS:session_20260710_130308:000297', 'observer_version': 'prototype_v1', 'approved_roi': [940, 36, 1620, 100], 'idle_streak': 0, 'idle_stability_frames': 4}}`
- session_20260710_125441: `None`
- session_20260710_130308: `{'frame': 63, 'scripted_state': 'RESULT_PENDING', 'predicted_state': 'HOOK', 'scripted_reason': 'recorded_hook_result_prompt_acknowledgement', 'predicted_reason': 'hook_episode_not_active', 'observer': {'predicted_label': 'UNKNOWN', 'raw_predicted_label': 'IDLE_CAST', 'similarity': 0.9319776296615601, 'second_label': 'PRESS_INSTRUCTION', 'second_similarity': 0.6156266331672668, 'ambiguity_margin': 0.3163509964942932, 'rejection_reason': 'idle_temporal_guard', 'prototype_id': 'IDLE_CAST:session_20260710_124419:000015', 'observer_version': 'prototype_v1', 'approved_roi': [940, 36, 1620, 100], 'idle_streak': 2, 'idle_stability_frames': 4}}`
- session_20260710_131254: `{'frame': 331, 'scripted_state': 'WAITING', 'predicted_state': 'IDLE', 'scripted_reason': 'stable_highest_supported_candidate', 'predicted_reason': 'idle_waiting_for_prompt_or_get_guard', 'observer': {'predicted_label': 'UNKNOWN', 'raw_predicted_label': 'UNKNOWN', 'similarity': 0.90080726146698, 'second_label': 'IDLE_CAST', 'second_similarity': 0.6234594583511353, 'ambiguity_margin': 0.2773478031158447, 'rejection_reason': 'ambiguous_top_two', 'prototype_id': 'WAITING_IN_PROGRESS:session_20260710_124419:000573', 'observer_version': 'prototype_v1', 'approved_roi': [940, 36, 1620, 100], 'idle_streak': 0, 'idle_stability_frames': 4}}`
