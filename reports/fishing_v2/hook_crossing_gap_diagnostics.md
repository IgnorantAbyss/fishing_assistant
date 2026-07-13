# Hook Crossing Gap Diagnostics: session_20260710_125441

- Reviewed frames: 360-396
- Root cause: frame 369 had fresh safe divider/fill geometry, but the old policy rejected it solely because qualified_active was false.
- Crossing edge was not lost or cached; the action policy is now level-triggered one-shot.
- First usable post-cross evidence: **369**
- First HOOK_ACTION intent: **369**

| frame | visible | state | raw | qualified | fusion | divider | endpoint | margin | latch | ready | intent | rejection |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| 360 | False | `READY` | False | False | False | None | None | False | `inactive` | False | False | `runtime_not_in_hook` |
| 361 | False | `READY` | False | False | False | None | None | False | `inactive` | False | False | `runtime_not_in_hook` |
| 362 | False | `HOOK_PENDING` | False | False | False | None | None | False | `inactive` | False | False | `runtime_not_in_hook` |
| 363 | False | `HOOK_PENDING` | False | False | False | None | None | False | `inactive` | False | False | `runtime_not_in_hook` |
| 364 | False | `HOOK_PENDING` | False | False | False | None | None | False | `inactive` | False | False | `runtime_not_in_hook` |
| 365 | False | `HOOK_PENDING` | False | False | False | None | None | False | `inactive` | False | False | `runtime_not_in_hook` |
| 366 | True | `HOOK_PENDING` | True | True | True | 1330.0 | None | False | `inactive` | False | False | `runtime_not_in_hook` |
| 367 | True | `HOOK_PENDING` | True | True | True | 1330.0 | None | False | `activated_after_frame` | False | False | `runtime_not_in_hook` |
| 368 | True | `HOOK` | True | True | True | 1330.0 | 1335.0 | False | `active` | False | False | `fill_crossed_divider_but_margin_pending` |
| 369 | True | `HOOK` | False | False | False | 1330.0 | 1373.0 | True | `active` | True | True | `None` |
| 370 | True | `HOOK` | True | True | True | 1330.0 | None | False | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 371 | True | `HOOK` | True | True | True | 1330.0 | None | False | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 372 | True | `HOOK` | True | True | True | 1330.0 | None | False | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 373 | True | `HOOK` | True | True | True | 1330.0 | None | False | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 374 | True | `HOOK` | True | True | True | 1330.0 | None | False | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 375 | True | `HOOK` | True | True | True | 1330.0 | None | False | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 376 | True | `HOOK` | True | True | True | 1330.0 | None | False | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 377 | True | `HOOK` | False | False | False | 1330.0 | 1476.0 | True | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 378 | True | `HOOK` | False | False | False | 1330.0 | 1355.0 | True | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 379 | True | `HOOK` | True | True | True | 1330.0 | None | False | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 380 | True | `HOOK` | True | True | True | 1330.0 | None | False | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 381 | True | `HOOK` | True | True | True | 1330.0 | None | False | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 382 | True | `HOOK` | True | True | True | 1330.0 | None | False | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 383 | True | `HOOK` | True | True | True | 1330.0 | None | False | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 384 | True | `HOOK` | True | True | True | 1330.0 | None | False | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 385 | True | `HOOK` | True | True | True | 1330.0 | None | False | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 386 | True | `HOOK` | False | False | False | 1330.0 | 1470.0 | True | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 387 | True | `HOOK` | False | False | False | 1330.0 | 1470.0 | True | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 388 | True | `HOOK` | False | False | False | 1330.0 | 1470.0 | True | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 389 | True | `HOOK` | False | False | False | 1330.0 | 1470.0 | True | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 390 | True | `HOOK` | False | False | False | 1330.0 | 1470.0 | True | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 391 | True | `HOOK` | False | False | False | 1330.0 | 1470.0 | True | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 392 | True | `HOOK` | False | False | False | 1330.0 | 1470.0 | True | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 393 | True | `HOOK` | False | False | False | 1330.0 | 1470.0 | True | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 394 | True | `HOOK` | False | False | False | 1330.0 | 1470.0 | True | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 395 | True | `HOOK` | False | False | False | 1330.0 | 1470.0 | True | `action_already_proposed` | False | False | `hook_episode_not_active` |
| 396 | False | `HOOK` | False | False | False | None | None | False | `action_already_proposed` | False | False | `hook_episode_not_active` |
