# Prompt Inventory Manual Review Template

This inventory is intentionally unlabelled. Global state only selects review strata; it must not be copied into Prompt observation ground truth.
Allowed suggestions after visual review: `IDLE_PROMPT`, `WAITING_PROMPT`, `READY_PROMPT`, `OTHER_PROMPT`, `NO_PROMPT`, `IGNORE`.
Optional prompt-id examples: `IDLE_CAST`, `FISHING_IN_PROGRESS`, `FISH_BITE_SPACE`, `PRESS_SEQUENCE_INSTRUCTION`.

Distinct Prompt appearance count: **not yet manually confirmed**.

| session_id | frame_index_or_range | global_state | visible_prompt_description | suggested_observation_kind | suggested_prompt_id | candidate_roi_complete | transition_or_stable | manual_review_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| session_20260709_192315 | 9, 12, 14, 17, 510, 513, 515, 518 | IDLE |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260709_192315 | 36, 100, 165, 229, 293, 357, 422, 590 | WAITING |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260709_192315 | 446, 447, 448, 449, 450, 451 | READY |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260709_192315 | 460, 461, 462, 463, 464, 465, 466 | HOOK |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260709_192315 | 475, 476, 477, 478, 479, 480 | PRESS |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260709_192315 | 489, 490, 491, 492, 493, 494 | GET |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260709_192315 | 16-35 | IDLE→WAITING |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 26; review each side. |
| session_20260709_192315 | 433-452 | WAITING→READY |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 443; review each side. |
| session_20260709_192315 | 445-464 | READY→HOOK |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 455; review each side. |
| session_20260709_192315 | 462-481 | HOOK→PRESS |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 472; review each side. |
| session_20260709_192315 | 474-493 | PRESS→GET |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 484; review each side. |
| session_20260709_192315 | 490-509 | GET→IDLE |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 500; review each side. |
| session_20260709_192315 | 517-536 | IDLE→WAITING |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 527; review each side. |
| session_20260710_061220 | 539, 540, 542, 543, 545, 546, 548, 549 | IDLE |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_061220 | 11, 76, 140, 205, 269, 334, 398, 590 | WAITING |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_061220 | 461, 462, 464, 465, 466, 467, 469, 470 | READY |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_061220 | 488, 489, 491, 492, 494, 495, 497, 498 | HOOK |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_061220 | 510, 511, 512, 513, 514 | PRESS |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_061220 | 442-461 | WAITING→READY |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 452; review each side. |
| session_20260710_061220 | 470-489 | READY→HOOK |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 480; review each side. |
| session_20260710_061220 | 497-516 | HOOK→PRESS |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 507; review each side. |
| session_20260710_061220 | 508-527 | PRESS→IGNORE |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 518; review each side. |
| session_20260710_061220 | 549-568 | IDLE→WAITING |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 559; review each side. |
| session_20260710_123210 | 11, 15, 19, 23, 28, 32, 36, 600 | IDLE |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_123210 | 59, 121, 182, 244, 306, 368, 429, 491 | WAITING |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_123210 | 512, 517, 522, 527, 531, 536, 541, 546 | READY |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_123210 | 562, 563, 564, 565, 566, 567, 568 | HOOK |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_123210 | 577, 578, 579, 580 | PRESS |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_123210 | 39-58 | IDLE→WAITING |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 49; review each side. |
| session_20260710_123210 | 492-511 | WAITING→READY |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 502; review each side. |
| session_20260710_123210 | 547-566 | READY→HOOK |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 557; review each side. |
| session_20260710_123210 | 564-583 | HOOK→PRESS |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 574; review each side. |
| session_20260710_123210 | 574-593 | PRESS→IGNORE |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 584; review each side. |
| session_20260710_124419 | 11, 15, 20, 24, 29, 33, 388, 392 | IDLE |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_124419 | 55, 108, 162, 215, 430, 483, 537, 590 | WAITING |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_124419 | 267, 271, 276, 280, 285, 289, 294, 298 | READY |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_124419 | 317, 318, 319, 320, 322, 323, 324, 325 | HOOK |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_124419 | 341, 342, 343, 344, 345, 346, 347, 348 | PRESS |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_124419 | 363, 364, 365, 366, 368, 369, 370, 371 | GET |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_124419 | 35-54 | IDLE→WAITING |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 45; review each side. |
| session_20260710_124419 | 247-266 | WAITING→READY |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 257; review each side. |
| session_20260710_124419 | 299-318 | READY→HOOK |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 309; review each side. |
| session_20260710_124419 | 324-343 | HOOK→PRESS |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 334; review each side. |
| session_20260710_124419 | 346-365 | PRESS→GET |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 356; review each side. |
| session_20260710_124419 | 369-388 | GET→IDLE |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 379; review each side. |
| session_20260710_124419 | 389-408 | IDLE→WAITING |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 399; review each side. |
| session_20260710_125441 | 11, 17, 24, 30, 37, 43, 446, 452 | IDLE |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_125441 | 69, 114, 159, 204, 250, 500, 545, 590 | WAITING |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_125441 | 284, 293, 303, 312, 322, 331, 341, 350 | READY |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_125441 | 371, 373, 375, 377, 379, 381, 383, 385 | HOOK |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_125441 | 403, 404, 406, 407, 408, 409, 411, 412 | PRESS |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_125441 | 426, 427, 428, 429, 430, 431, 432, 433 | GET |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_125441 | 49-68 | IDLE→WAITING |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 59; review each side. |
| session_20260710_125441 | 264-283 | WAITING→READY |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 274; review each side. |
| session_20260710_125441 | 351-370 | READY→HOOK |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 361; review each side. |
| session_20260710_125441 | 386-405 | HOOK→PRESS |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 396; review each side. |
| session_20260710_125441 | 410-429 | PRESS→IGNORE |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 420; review each side. |
| session_20260710_125441 | 429-448 | GET→IDLE |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 439; review each side. |
| session_20260710_125441 | 449-468 | IDLE→WAITING |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 459; review each side. |
| session_20260710_130308 | 72, 79, 87, 473, 480, 487, 495, 502 | IDLE |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_130308 | 110, 148, 187, 225, 263, 301, 552, 590 | WAITING |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_130308 | 7, 14, 336, 343, 349, 356, 362, 369 | READY |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_130308 | 31, 36, 41, 46, 51, 394, 399, 404 | HOOK |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_130308 | 422, 423, 425, 426, 427, 428, 430, 431 | PRESS |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_130308 | 446, 447, 448, 449, 450, 451, 452 | GET |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_130308 | 11-30 | READY→HOOK |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 21; review each side. |
| session_20260710_130308 | 52-71 | HOOK→IDLE |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 62; review each side. |
| session_20260710_130308 | 90-109 | IDLE→WAITING |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 100; review each side. |
| session_20260710_130308 | 311-330 | WAITING→READY |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 321; review each side. |
| session_20260710_130308 | 370-389 | READY→HOOK |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 380; review each side. |
| session_20260710_130308 | 405-424 | HOOK→PRESS |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 415; review each side. |
| session_20260710_130308 | 429-448 | PRESS→IGNORE |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 439; review each side. |
| session_20260710_130308 | 449-468 | GET→IDLE |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 459; review each side. |
| session_20260710_130308 | 503-522 | IDLE→WAITING |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 513; review each side. |
| session_20260710_131254 | 284, 290, 295, 301, 306, 312, 317, 598 | IDLE |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_131254 | 11, 46, 82, 346, 382, 417, 453, 488 | WAITING |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_131254 | 131, 139, 148, 156, 164, 172, 509, 517 | READY |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_131254 | 200, 204, 207, 211, 214, 538, 541, 545 | HOOK |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_131254 | 233, 235, 238, 240, 562, 564, 567, 569 | PRESS |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_131254 | 258, 259, 260, 261, 263, 264, 265, 266 | GET |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |
| session_20260710_131254 | 111-130 | WAITING→READY |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 121; review each side. |
| session_20260710_131254 | 180-199 | READY→HOOK |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 190; review each side. |
| session_20260710_131254 | 215-234 | HOOK→PRESS |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 225; review each side. |
| session_20260710_131254 | 240-259 | PRESS→IGNORE |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 250; review each side. |
| session_20260710_131254 | 264-283 | GET→IDLE |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 274; review each side. |
| session_20260710_131254 | 320-339 | IDLE→WAITING |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 330; review each side. |
| session_20260710_131254 | 489-508 | WAITING→READY |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 499; review each side. |
| session_20260710_131254 | 517-536 | READY→HOOK |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 527; review each side. |
| session_20260710_131254 | 544-563 | HOOK→PRESS |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 554; review each side. |
| session_20260710_131254 | 567-586 | PRESS→IGNORE |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame 577; review each side. |
