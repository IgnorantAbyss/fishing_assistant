# Hook Bar visual range candidates

這份資料是直接檢視七個正式 replay session 原始 frame 後建立的自動候選，尚未經人工確認，不是 human ground truth。

- 找到 9 個 Hook Bar episode。
- 9 個候選皆為 high confidence；沒有 uncertain frame。
- 每段都檢查 global HOOK 前後至少 5 frames。
- global ground truth 僅用來限定搜尋範圍。
- 未使用 Hook detector 的 detected、confidence 或 fill ratio。
- 最小人工複查項目為每段的 `visible_start - 1`、`visible_start`、`visible_end`、`visible_end + 1`。

候選範圍：

| Session | Episode | Visible range |
|---|---:|---:|
| session_20260709_192315 | 1 | 459-471 |
| session_20260710_061220 | 1 | 486-506 |
| session_20260710_123210 | 1 | 562-573 |
| session_20260710_124419 | 1 | 314-333 |
| session_20260710_125441 | 1 | 366-395 |
| session_20260710_130308 | 1 | 26-61 |
| session_20260710_130308 | 2 | 385-414 |
| session_20260710_131254 | 1 | 195-224 |
| session_20260710_131254 | 2 | 532-553 |
