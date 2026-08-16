# PROJECT_RULES.md

# Fishing Assistant — Project Rules

本文件是本 repository 所有 Codex / AI 修改工作的永久規則。

任何任務開始前，都必須先完整閱讀並遵守本文件。

除非使用者當次 Prompt 明確要求修改 `PROJECT_RULES.md`，否則不得自行修改本文件。

如果當次 Task Prompt 與本文件發生衝突：

1. 不得自行猜測使用者意圖。
2. 明確指出衝突內容。
3. 停止有風險的修改。
4. 等待使用者決定。

Task Prompt 可以增加更嚴格的限制，但不得默默削弱本文件的 safety / correctness 規則。

---

# 1. 核心優先順序

所有修改的優先順序：

1. Correctness
2. Safety
3. Liveness
4. Observability
5. Performance
6. Cleanup / elegance

不得為了：

- 效能
- CPU 使用率
- 程式碼簡潔
- 減少行數
- abstraction
- cleanup
- refactor
- architecture 美化

破壞 correctness、safety、liveness 或已經通過 Live 驗證的 Production behavior。

修 bug 時優先：

```text
取得 evidence
→ 找到 precise root cause
→ 證明實際 code path
→ 最小範圍修正
→ deterministic regression
→ full regression
→ Live canary
```

不得只根據症狀猜測後直接 patch。

---

# 2. 語言與事實標準

與使用者的 Final 回報使用繁體中文。

程式碼、class、function、event、field、CLI、commit message 保留 repository 原本英文命名。

分析時必須區分：

- confirmed
- high-confidence inference
- hypothesis
- unknown

不得把推測寫成已證實事實。

如果 Production log / evidence 不足：

明確說：

```text
現有證據不足以確認。
```

不得自行補完故事。

---

# 3. 任務開始前 Git 檢查

所有可能修改 repository 的任務開始前，至少執行：

```powershell
git status
git branch --show-current
git log -5 --oneline --decorate
git remote -v
```

不得自行假設：

- branch
- HEAD
- remote HEAD
- working tree 狀態

以實際 Git 輸出為準。

如果 working tree 不乾淨：

- 不得 `git reset`
- 不得 `git reset --hard`
- 不得 `git clean`
- 不得刪除未知檔案
- 不得覆蓋使用者尚未提交的修改
- 不得自行 stash 使用者工作
- 停止並回報

---

# 4. Git 永久禁止事項

除非使用者在當次任務中極明確要求，而且沒有更安全方案，否則永遠禁止：

```text
git add .
git reset --hard
git clean -fd
git clean -fdx
git push --force
git push --force-with-lease
```

只 add 本次任務必要的 source / tests / docs。

不得提交：

- `reports/`
- Production session
- runtime logs
- anomaly PNG
- screenshots
- videos
- MP4
- cache
- `__pycache__`
- `.pytest_cache`
- generated agent reports
- benchmark temporary output
- temporary analysis artifacts
- 使用者未要求加入版本控制的 evidence

不得 push `main`，除非使用者在當次任務中明確要求。

一般修改只 push Task Prompt 指定的 branch。

不得 force push。

如果 Codex / 安全系統要求新的 explicit push authorization：

1. 不得繞過。
2. 保留已完成且 clean 的 commit。
3. 回報 commit hash。
4. 等待使用者明確授權。
5. 未授權前不得自行 push。

---

# 5. 修改範圍

只修改當次 Task Prompt 明確指定的 subsystem。

除非任務本身要求，禁止順手：

- generic cleanup
- dead-code cleanup
- naming cleanup
- architecture rewrite
- large refactor
- performance tuning
- threshold tuning
- timing tuning
- FPS tuning
- unrelated telemetry changes
- unrelated test rewrites
- unrelated compatibility cleanup

如果修改途中發現另一個 bug：

1. 記錄 evidence。
2. 在 Final 回報。
3. 不得順手一起修。
4. 等待下一個獨立任務。

不同 correctness bug 優先使用獨立 commit。

Correctness fix 與 performance optimization 原則上不得混在同一 commit。

---

# 6. Production 架構邊界

Production 主要資料與控制流程應維持：

```text
Captured Frame
→ Prompt Observer
→ Specialized Detectors
→ Evidence Fusion
→ Action-aware FSM
→ ActionIntent
→ Safety Gate
→ WindowsActionSink
→ SendInput
```

Detector / perception 層負責：

```text
觀察畫面
產生 evidence
產生 confidence / qualification information
```

FSM / runtime lifecycle 負責：

```text
state
physical opportunity
action ownership
recovery
exactly-once
```

Safety Gate 負責：

```text
是否允許執行動作
```

WindowsActionSink 負責：

```text
真正 OS input emission
```

不得繞過既有 FSM / Safety / ActionSink，直接從 detector 發送遊戲輸入。

---

# 7. Windows 輸入安全邊界

Production action 必須維持：

- foreground-only
- 正確 target process
- 正確 target HWND / window
- process/window validity
- integrity check
- action allowlist
- exactly-once ownership
- F12 panic / emergency stop

不得新增或使用：

- background input
- `PostMessage`
- `SendMessage` 模擬遊戲操作
- driver input
- kernel input injection
- DLL injection
- process injection
- hook injection
- automatic elevation
- auto-focus
- 強制搶 foreground
- anti-cheat bypass
- stealth / evasion mechanism

不得為了提高成功率而削弱 foreground 或 Safety check。

Codex 不得自行對真實遊戲執行 SendInput。

所有自動測試必須使用 mock / fake ActionSink。

---

# 8. Action lifecycle authoritative semantics

必須明確區分：

```text
opportunity_created
proposal_created
scheduled
emission_started
action_applied
visual_acknowledged
terminal
```

`action_applied` 的 authoritative 語意：

```text
完整 OS input emission 已完成
```

不是：

```text
畫面已產生 visual ACK
```

Visual ACK 與 OS emission 是兩個不同概念。

Runtime 自己是否已經送出 input，應由 Runtime action lifecycle 判斷。

如果 Runtime 已有 authoritative：

```text
scheduled
emission_started
action_applied
```

不得再使用畫面視覺效果推測：

```text
Runtime 是否已經輸入過
```

並覆蓋 authoritative lifecycle。

---

# 9. Physical opportunity identity

重複出現的 physical gameplay opportunity 不得僅依：

```text
RuntimeState + ActionIntent
```

作為永久 action identity。

如果同類 action 可以在不同 physical episode 重複出現，identity 必須能區分不同 physical opportunity。

例如現有 architecture 可能包含：

```text
ready:<n>:START_HOOK
get_episode:<n>:COLLECT:attempt:<m>
cast opportunity id
press opportunity id
```

實際名稱以 code 為準。

核心 invariant：

```text
同一 physical opportunity
→ 最多一次真正 OS emission

新的 physical opportunity
→ 必須可以建立新的合法 action ownership
```

不得讓前一 physical episode 的：

```text
consumed marker
_actions_applied
dedupe identity
terminal lifecycle
proposal identity
```

污染新的 physical episode。

任何 identity 設計修改都必須檢查：

```text
cross-episode collision
stale ownership
dedupe collision
unbounded identity growth
```

---

# 10. Exactly-once

任何自動輸入功能都必須維持 exactly-once。

同一 physical opportunity：

```text
最多一次實際 OS emission。
```

必須考慮：

- repeated frame
- repeated detector evidence
- temporal confirmation
- scheduler retry
- recovery
- `SYNC_REQUIRED`
- foreground restore
- late visual ACK
- timeout
- stale proposal
- stale action marker

如果 OS emission 尚未開始：

不得只因 proposal / schedule bookkeeping 就錯誤認為 action 已經執行。

如果 OS emission 已開始或完成：

同一 physical opportunity 不得因 recovery 再送第二次。

---

# 11. Recovery coherence

任何 recovery 都必須同時保持：

```text
FSM state
+
physical episode identity
+
action lifecycle ownership
```

一致。

禁止出現：

```text
FSM 已恢復到可操作 state
+
action lifecycle 已 terminal / consumed
+
沒有 future service
```

造成永久 inert state。

也禁止：

```text
recovery
→ 不斷重新建立 opportunity
→ 無限 SendInput
```

Recovery 必須 bounded。

如果某 episode：

```text
actual OS emission = 0
```

不得無證據地等同於：

```text
action 已成功執行
```

如果該 physical opportunity 已有實際 emission：

不得因 sync/recovery 重新武裝相同 action。

Recovery 如果無法安全建立 coherent lifecycle：

應：

```text
保持 SYNC_REQUIRED
或
進入既有 bounded fail-closed path
```

而不是只強制改 FSM state。

---

# 12. Liveness

Production 不得永久停留於：

```text
存在 authoritative opportunity
+
action 尚未完成
+
沒有 future service
+
沒有 retry deadline
+
沒有 recovery deadline
+
沒有 bounded safe-stop
```

重要 Runtime state 應在有限時間內進入：

- action execution
- valid cancellation
- valid terminal
- recovery
- `SYNC_REQUIRED`
- bounded safe-stop

不得用 infinite retry 解決 liveness。

遇到 rare long-run stall，應建立相對應的 liveness invariant telemetry。

例如：

```text
state active
+
opportunity valid
+
action not emitted
+
no future work
```

應可被低頻 anomaly event 捕捉。

---

# 13. Detector / qualification 修改原則

除非 Task Prompt 明確要求，禁止修改：

- visual threshold
- LAB threshold
- HSV threshold
- Canny/Hough threshold
- geometry threshold
- confidence threshold
- occupancy threshold
- temporal consensus required count
- completeness required count
- detector activation cadence
- capture FPS
- action timing

遇到 detector miss：

先取得 evidence。

不得因單一 Live observation直接放寬 threshold。

必須先區分可能失敗層：

```text
capture / sampling miss
detector activation miss
locator miss
geometry miss
occupancy / foreground extraction miss
glyph classification miss
temporal qualification miss
lifecycle/action miss
```

然後只修改真正失敗的層級。

---

# 14. Visual detection 與 Runtime ownership 必須分離

視覺元件可以判斷：

```text
UI 是否出現
glyph 狀態
panel 狀態
animation
visual reaction
```

但不得把：

```text
visual state
```

直接等同於：

```text
Runtime action ownership
```

例如：

```text
glyph glow
color change
green/yellow/gold/red/cyan state
visual animation
```

不等同於：

```text
Runtime 已經執行 SendInput
```

除非 Task Prompt 正在修改這個 contract，否則視覺 telemetry 不應取代 authoritative action lifecycle。

---

# 15. Evidence 與 capture 原則

所有 detector / diagnostic recorder 優先共用同一張 Runtime captured frame。

禁止為 diagnostic 額外執行第二次畫面 capture，例如：

- 第二次 `mss.grab`
- `pyautogui.screenshot`
- `PIL.ImageGrab`
- 額外 Windows screenshot API

如果 recorder 所在位置沒有 frame：

應做最小必要資料傳遞，

而不是重新 capture。

Diagnostic evidence 優先保存：

```text
Runtime frame 中的原始 ROI
+
detector 已經產生的 metadata
```

---

# 16. Raw evidence 優先

當 detector 完全 miss 時，metadata 往往不足。

因此需要診斷 visual miss 時：

優先保留 preprocessing 前的 raw ROI。

Raw ROI 必須能讓人工回答：

```text
遊戲畫面當時到底有沒有該 UI？
```

不得只保存：

- threshold mask
- overlay
- detector crop after transformations

而沒有原始 ROI。

Overlay / masks 可以作為 secondary evidence。

---

# 17. Production evidence 必須 bounded

Production 不得持續大量寫入：

- full-screen screenshot
- PNG per frame
- JSON per frame
- video
- MP4

Production 禁止持續使用 `VideoWriter`。

Evidence recorder 優先使用：

```text
低頻 sampling
+
small ROI
+
bounded RAM ring buffer
+
只有 anomaly trigger 才落盤
```

必須有：

- max samples
- max episodes
- bounded RAM
- bounded disk output

正常成功 flow 應盡量：

```text
0 evidence disk write
```

Diagnostic I/O failure 必須：

```text
fail-open for observability
```

也就是：

- 可以記 bounded warning
- 不得因此讓 Production action/runtime crash
- 不得改變 gameplay control flow

---

# 18. Diagnostic observability 不得改變 Runtime behavior

純 diagnostic / telemetry 元件不得改變：

- detector qualification
- evidence fusion
- FSM transition
- ActionIntent
- action scheduling
- Safety
- ActionSink
- SendInput

除非當次 Task Prompt 明確就是修改該 lifecycle。

對 recorder 類功能，應盡可能建立：

```text
recorder disabled
vs
recorder enabled
```

behavioral-equivalence regression。

應比較至少：

- FSM transitions
- qualification
- ActionIntent
- ActionRequest
- mock ActionSink calls

Recorder 只能增加 telemetry / evidence side effect。

---

# 19. Logging 規則

Production 不得新增大量 per-frame log。

優先記錄：

- state boundary
- physical episode boundary
- action lifecycle boundary
- recovery
- timeout
- anomaly
- invariant violation
- evidence dump summary

高頻 frame metadata 如果只為 anomaly 使用：

應優先保存在 bounded RAM buffer，

而不是每 frame 寫 log。

---

# 20. Offline test 與 Live evidence 必須分開

以下不能稱為 Live proof：

- pytest
- mock
- fake clock
- prerecorded fixture
- offline image
- replay
- deterministic simulation
- agent_check

這些只能稱為：

```text
Offline deterministic verification
```

Live proof 必須來自：

```text
Production Live canary / long-run session
```

Codex Final 必須明確區分：

```text
offline verified
```

與：

```text
still requires Live verification
```

不得因 full pytest 通過就宣稱 Production 已修復。

---

# 21. Codex 不得自行操作真實遊戲

Codex 不得：

- 啟動黑色沙漠
- 自行操作遊戲
- 執行真實 SendInput
- 為了驗證而控制使用者桌面

Live canary 由使用者執行。

Codex 可以準備：

- deterministic tests
- fake ActionSink
- fixture
- telemetry
- diagnostics
- Live 驗證指標

但不得將 offline test 描述為 Live。

---

# 22. 基本驗證流程

每次修改後：

先跑本次變更相關 focused tests。

之後原則上執行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\agent_check.py
git diff --check
```

如果其中某項因環境問題不能執行：

不得假裝成功。

Final 必須明確回報：

- 哪一項未執行
- 原因
- 對可信度的影響

如果 test / `agent_check` 產生 tracked/generated artifact：

不得混入 commit。

應只恢復由工具自動產生、且與本次 source 修改無關的 generated tracked file。

不得藉此 reset 使用者工作。

---

# 23. Regression 要求

Bug fix 必須盡可能新增可以重現真實 root cause 的 deterministic regression。

Regression 不得只測：

```text
最後結果變成正確
```

還應測真正 failure boundary，例如：

- identity collision
- stale ownership
- stale `_actions_applied`
- zero-emission terminal
- proposal loss
- lifecycle recovery
- scheduler service hole
- timeout
- detector miss
- exactly-once
- stale certificate
- episode reappearance

Regression 應盡量走真正 Production call path。

避免只用：

```text
force_state(final_state)
```

直接塞出預期 state，

除非該測試本來就是針對 low-level FSM。

---

# 24. Stress test 原則

對 identity / recovery / exactly-once 類 bug：

應考慮 deterministic stress，例如：

```text
1000 episodes
```

檢查：

- duplicate emission = 0
- missed action = 0
- permanent inert state = 0
- unbounded identity growth = 0
- unbounded memory growth = 0

Stress test 的數量依 Task Prompt 與 architecture 調整。

不得為了追求固定數字而建立不合理測試。

---

# 25. Performance 修改規則

Correctness 尚未收斂時，不得順手做 performance optimization。

Performance 修改必須：

1. 獨立 Task。
2. 獨立 commit。
3. 有 before / after measurement。
4. 不降低 recognition correctness。
5. 不削弱 safety。
6. 不增加 race / liveness hole。
7. 最後需要 Live canary。

Microbenchmark 只證明 microbenchmark。

不得因單一 function benchmark 變快就宣稱 Production CPU 明顯下降。

---

# 26. Timing / FPS 修改規則

Action timing、capture FPS、detector cadence 都屬於 Production behavior。

除非 Task Prompt 明確要求，不得修改。

修改時必須：

- 單獨 commit
- 保留 safety responsiveness
- 保留 F12 responsiveness
- 不降低 critical state detector sampling
- 有前後 measurement
- 有 Live canary

不得用：

```text
拉長 timeout
增加 retries
提高 retries
降低 confirmation
```

掩蓋 correctness / lifecycle bug。

---

# 27. PERFECT / inferred gameplay path

如果目前沒有 explicit PERFECT visual detector：

不得把：

```text
no PRESS
no GET
```

自動宣稱為：

```text
PERFECT
```

若只能由行為推論，名稱必須表達 inference，例如：

```text
perfect_path_inferred
```

除非 Task Prompt 明確要求新增 PERFECT detector，否則不得因診斷需求順手建立新的 gameplay classification。

---

# 28. PRESS 特別原則

Production PRESS 的 authoritative high-level flow應保持：

```text
visual observation
→ temporal qualification
→ completeness
→ frozen sequence
→ PRESS opportunity
→ Safety
→ ActionSink
→ SendInput
```

一旦 authoritative sequence 已 freeze：

後續 detector drift 不得修改該 frozen sequence。

同一 physical PRESS episode：

```text
最多一次 PRESS_SEQUENCE emission
```

新的 physical PRESS episode：

必須能取得新的 opportunity identity。

如果 PRESS 已完成 qualification，但沒有 emission：

應追 action lifecycle。

如果畫面有 PRESS、但完全沒有 qualification：

應追：

```text
sampling
activation
locator
occupancy
classification
temporal qualifier
```

不得混為同一問題。

---

# 29. RESULT_PENDING 診斷原則

`no PRESS + no GET` 本身不必然是 anomaly。

因為可能存在：

- PERFECT
- no-result
- 合法 recovery
- 其他正常 post-HOOK path

只有 Task Prompt 明確指定的 terminal / timeout 才能作為 anomaly trigger。

例如：

```text
result_pending_maximum_timeout
```

才代表：

```text
Runtime 在 bounded window 內無法取得合法後續狀態
```

如果要蒐證：

優先保存整個 RESULT_PENDING window 的 bounded raw ROI history，

而不是只在 timeout 最後一刻截圖。

---

# 30. 現有 subsystem 不得因旁支任務被改動

如果 Task Prompt 只處理某 subsystem，例如：

```text
PRESS diagnostics
```

則不得順手修改：

- READY
- START_HOOK
- Hook
- CAST
- GET
- COLLECT
- Safety
- SendInput
- capture
- resolution handling
- durability/equipment

反之亦然。

只有 Task Prompt 明確跨 subsystem 時才能跨界修改。

---

# 31. Threshold tuning 需要 evidence

任何 threshold tuning 前至少回答：

```text
哪個 raw evidence 證明目前 threshold 是 root cause？
```

若回答不了：

不得調 threshold。

尤其不得僅因：

```text
某次 cyan / blue / red / green UI miss
```

直接假設顏色 threshold 太嚴。

必須先確認 detector 實際失敗層。

---

# 32. Cleanup

Cleanup 必須獨立於 correctness bug。

只有在：

```text
Production correctness 已長測穩定
```

後，才考慮：

- legacy path removal
- duplicate orchestration cleanup
- production shadow removal
- redundant telemetry cleanup
- dead-code cleanup

Cleanup 前必須先確認：

- 是否仍被 offline diagnostic 使用
- 是否仍被 compatibility tests 使用
- 是否仍是 fallback
- 是否仍提供 observability

不得因看起來沒用就刪除。

---

# 33. Documentation

只有當修改會改變：

- architectural invariant
- Production CLI
- runtime behavior contract
- long-term developer rule

才需要更新永久 docs。

不要為每個小 bug 建立大量一次性文件。

Live session evidence 原則上不提交 repository。

---

# 34. Commit 原則

一個 commit 應對應一個清楚目的。

Commit message 應描述實際改動，例如：

```text
fix: recover collect lifecycle after terminal sync
fix: bound start hook liveness after ready
feat: capture result pending timeout evidence
```

不得使用模糊訊息，例如：

```text
fix stuff
update
changes
misc
```

Commit 前：

- focused tests通過
- full validation按 Task 要求完成
- `git diff --check` 通過
- 確認 staged files只有本次必要檔案

---

# 35. Push 原則

只有在：

- Task Prompt允許 push
- tests完成
- commit完成
- branch確認正確

後才 push。

Push 前再次確認：

```powershell
git branch --show-current
git status
```

不得 push main，除非使用者當次明確要求。

不得 force push。

如果 push 被安全審核要求重新授權：

停止並等待使用者。

---

# 36. Final 回報最低內容

修改任務完成後，Final 至少要回報：

- root cause
- precise blocking code path / identity / condition
- 修改內容
- 為什麼修改能解決 root cause
- 保留哪些 safety invariant
- exactly-once 是否保持
- recovery 是否 bounded
- 新增哪些 regression
- focused test 結果
- full pytest 結果
- `agent_check` 結果
- `git diff --check` 結果
- commit hash
- push 結果
- git status
- 尚未驗證的部分
- 下一步需要什麼 Live canary

如果本次只是 observability feature：

則 root cause 可以是：

```text
尚未確認，本次目標是取得 evidence
```

不得硬湊 root cause。

---

# 37. Live canary 回報規則

Codex 不得宣稱：

```text
修正已在 Production 驗證
```

除非使用者之後提供實際 Live session evidence。

Offline 完成後應說：

```text
尚待 Live canary 驗證。
```

Live 驗證指標應具體，例如：

```text
ready_liveness_invariant_violation_count = 0
duplicate emission = 0
result_pending timeout evidence successfully saved
```

而不是只說：

```text
看起來正常
```

---

# 38. READ-ONLY AUDIT 模式

如果 Task Prompt 明確指定：

```text
READ-ONLY AUDIT
```

則禁止：

- 修改 source
- 修改 tests
- 修改 docs
- commit
- push

只允許：

- 搜尋
- trace
- 分析
- 執行不修改 repository 的安全檢查
- 回報 root cause
- 建議 patch

如果 audit 中發現 root cause：

不得自行開始修改。

等待下一個 Task Prompt。

---

# 39. Evidence-only 任務

如果 Task Prompt 明確表示：

```text
只新增 observability / diagnostic evidence
```

則必須保證 recorder enabled / disabled 不改變：

- detector outcome
- qualifier outcome
- FSM
- ActionIntent
- ActionRequest
- mock ActionSink call sequence

Evidence-only 任務不得：

- 修 threshold
- 改 FSM
- 改 action timing
- 改 detector cadence
- 改 recovery
- 改 timeout

除非 Task Prompt 明確要求。

---

# 40. Bounded anomaly recorder 標準

新增 anomaly recorder 時至少定義：

```text
trigger
start boundary
end boundary
sample cadence
max samples
max episodes
ROI
RAM upper bound
disk upper bound
normal-flow discard behavior
I/O failure behavior
```

只有 anomaly trigger 發生才落盤。

正常 episode優先：

```text
RAM only
→ terminal normal
→ discard
```

不得留下大量 normal screenshots。

---

# 41. Anomaly recorder 必須保留原始時間脈絡

如果 anomaly 只在 timeout 最後才知道：

不得只保存 timeout 當下最後一張 frame。

必須考慮：

```text
bounded pre-event ring buffer
```

以保存 anomaly 發生前的 visual history。

例如 10 秒 timeout：

應確保 buffer 可以涵蓋大部分或全部 10 秒窗口，而不是只有最後 0.5 秒。

---

# 42. Recorder metadata 不得造假

如果某 detector 在該 frame 沒有執行：

記：

```text
not_evaluated
```

或：

```text
press_detector_executed = false
```

不得：

- 重跑 detector後假裝是原本結果
- 填入 default 值讓它看似有 observation
- 將 unavailable 當 false detection

診斷資料必須能區分：

```text
false
```

與：

```text
not evaluated
```

---

# 43. Frame reuse / mutation

如果 Runtime frame buffer 可能被下一 tick reuse / mutate：

diagnostic ring buffer 不得只保存 unsafe reference。

只 copy 必要的小 ROI。

不得因避免 mutation 而 copy整張 full-resolution frame，除非 Task Prompt 明確證明有必要。

---

# 44. Memory / storage 計算

新增 buffer / recorder 時：

Final 必須能計算或合理估算：

```text
max samples
× ROI dimensions
× channels
× bytes/channel
```

得到 worst-case RAM。

不得只說：

```text
記憶體負擔很小
```

而不給數值。

同理，如果會落盤：

需要有 episode cap。

---

# 45. Production video 規則

Production 不得新增：

- continuous MP4
- continuous AVI
- `cv2.VideoWriter`
- full-session screen recording

若需要 visual evidence：

使用 bounded anomaly ROI still images。

Offline debug tool 是否可使用影片依現有專案規則處理，但不得因此偷偷打開 Production video。

---

# 46. Existing behavior compatibility

修某 subsystem 時：

除 root cause 對應 behavior 外，其他已驗證 behavior應保持。

應優先測：

```text
before/after behavioral equivalence
```

特別是：

- Safety
- exactly-once
- valid cancellation
- normal episode
- new physical episode
- foreground loss
- panic
- timeout
- recovery

不得只測新增案例。

---

# 47. 不用為 rare bug 放棄 fail-closed

如果 visual evidence不確定：

仍然優先：

```text
abstain
```

而不是猜測輸入。

但 fail-closed 不能演變為：

```text
永久 inert state
```

也就是：

```text
uncertain → abstain
```

與：

```text
liveness → bounded recovery
```

必須同時成立。

---

# 48. Production safety 與 liveness 同時存在

不得以：

```text
避免 duplicate
```

為理由造成：

```text
新的 physical opportunity 永遠不能執行
```

也不得以：

```text
避免 stall
```

為理由造成：

```text
同一 physical opportunity 無限重送
```

正確目標是：

```text
physical identity
+
exactly-once
+
bounded recovery
```

三者同時成立。

---

# 49. Task Prompt 應該保持精簡

`PROJECT_RULES.md` 保存長期、跨任務、穩定規則。

每次 Task Prompt 只應描述：

- 本次目標
- 當前 branch / relevant commit
- Live evidence
- 要追的 code path
- task-specific invariant
- task-specific forbidden subsystem（只有必要時）
- regression scenario
- task-specific Final 額外要求
- commit message
- Live canary目標

Task Prompt 不需要重複本文件已有的：

- Git safety
- SendInput safety
- full pytest 基本規則
- force push禁止
- Live/offline區別
- generic cleanup禁止
- second capture禁止
- bounded evidence原則

除非某一條是本次任務的核心風險，需要特別再次提醒。

---

# 50. Stop condition

完成當次 Task Prompt 後停止。

不得自行開始下一階段。

例如完成 correctness fix 後，不得自行開始：

- 下一個 bug
- performance optimization
- cleanup
- refactor
- PRESS timing tuning
- WAITING FPS tuning
- durability/equipment feature
- resolution handling
- production architecture cleanup

等待使用者下一個明確指示。

---

# 51. 專案目前的開發方法

本專案採用：

```text
一個問題
→ 一個 precise root cause
→ 一個最小修改
→ 一組 deterministic regression
→ 一個 commit
→ 一輪 Live canary
```

rare long-run bug 必須依靠：

```text
bounded telemetry
+
long-duration Production session
```

驗證。

不要因一次短測成功就過早 cleanup 或 performance optimization。

---

# 52. 最終原則

遇到任何不確定情況：

```text
先保留使用者資料
先保留 Production safety
先取得 evidence
再修改
```

永遠不要用「看起來應該」取代 evidence。

永遠不要用「測試有過」取代 Live proof。

永遠不要用「避免重複輸入」造成永久卡死。

永遠不要用「避免卡死」造成無限輸入。

目標始終是：

```text
Correct
Safe
Bounded
Observable
Recoverable
Exactly-once
```
