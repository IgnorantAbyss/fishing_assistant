# PROJECT_RULES.md

# Fishing Assistant — Project Rules

本文件是本 repository 所有 Codex / AI 修改工作的永久規則。

任何任務開始前都必須先閱讀並遵守本文件。

如果使用者當次 Prompt 與本文件衝突：

1. 不得自行猜測。
2. 明確指出衝突。
3. 停止有風險的修改。
4. 等待使用者決定。

除非當次 Prompt 明確要求修改本文件，否則不得自行改寫 PROJECT_RULES.md。

---

# 1. 工作原則

優先順序：

1. Correctness
2. Safety
3. Liveness
4. Observability
5. Performance
6. Cleanup / elegance

不得為了：

- 效能
- 程式碼漂亮
- 減少行數
- cleanup
- abstraction
- refactor

破壞 correctness、safety 或已驗證的 production behavior。

修 bug 時優先：

- 找出 precise root cause
- 證明實際 code path
- 做最小範圍修改
- 建立 deterministic regression
- 保留既有安全 invariant

不得只根據症狀猜測後直接 patch。

---

# 2. 語言與回報

與使用者的 Final 回報使用繁體中文。

程式碼、class、function、event、field、commit message 保留專案原本英文命名。

對尚未證明的事情必須區分：

- confirmed
- high-confidence inference
- hypothesis
- unknown

不得把推測寫成已證實事實。

---

# 3. Git 安全規則

每個修改任務開始前至少確認：

```powershell
git status
git branch --show-current
git log -5 --oneline --decorate
git remote -v