# EEF bound cost 檢視與 CPU 驗證計畫

> NOT for main: see `cpu_verify/README.md`.

## 現況：已重新設計

原本的 margin 式 corridor cost 已在本分支的函式庫 commit 中改為 reference frame 式的 tool pose
bound：錨點可以是 goal、start 或 base，加上 offset，每個分量（x, y, z, swing_x, swing_y, twist）
都可以選填上下限，另有選填的 `max_swing`。下表各項目的處理方式：

| ID | 狀態 |
|---|---|
| D1 | 已修：梯度精確、在 base frame，由 `test_cost_eef_bound.py` 的有限差分測試與本目錄的回歸檢查涵蓋 |
| D2 | 已移除：`disable_margin` 隨舊 API 刪除 |
| D3 | 已移除：PyTorch 參考類別刪除，改用 `tool_pose_bound_cost` 只做 forward 比對 |
| D4 | 已修：kernel 透過 `idxs_link_pose` 取 goal |
| D5 | 已修：cost manager 已接上，並加進 IK、trajopt、MPC 與 metrics 的設定檔 |
| R1 | 已修：buffer 預先配置；有 goal/start 資料時一律傳入；kernel 檢查索引範圍 |
| R2 | 已改：start 錨點改由 solver 對起始狀態做 FK |
| R3 | 已改：平方 hinge |
| R4、R6 | 已改：swing-twist 分解，梯度精確 |
| R5 | 保留「取違反最小的 goal」的語意（使用者決定） |
| R7 | 已清理（重寫的檔案沒有 BOM） |

以下為原始的檢視紀錄。

檢視對象：`origin/dev/eef_cost` @ `dac5bd7`
（`cost_eef_bound.py`、`cost_eef_bound_cfg.py`、`cost_eef_bound_py.py`、`wp_torch_eef_bound.py`、
`cost_manager_robot.py`）。

所有「已確認」項目都在 CPU 上以 Warp `cpu` device 重現。對應的檢查在
`test_eef_bound_cpu.py`，以 `xfail(strict=True)` 標示。

## 已確認的缺陷

| ID | 位置 | 問題 | 影響 |
|---|---|---|---|
| D1 | `wp_torch_eef_bound.py:226-230` | 旋轉梯度寫成 `g ⊗ (local_c ⊗ (v,0)) = c ⊗ (v_goal,0)`，`v` 仍在 goal frame，沒有轉到 world frame。`ToolPoseCost` 的慣例是先轉到 world frame（`transform_vector(g, ·)`），再算 `c ⊗ (v_world,0)`；kinematics backward（`quaternion_gradient_to_angular_velocity` → `xyz_rot_backward`）也把它當 world-frame 向量使用。 | 梯度方向與真值的 cosine 正好等於 `cos(goal yaw)`：goal 轉 90° 時梯度與真值正交，轉 180° 時方向完全相反。只要 goal 不是 identity，旋轉 corridor 就會把關節往錯的方向推。 |
| D2 | `cost_eef_bound.py:209-210`（`_py.py` 同） | `disable_margin=True` 用 `zeros_like(margins)`，把負值（停用的軸）也變成 0。 | 文件說「只對啟用的軸收緊」，實際上所有軸都變成寬度 0 的硬約束。 |
| D3 | `cost_eef_bound_py.py:290-291` | PyTorch reference 的 start（t=0）沒有 `detach()`；Warp 版則把 start 當常數。 | reference 的梯度會漏進 `position[:, 0]`，因此不能當梯度 ground truth。現有測試沒抓到，是因為測試資料剛好讓 begin 側不啟動。 |
| D4 | `wp_torch_eef_bound.py:165, 321` | goal 直接用 rollout batch 索引 `b`，沒有用 `goal.idxs_link_pose`（`ToolPoseCost` 有用）。 | `num_seeds > 1` 時 goal batch 小於 rollout batch，`reshape` 會丟 `RuntimeError`。若兩者剛好相等但對應關係不同，則會悄悄讀到錯的 goal。 |
| D5 | `cost_manager_robot.py:301-311` | 計算 hook 整段被註解。註解中的呼叫還是舊的 4-tensor 簽名，也沒傳 goal index。 | 設定 `eef_bound_cfg` 只會建立 cost，solver 實際上從不評估它。直接取消註解會 `TypeError`。 |

## 風險與設計問題（未寫成 xfail）

| ID | 位置 | 說明 | 建議 |
|---|---|---|---|
| R1 | `cost_eef_bound.py:212`、`wp_torch_eef_bound.py:345` | 每次 forward 都會 `torch.tensor(...)`（H2D 複製）和 `wp.zeros(...)`（Warp 配置）。CUDA graph capture 時前者通常會報錯，後者會綁定 capture 當下的記憶體。 | 把 offset 與 `out_cost` 預先配置在 `setup_batch_tensors`。這項需要 GPU 驗證（見下方「需要 GPU 的部分」）。 |
| R2 | `cost_eef_bound.py:204` | corridor 起點取軌跡的 `t=0`。IK（`H=1`）時起點就是自己，begin 側永遠不會啟動。trajopt/MPC 的 `t=0` 是否就是固定的 start state，需要依 transition model 確認。 | 從 goal registry 的 current state 取起點，或在 `update_margins` 時一併給起點。 |
| R3 | `warp_bound_util` 對比 | 使用 L1 hinge，梯度在邊界上從 0 跳到 ±1（見 `corridor_1d.png`）。其他 bound cost 用的是平方 hinge。 | 考慮改用平方 hinge，或讓它可設定，以利 L-BFGS line search。 |
| R4 | `wp_torch_eef_bound.py:221-237` | 忽略 log map 的 Jacobian，所以修好 D1 之後，偏軸旋轉在 2.8 rad 時梯度的 cosine 約 0.82（`rotation_gradient.png` 右圖）。 | 小角度可以接受，但要寫進文件；大角度 corridor 需要補 `J_l^{-1}`。 |
| R5 | kernel 的 goalset 迴圈 | 對每個 goal 取 corridor cost 最小者，與 `ToolPoseCost` 選中的 goalset index 無關。 | 可能沿著另一個 goal 的 corridor 約束。建議改用 tool pose cost 的 `goalset_idx`。 |
| R6 | 旋轉 corridor | rotvec 的各分量並不是獨立的軸，大角度時「逐軸 corridor」沒有幾何意義。 | 寫進文件或限制角度。 |
| R7 | 雜項 | `cost_eef_bound.py`、`cost_eef_bound_cfg.py` 開頭有 UTF-8 BOM；`wp_torch_eef_bound.py:39` 在 import 時呼叫 `wp.init()`；有未使用的 import（`Optional`、`init_warp`）。 | 清理。 |
| R8 | 現有測試 | `test_warp_cost_gradient_matches_pytorch` 用 identity 姿態，旋轉 margin 不啟動，兩邊的四元數梯度都是 0。PyTorch 給的是歐氏四元數梯度，和 cuRobo 的 `c ⊗ (v,0)` 慣例不同，旋轉啟動後兩者本來就不該直接比較。 | 改用 world-frame 有限差分當真值（本 harness 的做法）。 |

## 驗證計畫（全部可在無 GPU 環境執行）

1. **Forward 一致性**：在隨機的多 link、多 goal、旋轉 goal、部分停用軸、per-link weight 下，比較 Warp(cpu) 與 PyTorch reference（`test_forward_warp_matches_torch_random`，圖 `warp_vs_torch.png`）。
2. **梯度真值**：以 world-frame 中央差分當 ground truth，不依賴 reference 的 autograd。
   - 位置：隨機 goal/start，並以兩個步長的差分是否一致自動略過 kink 附近的點（`test_position_gradient_matches_finite_difference`）。
   - 旋轉：先把 cost 的四元數梯度經過 `quat_grad_to_world_omega`（逐行對應 CUDA `quaternion_gradient_to_angular_velocity`）轉成機器人實際使用的 world-frame 向量，再和差分比較（D1）。
3. **缺陷回歸**：D1–D4 各有一個 strict xfail。修正時拿掉 marker，它就成為回歸測試。我已在本地暫時套用 D1–D3 的修正確認過：三者都會 XPASS（strict 模式下變成失敗、提醒移除 marker），原有的 24 個測試仍全數通過。
4. **cost manager 層級**（D4/D5）：`gather_goal_for_seeds` 模擬 `idxs_link_pose` 的 goal 取用方式，`test_gathered_goal_matches_per_problem_evaluation` 驗證這個轉接等價於逐題計算。修 D5 時，hook 應以相同方式傳入 goal index（或讓 kernel 直接接收 `idxs_goal`）。
5. **視覺化**（matplotlib Agg，headless）：
   - `corridor_1d.png`：逐軸掃描 cost 與梯度，kernel 對照有限差分，並畫出 corridor 範圍。
   - `trajectory_xy.png`：goal frame 俯視圖，畫出 corridor 方框、依 cost 上色的軌跡點，以及 -gradient 箭頭。
   - `rotation_gradient.png`：左圖為 D1（cosine 對 goal yaw），右圖為修正後 R4 的近似誤差。
   - `warp_vs_torch.png`：forward 散佈圖。

   之後若要 3D 互動，可以用專案既有的 `viser` 依賴輸出軌跡與 corridor box。它同樣只需要 CPU。

### 需要 GPU 的部分（CPU 無法取代）

- R1：CUDA graph capture 與 replay。
- 經由 curobolib CUDA kernel 的 kinematics backward。CPU 端已用 Python mirror 驗證梯度的慣例，但沒有驗證 kernel 本身。
- 實際 solver（IK/trajopt）啟用 `eef_bound` 後的收斂行為。

## 不進 main 的保證

詳見 `cpu_verify/README.md`。重點如下：

- 所有 harness 檔案都在頂層 `cpu_verify/`，不會被打包，也不在 pytest `testpaths` 內。
- `curobo/` 內沒有任何 CPU 專用的 hook。Warp 本來就支援 `cpu` device，函式庫不需要改。
- `cpu-verify-guard.yml` 會擋下任何帶有 harness 的、進入 `main` 的 PR。
- 修正只以「只改 `curobo/`」的 commit port 到 `dev/eef_cost`，不合併這個分支。
