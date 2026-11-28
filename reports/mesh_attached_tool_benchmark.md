# 扁平夾持工具在障礙物附近的規劃：sphere 判定 vs mesh 判定

日期：2026-11-28　分支：`dev/mesh_success_followups`
腳本：`benchmark/mesh_attached_tool_benchmark.py`（共用 `benchmark/mesh_slot_benchmark.py` 的輔助函式）

## 問題

夾持的工具是扁平的，sphere 擬合只有兩種選擇：

- 很多顆小球：貼合，但計算量大。
- 少量大球：計算量小，但球比工具本體大很多。

這份實驗看的是少量大球的情況：球已經碰到孔壁或牆、工具本體的 mesh 卻還沒碰到時，規劃能不能成功。

## 實驗條件

| 項目 | 設定 |
|---|---|
| GPU | Tesla T4（Kaggle） |
| 機器人 | Franka（`franka.yml`），基座 `panda_link0` |
| 工具 | 長方體，50 × 10 × 200 mm，中心在 `panda_hand` 的 z = 190 mm，沿手的 z 軸 |
| 夾持方式 | `AttachmentManager.attach()`：同時寫入工具 mesh（給 mesh 檢查）與工具 sphere |
| 碰撞檢查 | scene 與 self-collision 兩種 mesh 檢查都開；mesh margin 0 mm |
| graph planner | PRM |
| 每次規劃 | `max_attempts=4`，CUDA graph 開；計時只算規劃呼叫，不含建構與 warmup |
| 樣本 | 每列 20 個案例；同一批案例用於所有球組與兩種判定 |

### 球組

`attach()` 先用 MORPHIT 擬合；cover 組再用 `update()` 換成沿工具軸排列、剛好包住該段工具的球（mesh 保留）。

| 球組 | 球數 | 最大半徑 (mm) | 超出工具本體 (mm) |
|---|---|---|---|
| cover4 | 4 | 36 | 31 |
| cover8 | 8 | 28 | 23 |
| morphit30 | 30 | 14–16 | 9–11 |

MORPHIT 有隨機性，兩次執行的最大半徑分別是 14 與 16 mm。

### 情境

間隙 c 是工具 mesh 到障礙物的距離，測 5、10、20 mm。

- **insert（插孔）**：板上開孔，孔比工具每邊寬 c（例如 c = 5 mm 時孔為 60 × 20 mm），板厚 30 mm。
  起點在孔上方 10 mm，用 `plan_pose_waypoints` 沿直線往下插 50 mm。
  路徑容差 5 mm（另測 10 mm）與 2°，終點容差 2 mm 與 2°。
- **beside（貼牆）**：起點與終點各有一面牆，平行於工具寬面，距離 c。
  牆的高度從夾爪指尖以下 c 處開始，所以只有工具會靠近牆。
  起點與終點水平距離至少 250 mm，中間沒有障礙物。用 `plan_pose`。

目標姿態都是工具朝下、隨機 yaw，位置在 x 0.35–0.65 m、y −0.35–0.35 m 內隨機取樣。
起點與終點的關節位置在無障礙場景下用 IK 求得。
若其他連結比工具更靠近障礙物（量到的 mesh 間隙小於 c），該案例丟棄；本次沒有案例被丟棄。

### 兩種判定

兩種判定跑的是同一條規劃流程：IK、PRM、TrajOpt、驗證。
最佳化代價裡都有 sphere 碰撞，由 sphere 引導；差別只在誰判定碰撞與成功。

- **sphere 判定**：`mesh_only_success=False`。碰撞用 sphere 判定，mesh 檢查另外再加上去。
- **mesh 判定**：`mesh_only_success=True`。被 mesh 取代的 sphere 碰撞條件不再算進成功判定；PRM 的可行性也用 mesh。

成功率是規劃器回傳的 `result.success`，也就是整條流程的成功率。
規劃器沒有另外逐點重算回傳軌跡的 mesh 間隙。

## 結果

路徑容差 5 mm。成功率 (%)，格式為 c = 5 / 10 / 20 mm：

| 情境 | 球組 | sphere 間隙中位數 (mm) | sphere 判定 | mesh 判定 |
|---|---|---|---|---|
| insert | cover4 | −25.7 / −20.7 / −10.7 | 0 / 0 / 0 | 100 / 100 / 100 |
| insert | cover8 | −18.4 / −13.4 / −3.4 | 0 / 0 / 0 | 90 / 100 / 100 |
| insert | morphit30 | −2.2 / 2.8 / 12.8 | 0 / 100 / 100 | 100 / 100 / 100 |
| beside | cover4 | −25.7 / −20.7 / −10.7 | 0 / 0 / 0 | 100 / 100 / 100 |
| beside | cover8 | −18.4 / −13.4 / −3.4 | 0 / 0 / 0 | 100 / 100 / 100 |
| beside | morphit30 | −3.7 / 1.3 / 11.3 | 0 / 100 / 100 | 100 / 100 / 100 |

sphere 間隙是起點與終點上，sphere 到障礙物最近距離的中位數；負值表示球已和障礙物重疊。

路徑容差 10 mm（只測 insert）：

| 球組 | sphere 判定 | mesh 判定 |
|---|---|---|
| cover4 | 0 / 0 / 0 | 100 / 100 / 100 |
| cover8 | 0 / 0 / 0 | 100 / 100 / 100 |
| morphit30 | 0 / 0 / 100 | 100 / 100 / 100 |

### 失敗原因

- **sphere 判定**：幾乎都是 `no_result`，起點或終點一開始就被判為碰撞，規劃器拿不到可用的解。
  - beside、cover8、20 mm：`collision_valid`，軌跡被 sphere 碰撞否決。
  - beside、morphit30、5 mm：19 次 `collision_valid`，1 次 `no_result`。
  - insert、morphit30、10 mm、容差 10 mm：`collision_valid`。這次的 MORPHIT 球較大（16 mm），sphere 間隙只有 2.2 mm。
- **mesh 判定**：唯一的失敗是 insert、cover8、5 mm、路徑容差 5 mm 的 2 次 `path_valid`，是直線路徑容差沒通過，不是碰撞。
  容差放寬到 10 mm 就是 100 %。

### 耗時（中位數，ms）

| 情境 | sphere 判定 | mesh 判定 |
|---|---|---|
| insert | 50–52（多半是很早就失敗） | 64–72 |
| beside | 44–47（有兩列 228、241：改走 PRM 後仍失敗） | 44–48 |

## 另外的發現

- **路徑容差**：先前的短測中，2 mm 路徑容差即使在 50 mm 間隙下也全部 `path_valid` 失敗。
  這是 `plan_pose_waypoints` 本身的限制，與碰撞模型無關；直線插入建議 5–10 mm。
- **mesh margin**：本次為 0，代表 mesh 可以剛好貼到障礙物。實機使用時建議留幾 mm。

## 結論

- 少量大球加上 `mesh_only_success`，在 5–20 mm 的插孔與貼牆情境下，成功率 90–100 %。
- 只要球碰到障礙物，sphere 判定就是 0 %。
- 縮小最佳化用的 sphere（A）或把 sphere 擬合在 mesh 內（B），對這個工具不需要。

## 未驗證

- 沒有逐點獨立驗證回傳軌跡的 mesh 間隙。
- 只測長方體的工具與孔，沒有測實際的工具 mesh。
- 每列 20 個案例；90 % 與 100 % 的差異在這個樣本下不顯著。
