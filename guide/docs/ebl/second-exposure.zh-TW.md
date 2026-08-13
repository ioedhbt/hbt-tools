# 二次曝光

把新圖層對準到晶片上先前曝光已經寫上的標記，讓它相對於既有圖案落在正確的
位置。

如果這是你第一次來這一頁，請先讀[電子束微影計算機基礎](index.md)。

## 1. 載入遮罩並選圖案圖層

上傳你的 `.gds` 檔案，然後設定 **要曝光的圖層 (Layer to expose)**。

![GDS 遮罩區塊：已載入 second_exposure.gds，選取圖層 L2/D1](../assets/ebl/second_layer.png)

## 2. 設定標記的設計位置

在 **曝光流程 (Workflow)** 底下選 **二次對準 (Second Alignment)**，再選
**十字標記位置預設 (Cross-position preset)** 底下的 **自訂 (Custom)**，
在這裡輸入你自己遮罩的標記位置。建議選兩個彼此呈對角線的標記，而不是
垂直或水平排列的兩個。

![十字標記位置預設 = 自訂，已輸入標記 M1 與 M2](../assets/ebl/second_custom.png)

這兩個數字是標記在遮罩設計中的位置 — 在 KLayout 或 ADS 中開啟你的
`.gds` 檔案並輸入。下一步則是告訴工具它們實際落在物理晶片上的哪裡。

## 3. 輸入在晶片上找到的標記位置

在 **晶片上的既有圖案 (Existing Pattern on Chip)** 底下，把 **既有圖案
圖層 (Existing pattern layer)** 設為你的首次曝光圖案。

假設你在 SEM 下找到 M1 標記，其載台位置為 (100.0, 115.0) mm。把兩個數字
輸入 **目標 x 和 y (Target x and y)**。這會把 M1 從遮罩位置移到實際載台
位置，M2 會自動跟隨。將滑鼠停在圖上可確認位置。

![既有圖案圖層 = 標記，目標 x/y 已輸入為找到的位置](../assets/ebl/second_existing.png)

## 4. 調整網格讓兩個標記都在裡面

**job1 中的網格數量 (Grid Count in job1)**（Nx、Ny）決定在 x 與 y 方向建立
幾個網格。如果圖案落在選定網格尺寸之外，應用程式會發出警告：

![二次對準圖案：4x2 自動貼合網格，M2 在外面，紅色警告](../assets/ebl/second_sap_warn.png)

這種情況下，手動修正：

把 **Ny** 從 2 增加到 **4**。

![Ny 調成 4：橫幅消失，兩個標記都在網格內，對位標記位置面板](../assets/ebl/second_sap_fixed.png)

下方的 **job1 中的對位標記位置 (Registration mark position in job1)** 面板
現在會以 job1 載台座標顯示兩個標記：標記 1 (M1) 在 (9.8000, 9.8000)，標記
2 (M2) 在 (11.6000, 11.6000)，即標記設計位置加上 Cel 原點。

## 5. 讀出疊圖結果

捲到 **疊圖 (Overlayed)**。**Mark 1** 與 **Mark 2** 預設為 `M1` / `M2`。
**位移 x (mm) (Shift x (mm))** / **位移 y (mm) (Shift y (mm))** 會自動算出。

![疊圖：Mark 1/2 選擇器，job3 用的自動位移 x/y，最終圖](../assets/ebl/second_overlay.png)

**位移 x/y (Shift x/y)** 就是要寫上工作單 job3 的數字。

## 6. 設定劑量並讀出時間

在 **曝光時間計算 (Time Calculator)** 底下，輸入 **劑量時間 (μs / dot)
(Dose time (μs / dot))** 與 **載台移動時間 (s / grid) (Stage movement
time (s / grid))**（通常約 10~15 秒），然後按 **計算時間 (Calculate time)**。

![按下計算後的明細與預估時間](../assets/ebl/second_result.png)

應用程式會估計曝光所需的時間。但對二次對準而言，在 SEM 下尋找對準標記
通常還會在這個數字之上再多花 1~2 小時。見[工作編號表](job-sheet.md)看
設定中所用數字的彙整。

## 設定說明

小提示：開啟 **左側電腦設定 (Left Computer Setup)** 區塊中的 **設定說明
(Setup Instruction)**。它會彙整你的步驟與左側電腦所需的數字。

![設定說明彙整你的步驟與左側電腦所需的數字](../assets/ebl/setup_instruction.png)
