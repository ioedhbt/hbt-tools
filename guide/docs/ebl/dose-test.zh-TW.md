# 劑量時間測試

在同一顆晶片上以一系列劑量曝光同一個圖案，再從 SEM 下讀出顯影後的線寬，藉此
在真正對元件下手前挑出可用的劑量。

如果你還沒開過這個頁面，請先讀[電子束微影計算機基礎](index.md)；本頁是從
已載入遮罩之後接著講。

## 1. 載入遮罩並選圖層

上傳 `.gds` 檔案，然後設定 **要曝光的圖層 (Layer to expose)**。

![GDS 遮罩區塊：已載入 dose_test.gds，選取圖層 L1/D1](../assets/ebl/dose_layer.png)

## 2. 把圖案定位在寫入場中

在 **曝光流程 (Workflow)** 底下選 **劑量時間測試 (Dose Time Testing)**。
在這裡輸入 **job1 中的 Cel 原點 (mm) (Cel Origin (mm) in job1)** 的 x 與 y。
這是遮罩 GDS 座標 `(0,0)` 對應到的實際載台位置。

![Cel Origin x/y 欄位，輸入為 8.610 / 9.710](../assets/ebl/dose_cel.png)

Cel 原點通常是 (9.7, 9.7)，因此圖案位置會被平移 (9.7, 9.7)。

## 3. 網格幾何參數維持預設值

**job3 中的增量 (mm) (Increment (mm) in job3)**（dx、dy）與 **job3 中的
網格數量 (Grid Count in job3)**（Nx、Ny）不需要更改，dx/dy 會自動跟隨晶片
尺寸讓區塊緊密相接，而預設的 5×5 數量正好對應這個範例的劑量階數。**job3
中的初始位移 (mm) (Initial Shift (mm) in job3)** 會自動算出以將陣列置中於
晶片上。

![單一網格 (Single Grid) 與晶片位置與網格 (Chip Position with Grids) 圖，兩者圖案皆置中](../assets/ebl/dose_grids.png)

左圖是在你剛設定的 Cel Origin 上的單一區塊，沒有複製，先在這裡確認遮罩落在
橘色框內。右圖是自動置中位移後，整個晶片上的完整 5×5 陣列。

## 4. 設定劑量遞增

在 **曝光時間計算 (Time Calculator)** 底下，輸入你的 **初始劑量 (μs / dot)
(Initial dose (μs / dot))**、**增量劑量 (μs / grid) (Incremental dose (μs /
grid))** 與 **載台移動時間 (s / grid) (Stage movement time (s / grid))**，
然後按 **計算時間 (Calculate time)**。

![按下計算前的曝光時間計算劑量遞增輸入](../assets/ebl/dose_time_in.png)

每個區塊各自有自己的劑量。左下角區塊劑量最短（初始劑量），右上角區塊最長。

## 5. 讀出每個劑量階的曝光時間

![明細：劑量遞增範圍與預估時間](../assets/ebl/dose_result.png)

要看的是 **劑量遞增 (Dose ramp)** 這一行：`2.000 μs + 0.200 μs/grid → 有效範圍
2.000, 6.800 μs`，也就是 25 個區塊依迭代順序實際用到的最低與最高單區塊劑量。
底部的 **預估時間 (Estimated Time)**（`00:35:35.000`）顯示曝光所需時間。
見[工作編號表](job-sheet.md)看這些數字各自對應到 JEOL 系統中的哪個欄位。

## 設定說明

小提示：開啟 **左側電腦設定 (Left Computer Setup)** 區塊中的 **設定說明
(Setup Instruction)**。它會彙整你的步驟與左側電腦所需的數字。

![設定說明彙整你的步驟與左側電腦所需的數字](../assets/ebl/setup_instruction.png)
