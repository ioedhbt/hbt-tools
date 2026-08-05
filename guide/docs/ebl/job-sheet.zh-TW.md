# 工作編號表

[劑量時間測試](dose-test.md)、[首次曝光](first-exposure.md)與
[二次曝光](second-exposure.md)裡的每一個數字，依 JEOL ELS-7000 軟體要求的
順序整理：先 **Job 1**，再 **Job 2**，最後 **Job 3**。挑選你要執行的曝光
對應的章節，每個章節都是獨立完整的。**預設值 (default)** 表示保留計算機
的原始數值；**量測值 (measured)** 表示由你在機台上實際找到的數值填入；
**計算值 (computed)** 表示由計算機算出，只需讀出，除了對應的 JEOL 欄位外，
不要在其他任何地方手動輸入。

## 劑量時間測試

來源：[dose-test.md](dose-test.md)，`dose_test.gds`，圖層 `L1/D1`。

**Job 1**

| 欄位 | 數值 | 來源 |
|---|---|---|
| Chip Origin x (mm)（晶片原點 x） | 10.000 | 預設值 |
| Chip Origin y (mm)（晶片原點 y） | 10.000 | 預設值 |
| Cel Origin x (mm)（Cel 原點 x） | 8.610 | 量測值，定位讓遮罩落在寫入場區塊之內 |
| Cel Origin y (mm)（Cel 原點 y） | 9.710 | 量測值，同上 |

**Job 2**

| 欄位 | 數值 | 來源 |
|---|---|---|
| Chip Size (μm)（晶片尺寸） | 600 | 預設值 |
| Dotmap（點陣圖） | 60000 | 預設值 |
| Resolution（解析度，唯讀） | 10 nm | 計算值 |

**Job 3**

| 欄位 | 數值 | 來源 |
|---|---|---|
| Increment dx (mm)（增量 dx） | 0.600 | 預設值（= 晶片尺寸） |
| Increment dy (mm)（增量 dy） | 0.600 | 預設值（= 晶片尺寸） |
| Grid Count Nx（網格數量 Nx） | 5 | 預設值 |
| Grid Count Ny（網格數量 Ny） | 5 | 預設值 |
| Initial Shift x (mm)（初始位移 x） | 98.800 | 計算值 |
| Initial Shift y (mm)（初始位移 y） | 109.200 | 計算值 |

**劑量／載台時間**（不是編號的 JEOL 工作欄位，但要在寫入機上設定後才能執行）

| 欄位 | 數值 | 來源 |
|---|---|---|
| Initial dose (μs / dot)（初始劑量） | 2.000 | 預設值 |
| Incremental dose (μs / grid)（增量劑量） | 0.200 | 預設值 |
| Stage movement time (s / grid)（載台移動時間） | 15.00 | 預設值 |
| Estimated Time（預估時間） | 00:35:35.000 | 計算值，僅供規劃參考 |

## 首次曝光

來源：[first-exposure.md](first-exposure.md)，`first_exposure.gds`，圖層
`L1/D2`（`L1/D1` 與 `L1/D3` 用相同的角落與 Cel Origin 重複一次）。

**Job 1**

| 欄位 | 數值 | 來源 |
|---|---|---|
| Chip Origin x (mm)（晶片原點 x） | 10.000 | 預設值 |
| Chip Origin y (mm)（晶片原點 y） | 10.000 | 預設值 |
| Bottom Left (BL) corner x (mm)（左下角落 x） | 105.000 | 量測值，在鏡下找到的晶片角落 |
| Bottom Left (BL) corner y (mm)（左下角落 y） | 115.000 | 量測值 |
| Top Right (TR) corner x (mm)（右上角落 x） | 107.000 | 量測值，在鏡下找到的對準標記 |
| Top Right (TR) corner y (mm)（右上角落 y） | 117.000 | 量測值 |
| Cel Origin x (mm)（Cel 原點 x） | 9.700 | 預設值 |
| Cel Origin y (mm)（Cel 原點 y） | 9.700 | 預設值 |
| Grid Count Nx（網格數量 Nx） | 4 | 計算值，自動貼合遮罩圖層 |
| Grid Count Ny（網格數量 Ny） | 2 | 計算值，自動貼合遮罩圖層 |

左上與右下角落不需要在任何地方輸入，它們由 BL 與 TR 計算而得，並以禁用
狀態顯示在晶片位置圖上。

**Job 2**

| 欄位 | 數值 | 來源 |
|---|---|---|
| Chip Size (μm)（晶片尺寸） | 600 | 預設值 |
| Dotmap（點陣圖） | 60000 | 預設值 |
| Resolution（解析度，唯讀） | 10 nm | 計算值 |

**Job 3**

| 欄位 | 數值 | 來源 |
|---|---|---|
| Shift x (mm)（位移 x） | 95.100 | 計算值，將網格置中於晶片上 |
| Shift y (mm)（位移 y） | 105.700 | 計算值 |

**劑量／載台時間**

| 欄位 | 數值 | 來源 |
|---|---|---|
| Dose time (μs / dot)（劑量時間） | 2.000 | 預設值 |
| Stage movement time (s / grid)（載台移動時間） | 15.00 | 預設值 |
| Estimated Time（預估時間） | 00:00:35.400 | 計算值，僅供規劃參考 |

## 二次曝光

來源：[second-exposure.md](second-exposure.md)，`second_exposure.gds`，
圖層 `L2/D1`（`L2/D2` 用相同的角落、標記與 Cel Origin 重複一次）。

**Job 1**

| 欄位 | 數值 | 來源 |
|---|---|---|
| Chip Origin x (mm)（晶片原點 x） | 10.000 | 預設值 |
| Chip Origin y (mm)（晶片原點 y） | 10.000 | 預設值 |
| Bottom Left (BL) / Top Right (TR) corners（左下／右上角落） | 與首次曝光相同 | 量測值，同一片實體晶片，位置沒有變動 |
| Cel Origin x (mm)（Cel 原點 x） | 9.700 | 預設值 |
| Cel Origin y (mm)（Cel 原點 y） | 9.700 | 預設值 |
| Grid Count Nx（網格數量 Nx） | 4 | 計算值，自動貼合遮罩圖層 |
| Grid Count Ny（網格數量 Ny） | 4 | **手動修正**，自動貼合算出 2，不夠到達標記，因此調高為 4 |
| Reg-2 Mark, Mark 1 (M1) x (mm)（標記 1 x） | 9.8000 | 計算值，標記設計位置加上 Cel Origin |
| Reg-2 Mark, Mark 1 (M1) y (mm)（標記 1 y） | 9.8000 | 計算值 |
| Reg-2 Mark, Mark 2 (M2) x (mm)（標記 2 x） | 11.6000 | 計算值 |
| Reg-2 Mark, Mark 2 (M2) y (mm)（標記 2 y） | 11.6000 | 計算值 |

Reg-2 Mark 的數值取決於你在 SEM 下量到的標記位置（計算機「既有圖案」步驟
中的 Target x/y），若要對準到不同的晶片，需在那裡重新計算。

**Job 2**

| 欄位 | 數值 | 來源 |
|---|---|---|
| Chip Size (μm)（晶片尺寸） | 600 | 預設值 |
| Dotmap（點陣圖） | 60000 | 預設值 |
| Resolution（解析度，唯讀） | 10 nm | 計算值 |

**Job 3**

| 欄位 | 數值 | 來源 |
|---|---|---|
| Shift x (mm)（位移 x） | 0.0010 | 計算值 |
| Shift y (mm)（位移 y） | −0.0005 | 計算值 |

**劑量／載台時間**

| 欄位 | 數值 | 來源 |
|---|---|---|
| Dose time (μs / dot)（劑量時間） | 2.000 | 預設值 |
| Stage movement time (s / grid)（載台移動時間） | 15.00 | 預設值 |
| Estimated Time（預估時間） | 00:03:48.000 | 計算值，**不含**尋找對準標記通常需要的 1, 2 小時 |
