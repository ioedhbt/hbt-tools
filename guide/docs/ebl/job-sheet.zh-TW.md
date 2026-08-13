# 工作編號表

[劑量時間測試](dose-test.md)、[首次曝光](first-exposure.md)與
[二次曝光](second-exposure.md)裡需要的每一個欄位，依 JEOL ELS-7000 軟體要求的
順序整理：先 **Job 1**，再 **Job 2**，最後 **Job 3**。挑選你要執行的曝光
對應的章節，每個章節都是獨立完整的。

## 劑量時間測試

在真正元件上曝光之前請先進行劑量測試。它以不同的曝光時間多次繪製同一個
圖案，讓你能挑出最好的曝光時間：曝光量必須足夠，使圖案能在一定容差內曝光，
同時維持可接受的良率。

**Job 1**

* `Chip origin x, y (mm)`（晶片原點 x, y）：網格的起點，而不是元件在載台上
  的位置。
* `Cel origin x, y (mm)`（Cel 原點 x, y）：`.cel` 檔案放置的位置。

**Job 2**

* `Chip size (μm)`（晶片尺寸）：網格邊長，通常為 600 μm。測試圖案通常
  會落在單一網格內，網格在 job3 中再倍增。
* `Dotmap`（點陣圖）：要繪製多少「像素」。

解析度 (Resolution) = 晶片尺寸 (Chip Size) / 點陣圖 (Dotmap)。範例：晶片尺寸
= 600 μm，點陣圖 = 60000，解析度 = 10 nm。

**Job 3**

* `Increment dx, dy (mm)`（增量 dx, dy）：每個網格之間的距離，通常為
  `0.6,0.6` mm
* `Grid Count Nx, Ny`（網格數量 Nx, Ny）：x 與 y 方向的網格數量。使用
  `5,5` 會產生一個 5 乘 5 的網格。
* `Initial Shift x, y (mm)`（初始位移 x, y）：依晶片位置而定的位置位移。
* `Initial dose`（初始劑量）：你想測試的最小劑量（駐留時間，dwell time）。
  例如 5 µs。
* `Dose increment`（劑量增量）：網格之間的劑量增量。若有 5 個網格，初始
  劑量為 5 µs，劑量增量為 0.2 µs，EBL 會以 5 µs、5.2 µs、5.4 µs、5.6 µs
  與 5.8 µs 曝光。

`modx, mody` 為 `0,0`，`focus shift` 為 `0`。

## 首次曝光

首次曝光是指不進行對準的一般曝光。請先確認你已有所需的劑量時間。一旦輸入
你在 SEM 模式下找到的角落（或位置），應用程式就會自動將遮罩圖案置中於
元件上。

**Job 1**

* `Chip origin x, y (mm)`（晶片原點 x, y）：網格的起點，而不是元件在載台上
  的位置。
* `Cel origin x, y (mm)`（Cel 原點 x, y）：`.cel` 檔案放置的位置。
* `Grid Count Nx, Ny`（網格數量 Nx, Ny）：要容納你遮罩中所有圖案所需的網格
  數量。

**Job 2**

* `Chip size (μm)`（晶片尺寸）：網格邊長，通常為 600 μm。
* `Dotmap`（點陣圖）：要繪製多少「像素」。

解析度 (Resolution) = 晶片尺寸 (Chip Size) / 點陣圖 (Dotmap)。範例：晶片尺寸
= 600 μm，點陣圖 = 60000，解析度 = 10 nm。

**Job 3**

* `Shift x, y` (mm)（位移 x, y）：需要多少位置位移，才能把 `job1` 中設定的
  位置對合到 SEM 偵測到的實際元件位置。

## 二次曝光

來源：[second-exposure.md](second-exposure.md)，`second_exposure.gds`，圖層
`L2/D1`（`L2/D2` 用相同的角落、標記與 Cel Origin 重複一次）。

**Job 1**

* `Chip origin x, y (mm)`（晶片原點 x, y）：網格的起點，而不是元件在載台上
  的位置。
* `Cel origin x, y (mm)`（Cel 原點 x, y）：`.cel` 檔案放置的位置。
* `Grid Count Nx, Ny`（網格數量 Nx, Ny）：要容納你遮罩中所有圖案所需的網格
  數量。
* `Reg-2 Mark, Mark 1 (M1) and Mark 2 (M2) x, y (mm)`（標記 1 (M1) 與標記 2
  (M2) x, y）：你原始 `.gds` 檔案中的標記位置，加上 `Cel origin x, y (mm)`
  的數值。

**Job 2**

* `Chip size (μm)`（晶片尺寸）：網格邊長，通常為 600 μm。
* `Dotmap`（點陣圖）：要繪製多少「像素」。

解析度 (Resolution) = 晶片尺寸 (Chip Size) / 點陣圖 (Dotmap)。範例：晶片尺寸
= 600 μm，點陣圖 = 60000，解析度 = 10 nm。

**Job 3**

* `Shift x, y` (mm)（位移 x, y）：需要多少位置位移，才能把 `job1` 中設定的
  位置對合到 SEM 偵測到的實際元件位置。
