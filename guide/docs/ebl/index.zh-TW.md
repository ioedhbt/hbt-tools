# 電子束微影計算機基礎

從 GDS 遮罩計算 JEOL ELS-7000 晶片位置與曝光工作。本頁涵蓋開啟工具、載入
遮罩、閱讀檢視器，之後的三個工作流程頁面（[劑量時間測試](dose-test.md)、
[首次曝光](first-exposure.md)、[二次曝光](second-exposure.md)）都是在本頁
的基礎上進行。

## 1. 開啟計算機

大部分情況下，[Streamlit 線上網頁](https://hbt-tools.streamlit.app/)應用程式就足以處理遮罩檔案。如果你的遮罩檔案大於 350 MB，請改在本機執行應用程式：

1. 在電腦上安裝 python。Windows 上安裝時要勾選「add to PATH」。

2. 前往[電子束工具 GitHub 儲存庫](https://github.com/ioedhbt/ebeam)。點選綠色
   「<> Code」按鈕，再點「download zip」。把 zip 解壓縮到一個資料夾。

3. 在該資料夾開啟終端機或命令提示字元，執行
   `python launch_ebl_calculator.py` 或 `python3 launch_ebl_calculator.py`。
   它會安裝頁面所需的套件，並透過瀏覽器啟動應用程式。

![電子束微影計算機頁首，獨立啟動，沒有平台外框，沒有密碼關卡](../assets/ebl/index_header.png)

## 2. 左側電腦設定

設定晶片原點，也就是網格要對齊的位置（預設為 10.0, 10.0）。你可以用其他
數字，但不要用 0, 0；那會讓網格貼到畫面最左邊，你將找不到它。

接著設定 **晶片尺寸 (Chip Size)** 與 **點陣圖 (Dotmap)**。晶片尺寸是每個
網格有多大，點陣圖是每個網格要畫幾個點（像素）。因此曝光的解析度就是
晶片尺寸除以點陣圖。

![左側電腦設定](../assets/ebl/index_left_computer.png)

## 3. 上傳遮罩

在 **GDS 遮罩 (GDS Mask)** 底下的 **上傳 .gds 檔案 (Upload .gds file)** 放入
一個 `.gds` 檔。檔案解析成功後，**頂層元件 (Top Cell)** 與 **要曝光的圖層
(Layer to expose)** 會自動填入。先選頂層元件，再選你想檢視或接下來要曝光
的 `(layer, datatype)` 配對。

![已載入遮罩：元件／圖層選擇器與解析後的尺寸行](../assets/ebl/index_loaded.png)

## 4. 閱讀檢視器

選擇器下方的圖會畫出所選圖層。小型圖層（少於 50,000 個多邊形）會逐一畫出
每一個多邊形：

![GDS 檢視器：精確繪出六個基極焊墊多邊形](../assets/ebl/index_viewer.png)

超過 50,000 個多邊形後，檢視器改為顯示低解析度概覽圖。在上面拖曳框選，
即可重新以完整細節繪出該區域，適合在有上千個元件的遮罩中檢查單一元件。

![圖案太大無法檢視](../assets/ebl/index_large.png)
![顯示框內的圖案](../assets/ebl/index_large2.png)

## 三種使用本頁的方式

遮罩載入後，頁面最下方的 **曝光流程 (Workflow)** 區塊決定頁面接下來要用它
做什麼：

![模式選擇器：劑量時間測試、首次曝光、二次對準](../assets/ebl/index_mode.png)

**劑量時間測試 (Dose Time Testing)** 會以逐格遞增的劑量多次曝光同一個遮罩
圖案，讓你能在真正對元件下手之前，從顯影結果中挑出可用的劑量。見
[劑量時間測試](dose-test.md)。

**首次曝光 (First Exposure)** 將遮罩檔案曝光到元件上，不進行任何標記對準。
你必須事先測試劑量。見[首次曝光](first-exposure.md)。

**二次對準 (Second Alignment)** 將遮罩檔案曝光到已有對準標記的元件上，
利用既有標記對準。你必須事先測試劑量。見[二次曝光](second-exposure.md)。

要知道三種模式依序在 JEOL 系統中該輸入哪些數字，見[工作編號表](job-sheet.md)。

## 設定說明

小提示：開啟 **左側電腦設定 (Left Computer Setup)** 區塊中的 **設定說明
(Setup Instruction)**。它會彙整你的步驟與左側電腦所需的數字。

![設定說明彙整你的步驟與左側電腦所需的數字](../assets/ebl/setup_instruction.png)
