# 自訂模型

以元件為單位逐一搭建小訊號拓樸，而不是從四個內建模型中選一個。開啟
**小訊號模型模擬與擬合 (SSM Simulation & Fitting)**，點選
**🧩 自訂模型 (🧩 Custom model)** 晶片，再選擇 **🛠 建立模型 (🛠 Build
model)**。

## 功能說明

每一段你畫出的支路，R、L、C，不論任何串並聯組合，都會化簡為兩個具名節點
之間的雙埠網路，再加上核心處一個本質 π 或 T 受控源。求解器
（`custom_model/core.py`）將每個支路填入節點導納矩陣，以 Kron 消去法去除
內部節點，再把得到的 2×2 Y 矩陣轉換為 S 參數，這與 SPICE 類 AC
求解器所走的 netlist→Y→S 路徑相同，並針對頻率向量化運算。

每次編輯後，控制項上方都會即時重繪一張 SVG 示意圖，讓你在模擬前就能看到
正在搭建的拓樸。

建構器由**內而外**依序揭露五個區段，一次一段：元件與本質核心、外質電容、
延遲/埠附加元件、接觸電阻與引線電感、寄生焊墊電容。確認一個區段後即可
展開下一個；或開啟 **📂 修改既有模型（內建或 .json） (📂 Modify an
existing model (built-in or a .json))**，載入某個內建模型（Cheng T/π、
Xu T、Kun-Yang HEMT）或先前儲存的 `.json`，該情況下**每個**區段會一次
全部展開，隨處可編輯。

## 實作範例：Cheng's T 加上一段射極延遲支路

從 Cheng's T 拓樸開始，在本質射極與接觸網路的 Re/Le 之間加入一個並聯的
電阻與電容，這與內建的 Kun-Yang HEMT 模型在其源極腿上使用的
R_delay∥C_delay 支路屬於同一種手法。

1. 在 **📂 修改既有模型（內建或 .json）** 下，從**從內建模型開始
   (Start from a built-in model)** 選擇 **Cheng, T (current-source T
   HBT)**，點選**載入 (Load)**。

![修改既有模型：內建拓樸下拉選單與載入按鈕](../assets/simfit/custom_build_modify.png)

載入預設模型會一次展開所有區段並重繪示意圖：α·Ie 受控源背後的 Cbex、
Cbe∥Rbe、Cbc∥Rbc、Cbcx、接觸網路，以及三個焊墊電容：

![已載入 Cheng T 預設模型的即時示意圖](../assets/simfit/custom_schematic_chengt.png)

2. 捲動到 **3 · 延遲 / 埠附加元件 (3 · Delay / port extras)**，點選
   **E delay** 晶片，即共用（射極）延遲支路，預設為空。

![第 3 節，已選取 E delay 晶片，「No components yet」](../assets/simfit/custom_emitter_empty.png)

3. 點選**➕ 新增串聯段 (➕ Add series step)**，再於該段內點選
   **➕ R（並聯） (➕ R (parallel))** 與 **➕ C（並聯） (➕ C (parallel))**，
   兩者會落在同一個並聯群組中，彼此並聯，並與射極腿其餘部分串聯。
4. 將兩個 **name（名稱）** 欄位改名為 `r_delay_e` 與 `c_delay_e`。

示意圖現在會顯示這條新支路，位於本質射極（`Ie` 電流源接點）與
`Re`/`Le` 之間：

![示意圖顯示 r_delay_e 並聯 c_delay_e 插入於 Re/Le 之上](../assets/simfit/custom_emitter_added.png)

`r_delay_e` 與 `c_delay_e` 會滾降射極的高頻響應，`c_delay_e` 在其轉角
頻率以上會將 `r_delay_e` 短路，為射極接觸路徑加入一個獨立於本質
Cbe/Rbe 接面的極點。這與 Kun-Yang HEMT 模型用來擬合看似頻率相依的源極
電阻，是同一種手法。

## 儲存、重複使用與擬合

建構器底部有 **💾 儲存 / 使用模型 (💾 Save / use model)**：

- **⬇ 下載 .json (⬇ Download .json)** 將拓樸（結構與名稱，不含數值）
  下載到本機。
- **📤 傳送至模擬 / 擬合頁 (📤 Send to Simulate / Fit view)** 直接將其
  載入**模擬 / 擬合 (Simulate & fit)** 分頁，不必重新上傳，同時一併下載
  該 `.json`。

**模擬 / 擬合 (Simulate & fit)** 分頁會為每個具名元件各要求一個數值，
依由外而內排列（寄生焊墊 → 引線 L → 接觸電阻 R → 外質電容 → 埠/延遲
附加元件 → 本質核心），`r_delay_e` 與 `c_delay_e` 會出現在
**Port / delay extras** 之下：

![元件數值：Port / delay extras 下的 r_delay_e (Ω) 與 c_delay_e (fF) 輸入欄](../assets/simfit/custom_values_panel.png)

輸入數值後立即模擬，右側顯示 Smith 圖與 fT/fmax，版面配置與每個內建模型
相同：

![修改後模型的 Smith 圖與 fT/fmax](../assets/simfit/custom_sim_result.png)

在**📂 擬合至量測元件（選用） (📂 Fit to a measured device)**（頁面頂端，
與內建模型共用同一個控制項）上傳量測 `.s2p`，即可疊圖並進行擬合；
殘差顯示、[視覺化調諧與自動調諧](tuning.md)在自訂模型上運作方式完全相同，
只是背後由通用的網表求解器取代解析公式。

下一步：[視覺化與自動調諧](tuning.md) · [圖表控制](charts.md)。
