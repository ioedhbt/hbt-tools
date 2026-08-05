# 批次去嵌入

對每個已上傳的 DUT 檔案一次執行與前一頁相同的 Open/Short 去嵌入，而不必
逐一處理。

## 使用時機

當有多個偏壓點要對應同一組 Open/Short 校準時，使用**批次去嵌入 (Batch
De-embed)** 分頁，這是元件掃描的常見情況（多個 `Ib` 或 `Vce` 點、同一組
探針焊墊佈局）。它會執行一次
[開路/短路去嵌入](deembedding.md)頁所述的相同模型化萃取，並一次套用到
主要上傳區的每個檔案。

## 設定

與三步驟頁相同：在側邊欄開啟**② 元件 Dummy（Open-Short） (② Device
Dummy (Open-Short))**，拖入 Open 與 Short 檔案，再透過主要的**上傳 DUT
.s2p / .csv 檔案 (Upload DUT .s2p / .csv files)** 區塊上傳所有要去嵌入的
DUT 檔案。載入全部 7 個示範偏壓掃描檔案後，批次去嵌入分頁會為每個檔案列出
一個子分頁：

![7 個已上傳元件各自的逐檔結果分頁](../assets/rf/batch_perfile_tabs.png)

每個子分頁都有與[單一元件走查](deembedding.md#元件的前後對比)
相同的 fT/fmax 卡片與 Bode/Smith 圖。Cpbe/Cpce/Cpbc/Lb/Lc/Le 覆寫欄位與
模型計算/量測值來源選擇，會同樣套用到每個檔案；只有一組校準，而非每個
元件各一組。

## 輸出 ZIP

1. 點選**📥 下載去嵌入後的量測檔案 (📥 Download de-embedded measurement
   files)**。

    ![下載去嵌入後量測檔案按鈕](../assets/rf/batch_zip_download.png)

ZIP 內每個已上傳檔案各對應一個 `.s2p`，命名為 `<原始檔名>_deemb.s2p`
（`deemb_preext_vce3.5_ib280u.s2p` → `deemb_preext_vce3.5_ib280u_deemb.s2p`）。
每個檔案的檔頭都帶有從中扣除的寄生值：Cpbe、Cpce、Cpbc 以 fF 為單位，
Lb、Lc、Le 以 pH 為單位，Rb、Rc、Re 以 Ω 為單位，因此僅憑檔案本身就能
追溯它經過的去嵌入處理。

## 將元件批次傳送出去

同一個容器也提供傳送至 SSM 頁面的功能，但是以批次方式：從下拉選單選擇一個
**主要元件 (Primary device)**，然後

- **→ 小訊號模型萃取 (→ SSM Extraction)** 會傳送主要元件*以及*其他所有
  去嵌入後的檔案作為額外資料，萃取頁的 Z 參數、冷 HBT 與 τ_total 方法都
  可以使用它們。
- **→ 模擬與擬合 (→ Simulation & Fitting)** 只傳送主要元件。

這與[單一元件走查](deembedding.md#元件的前後對比)中顯示的去
嵌入結果是同一個容器，只是背後由每個已上傳檔案而非單一檔案支撐。
