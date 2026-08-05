# 模型萃取：設定

載入元件並選擇小訊號模型拓樸。本頁示範如何使用原始（未去嵌入）示範檔案，
從空白上傳走到完成去嵌入的元件，逐步操作 SSM 萃取頁。

## 1. 上傳原始元件檔案

從側邊欄開啟**剝離法小訊號模型萃取 (Small Signal Model Extraction by
Peeling)**。

1. 在**① 上傳偏壓檔 (① Upload bias files)** 下上傳一個或多個原始偏壓
   `.s2p` 檔案。可使用示範檔案中的
   `raw/vce3.5_ib200u.s2p`，此檔案仍保留焊墊與引線。

![上傳 DUT 檔案](../assets/ssm/setup_upload_main.png)

一次可放入多個偏壓檔案，後續步驟（Z 參數接觸電阻、τ_total 擬合）需要
一個以上的點才能擬合出一條線。

## 2. 上傳 Open 與 Short

Open 與 Short dummy 檔案位於側邊欄，而非主要上傳區。

2. 在側邊欄開啟**啟用元件 dummy 去嵌入 (Enable device-dummy de-embedding)**。

![開啟側邊欄開關](../assets/ssm/setup_sidebar_toggle.png)

畫面會出現兩個檔案欄位：**元件 Open (Dev Open)** 與**元件 Short
(Dev Short)**。將 `open.s2p` 拖入第一個，`short.s2p` 拖入第二個。

![側邊欄中已載入的 Open 與 Short](../assets/ssm/setup_sidebar_files.png)

出現綠色的**✅ Pad 去嵌入已啟用。(Pad de-embedding active.)** 訊息，
確認兩個檔案都已成功解析。

## 3. 執行萃取

3. 在**② 選擇元件與模型 (② Select device & model)** 下選擇使用中的檔案
   （只有在載入超過一個 DUT 檔案時才需要），然後點選**▶ 執行小訊號模型
   萃取 (▶ Run SSM Extraction)**。

![點選執行](../assets/ssm/setup_ready_to_run.png)

## 4. 檢查焊墊電容

萃取頁會開啟於**1：焊墊電容與串聯電感 (1 - Pad Capacitance & Series
Inductance)**，位於**📌 Open/Short Dummy 去嵌入 (📌 Open & Short Dummy
De-embedding)** 展開區內。Open dummy 給出三個焊墊並聯電容：

![Open dummy 電容表](../assets/ssm/setup_open_caps.png)

表格應給出 Cpbe、Cpce 和 Cpbc 的值。如有必要，移動頻率範圍滑桿，使其找到掃描結果前 50% 的中位數，此時 Open 虛擬訊號最乾淨。

## 5. 檢查引線電感

在同一個展開區內往下捲動至 Short dummy 區塊：

![Short dummy 電感表](../assets/ssm/setup_short_leads.png)

預期Lb、Lc、Le。

## 6. 選擇要輸入萃取流程的值

捲動至**3：串聯／存取電阻萃取 (3 - Series / Access Resistance
Extraction)**，開啟 **✏️ 選擇串聯電阻來源 (✏️ Choose series resistance)**。
此面板決定 Rb、Rc、Re 實際輸入模型的值，而非直接使用上方 Short dummy
的結果。

![選擇串聯電阻來源面板](../assets/ssm/setup_preoverride.png)

Rb、Rc、Re 各自有自己的來源選項。完成步驟 1 後，唯一的選項是
**Custom**，預設為 0 Ω，若已知數值可直接在右側的 Rb/Rc/Re 欄位輸入。
一旦執行了 Cold-HBT、Z-parameter 或 open-collector 方法（下一頁），這裡
會出現額外的選項，各自標示其萃取值，選中的那一個就是往下傳遞給所有模型
的值。**Short（步驟 1b）刻意未提供此選項**，應用程式將 Short 方法視為
四種接觸電阻萃取方法中最不可靠的一種，因此預設不會被選中。

焊墊電容（Cpbe/Cpce/Cpbc）與引線電感（Lb/Lc/Le）在此面板中不可編輯；
它們永遠依循步驟 1 的 Open/Short 萃取結果（或你在「Override Open
Capacitances」/「Override Short Lead Values」展開區內調整過的值）。

下一步：[接觸電阻](ssm-access-r.md)。
