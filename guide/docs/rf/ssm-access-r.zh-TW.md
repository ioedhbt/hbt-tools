# 模型萃取：接觸電阻

在本質擬合之前先取得 Re、Rb 與 Rc。後續每一步都依賴這三個數字，值得花幾
分鐘處理好。

在萃取頁開啟**🍊 存取電阻萃取 (🍊 Access Resistance Extraction)**。

!!! warning "順序很重要"
    先執行 **Z-parameter 方法求 Re**，再執行 Cold-HBT 求 Rb 與 Rc。冷態
    萃取把 Re 當作*輸入值*。若先開啟 Cold-HBT，其 **Re (Ω)** 欄位會顯示
    `0.0000`，在你毫無察覺的情況下，用完全沒有射極電阻擬合出 Rb 與 Rc。

## 步驟 1：Z-parameter 方法求 Re

Re 取自 Re(Z₁₂) 對 1/I_E 在整個偏壓掃描上的截距。開始前先載入多個偏壓
點，單一個點無法擬合出一條線。

1. 表格列出每個已載入的檔案。勾選要納入的檔案，並輸入各自的射極電流。
   使用 **I_E = I_C + I_B**，而非 I_C。

    ![Z-parameter 偏壓表，每個檔案有核取方塊、IE (mA) 輸入欄與 Re(Z12) 讀值](../assets/ssm/accessr_zparam_widget.png)

    冷態點（I_B = 0）不屬於此擬合；它會提供下方步驟 2 的 Cold-HBT 萃取
    使用。

2. 從截距讀出 **Re**。

    ![Re(Z12) 對 1/IE 圖，含擬合線與標出的 Re 截距](../assets/ssm/accessr_zparam_fit.png)

    將 Re 帶入步驟 2。

!!! tip
    幾乎重疊的偏壓點合起來對擬合的約束力並不比單一個點更強。若截距看起來
    不穩定，將偏壓掃描拉開一些。

## 步驟 2：Cold-HBT 求 Rb 與 Rc

「冷態」量測是指元件未施加偏壓時的量測。兩個接面都關閉時，一旦知道
Re，Rb 與 Rc 便可解出。

1. 將 **Cold-HBT 來源 (Cold-HBT source)** 設為**📂 從已載入檔案 (📂 From
   loaded files)**，並從下拉選單選擇你的冷態（無偏壓）量測。若該檔案不在
   已載入的 DUT 中，可在此直接上傳。

    ![Cold-HBT 來源選擇器設為「從已載入檔案」，並選取冷態量測](../assets/ssm/accessr_cold_source.png)

2. 檢查 Cold-HBT 下方的 **Re (Ω)** 欄位。它應該已經帶有步驟 1 得到的
   Re。若顯示 `0.0000`，代表 Z-parameter 擬合尚未執行：回頭先完成它。

3. 開啟**📊 互動式參數萃取 (📊 Interactive Parameter Extraction)**。步驟
   2 萃取 Cex、Cbc 與 Rbi；Rb 與 Rc 都相依於它們。若你修改某個數值，
   下面的每一步都會沿用修改後的值。

4. 步驟 5 現在給出 **Rb** 與 **Rc**。

    ![步驟 5 Rb 與 Rc 面板，含頻率範圍滑桿、平台圖、數值框與統計晶片](../assets/ssm/accessr_cold_rbrc.png)

    1. **頻率範圍**：將其收窄至曲線平坦的頻段。
    2. 紅色虛線是目前使用中的數值。
    3. 數值框，直接輸入即可覆寫。
    4. 下方的晶片可由統計值填入數值框，點選即可套用。

    兩條曲線都應在平坦的頻段上收斂到各自的虛線。當平均值與中位數差異
    懸殊時，把頻率範圍拖到平坦的區段，再由該處取值。

## 應該得到的結果

完成兩個步驟後，你應該得到：

- Rb (= Rpb)
- Rc (= Rpc)
- Re (= Rpe)
- Rbi（冷態）
- Cbe（冷態）
- Cbc（冷態）
- Cex

## Open-collector

第三種方法在集極開路的狀態下驅動基極-射極接面。當你有這項量測、又沒有
可用的冷態掃描時使用它；它是 Cold-HBT 步驟的替代方案，而非額外附加項目。

---

下一步：[本質模型](ssm-intrinsic.md)。
