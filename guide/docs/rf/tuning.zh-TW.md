# 視覺化與自動調諧

兩者都位於[擬合模式](sim-fit.md#3-擬合量測元件)出現的 **Fine-tune**
展開區內，需要已載入量測元件並選好模型晶片。可用滑桿手動調整參數，
或讓調諧器代為搜尋。

以下內容都從萃取流程產生的模型開始，並套用在原始（未去嵌入）的量測
資料上，讓 pad 與引線寄生參數也有可以擬合的對象。τB 與 τC 應是渡越時間
擬合的值，而非 Cheng 的解析拆分，詳見[微調與檢查](ssm-finetune.md)的說明。
若你的起始殘差高達數百百分比，代表模型尚未載入：檢查元件名稱旁的
`📌 cache from …` 標籤，或再次從
[萃取頁](ssm-finetune.md#將模型傳送出去)把元件交接過來。

## 1. Visual Tuning（視覺化調諧），滑桿即時預覽

1. 開啟 **🎯 Visual Tuning（視覺化調諧）**，在**要拖曳的參數
   (Parameters to slide)** 下選擇一個或多個參數。每個參數各有自己的
   Min/Step/Max 列與一支滑桿。
2. 拖動滑桿。預覽用的 Smith 圖、預覽標題中的殘差，以及 fT/fmax 面板都會
   隨之更新，頁面上其餘部分在你提交數值前都不會改變。

![Rbc 滑桿從 10 kΩ 掃到 500 kΩ，Smith 曲線與 fT/fmax 面板即時更新](../assets/simfit/tuning_rbc.gif)

Rbc 在其範圍內掃動。滑桿下方的「`main: … → preview: …`」列，顯示目前
**Fine-tune** 中生效的值，與你正在預覽的值兩者對比；接受之前不會寫回任何
內容。留意 S12（紅色）與 S22（橘色）：在範圍下端，基極-集極分流會把兩者
都拉離量測值，殘差隨之上升。

![alpha0 滑桿從 0.95 掃到 0.99，Smith 曲線與 fT/fmax 面板即時更新](../assets/simfit/tuning_alpha.gif)

α₀ 的 Min/Step/Max 被夾在合理範圍內，不論輸入什麼都不會超過 1。α₀ 會
同時牽動 S21 與 |h21|² 的滾降。

參數選擇器上方有兩種模式：

- **🎯 All sweep（全參數掃描）**：每次拖動都模擬出一個全新的點。精確，
  但流暢度取決於應用程式往返速度。
- **⚡ Smooth sweep（平滑掃描）**：預先算好整個範圍，再順暢地播放。
  適合需要反覆來回拖動同一個參數的情況。

## 2. Auto Tuning（自動調諧），一鍵完成

開啟 **🔧 自動調諧至最小殘差 (Auto Tuning for Minimum Residuals)**
   → **🪜 全自動調諧 (🪜 Full Auto Tune (recommended))**。**要擬合的參數
   (Parameters to fit)** 預設為每個標準參數，Cpbe/Cpce/Cpbc 與
   Lb/Lc/Le（寄生 pad/引線組）預設**未選取**，因為它們來自開路/短路
   去嵌入，而不是靠擬合本質元件得到。

![Full Auto Tune 卡片：預設的 Parameters to fit 範圍排除 Cpxx/Lx，一個「以 CPU 評估 (Evaluate with CPU)」按鈕](../assets/simfit/tuning_scope.png)

點選**以 CPU 評估 (Evaluate with CPU)**。上方置頂的殘差列與 Smith/
   fT-fmax 圖表顯示起始狀態：

![Full Auto Tune 執行前：起始殘差，下方為起始的 Smith 圖與 fT/fmax 面板](../assets/simfit/tuning_before_charts.png)

搜尋過程由粗到細，且不會阻擋頁面。按鈕會變成**⏹ 停止 (⏹ Stop)**，
並有一行即時的**目前最佳 (Best so far)** 每個週期更新，直到你按下停止，
或搜尋觸及步進下限：

![Full Auto Tune 執行中：即時週期與評估次數、目前最佳總殘差，下方印出完整參數列](../assets/simfit/tuning_after.png)

**目前最佳 (Best so far)** 這一行會印出每個參數，讓你能看出實際變動了
什麼。從一次可靠的解析萃取出發，自動調諧是在打磨而非搶救。若某個參數
跑到荒謬的數值，代表擬合對它不敏感，而不是元件真的如此。

殘差夠好時按下**⏹ 停止 (⏹ Stop)**，再點選**🏆 使用最佳值 (🏆 Use best
   values)** 把最佳一列的數值寫回 **Fine-tune**。殘差列與主圖表會依
   已提交的數值重新繪製：

![點選 Use best values 之後：改善後的殘差，下方為更新後的 Smith 圖與 fT/fmax 面板](../assets/simfit/tuning_after_charts.png)

Semi-Auto Tune（半自動調諧，暴力法/最佳化/優先排序網格掃描）位於
Full Auto Tune 下方，適合想手動控制搜尋範圍或指標，而非使用一鍵預設值的
情況，本頁不做說明。

### 固定某個參數，不讓它參與擬合

Rbc 與 α₀ 預設在搜尋範圍內。它們決定 fT/fmax 增益（dB）的位置，尤其是
Bode 圖低頻段。自動調諧只最佳化殘差，因此模擬的 fT/fmax 曲線在低頻可能
對不上實驗曲線。所以最好先選定最能反映起始增益（dB）的 Rbc/α₀ 值，再把
這兩個參數移出自動擬合，固定在你要的數值：

開啟**要擬合的參數 (Parameters to fit)**。每個已選取的參數都是一個
   獨立的標籤，各自有自己的刪除圖示。點選 **Rbc (kΩ)** 標籤上的 **×**
   將其移除。

![移除 Rbc 後的 Parameters to fit，原本位於 Cbe 與 Cbc 之間的位置以圈記標出](../assets/simfit/tuning_scope_no_rbc.png)

Rbc 從搜尋範圍中移除，維持你所選定的數值，不再被搜尋。固定 Rbc 後再次
執行調諧，幾乎不影響殘差，卻讓模型保持物理意義。

下一步：[圖表控制](charts.md) · 返回
[模擬與擬合](sim-fit.md)。
