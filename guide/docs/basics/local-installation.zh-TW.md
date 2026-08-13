# 本機安裝

多數情況下，[streamlit 線上網頁應用程式](https://hbt-tools.streamlit.app/) 已足夠。需要處理大型光罩檔的更多 RAM、GPU 加速或取消上傳上限時，再於本機安裝。

## 1. 安裝 Python

從 [python.org](https://www.python.org/downloads/) 安裝 **Python 3.9+**。在 Windows 上，安裝時請勾選 **「Add python.exe to PATH」**。

## 2. 下載程式碼

1. 前往 [HBT tools](https://github.com/ioedhbt/hbt-tools) Github 儲存庫。
2. 點擊綠色的 **「<> Code」** 按鈕，再點擊 **Download ZIP**。
3. 將 zip 解壓縮到一個資料夾。

## 3. 啟動

在專案資料夾中開啟終端機並執行：

```bash
python3 LAUNCH_Tool.py
```

在 Windows 上，直接雙擊 `LAUNCH_Tool.py` 亦可。首次執行需數分鐘進行設定，然後在瀏覽器中開啟應用程式。本機啟動無需密碼驗證。

!!! tip
    電子束計算機可獨立執行（不經入口網站）：在 Windows 上雙擊
    `launch_ebl_calculator.py`，或在 macOS/Linux 執行
    `python3 launch_ebl_calculator.py`。

## 4. 選用：NVIDIA GPU 加速

SSM 調諧掃描在 NVIDIA GPU 上會快上許多。請從 NVIDIA 官方頁面安裝 **CUDA 12 或 13**：
<https://developer.nvidia.com/cuda-downloads>。然後再次執行啟動程式——它會偵測 CUDA 並自動安裝對應的 GPU 套件。沒有 GPU？所有功能仍可在 CPU 上運作。
