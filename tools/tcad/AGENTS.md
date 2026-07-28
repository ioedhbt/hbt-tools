# tools/dc/ · tools/tcad/ · tools/data/

DC and TCAD measurement pages. Small, independent, pandas-heavy.

| page | reads |
|---|---|
| `dc/b1500a_plot.py` | B1500A `.xlsx` — curve viewer, parameter extraction, TLM |
| `dc/hp4155a_plot.py` | HP/Agilent 4155A dumps — quick SMU plots |
| `tcad/gummel_analyzer.py` | TonyPlot Gummel CSV vs the built-in UIUC reference |
| `data/csv_process.py` | batch CSV/CITI conversion; hands workbooks to DC Analysis |

## Gotchas

- **SMU dumps may be whitespace- or comma-delimited.** Use `sep=r"\s+"` and
  decide by column count. Never `delim_whitespace=True` — it was removed in
  pandas 2.2/3.x, and the old `except`-and-retry hid the failure as one garbage
  column with no error shown.
- **Excel sheet names must go through `unique_sheet_name`.** They are derived
  from filenames and truncated to 31 chars; openpyxl does not raise on a
  duplicate, it *merges* the frames into one sheet. Lab names routinely
  collide past 31 chars.
- **Uploads live inside `if page == ...` branches.** Streamlit garbage-collects
  a widget's state on any run that skips it, so switching sub-pages loses them.
  The keep-alive re-assignment trick used in `IOED_Tool_Web.py` does **not**
  work for `st.file_uploader` (Streamlit refuses value assignment). Cache the
  uploaded *bytes* under a plain non-widget key instead — see
  `b1500a_upload_cache`.
- **Two selected files can share a name.** Key dicts on
  `common.widgets.dedupe_upload_names`, not on `f.name`.
- `data/csv_process.py` sends to DC Analysis via `common.handoff.send_dc()`.
