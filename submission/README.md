# MarkUs submission layout

All hand-in files live under **`submission/`** in three subfolders (prediction / code for zip / PDF).  
Root source files stay at the repo root; **`sync_submission.py`** copies what you need into here.

| Subfolder | Purpose |
| --------- | ------- |
| **`01_prediction_markus/`** | Upload these three: `pred.py`, `model_state.json`, `model_weights.npz`. |
| **`02_report_code_zip/`** | Zip **everything inside** this folder → upload as **`code.zip`**. |
| **`03_report_pdf/`** | Put **`report.pdf`** here (or copy in) before upload. |

## Steps (from repository root)

1. Regenerate prediction artifacts (if needed):
   ```bash
   python export_model.py
   ```
2. Copy the right files into each folder:
   ```bash
   python submission/sync_submission.py
   ```
3. Upload using the table above (prediction slot vs report `code.zip` vs report PDF).

## Notes

- **`code.zip`**: must contain the files inside `02_report_code_zip/` (e.g. `pred.py`, `appendix_code/`, …). Do **not** nest an extra top-level directory unless you intend to.
- **Size**: `model_state.json` + `model_weights.npz` must stay **≤ 10 MB** combined.
- Re-run **`sync_submission.py`** after you change any `.py` or `appendix_code/` so the hand-in folder stays up to date.
