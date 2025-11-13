import csv, os
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pred_csv = Path("results/video_predictions.csv")
test_root = Path("data/dataset/test")

if not pred_csv.exists():
    print("ERROR: results/video_predictions.csv not found. Run infer_video.py first.")
    raise SystemExit(1)

# load predictions (filename, true, pred) or (filename,pred) fallback
df = pd.read_csv(pred_csv)
# prefer columns: filename,true,pred  (we used that earlier)
if 'true' not in df.columns:
    # try with image_predictions style (filename,prediction)
    if 'prediction' in df.columns:
        df = df.rename(columns={'prediction':'pred'})
        df['true'] = None
    else:
        print("Unexpected CSV columns:", df.columns)
        raise SystemExit(1)

# map each basename to its full path and video folder
basename_to_path = {}
for p in test_root.rglob("*"):
    if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png"):
        basename_to_path.setdefault(p.name.lower(), []).append(str(p))

rows = []
for _, r in df.iterrows():
    fname = str(r['filename'])
    key = fname.lower()
    paths = basename_to_path.get(key, [])
    if not paths:
        # not found in test tree; skip
        continue
    # if multiple matches, pick first
    path = paths[0]
    # video id is the parent folder of the file (one level above filename)
    video_id = Path(path).parent.name
    true_label = r.get('true') if 'true' in r else None
    pred_label = r.get('pred') if 'pred' in r else r.get('prediction')
    rows.append((video_id, true_label, pred_label, path))

# group by video_id and majority vote
grp = defaultdict(list)
for vid, true, pred, path in rows:
    grp[vid].append((true, pred, path))

video_results = []
for vid, items in grp.items():
    preds = [str(p[1]) for p in items]
    trues = [p[0] for p in items if p[0] is not None]
    # majority vote
    mv = Counter(preds).most_common(1)[0][0]
    # if true labels differ across frames, pick most common (should be same)
    true_mv = None
    if trues:
        true_mv = Counter(trues).most_common(1)[0][0]
    video_results.append((vid, true_mv, mv, len(items)))

out_df = pd.DataFrame(video_results, columns=["video_id","true","pred","n_frames"])
out_df.to_csv("results/video_metrics_by_video.csv", index=False)

# compute metrics (drop unknown true)
mask = out_df['true'].notnull()
y_true = out_df.loc[mask,'true'].tolist()
y_pred = out_df.loc[mask,'pred'].tolist()

print("Number of videos evaluated:", len(out_df))
if len(y_true)>0:
    print("Per-video accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification report (per-video):\n")
    print(classification_report(y_true, y_pred, digits=4))
    print("\nConfusion matrix (per-video):\n")
    print(confusion_matrix(y_true, y_pred))
else:
    print("No true labels found for videos (true column missing).")
print("\\nSaved results/video_metrics_by_video.csv")
