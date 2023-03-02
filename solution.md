
First of all, I would like to thank our hosts for organizing the competition.
It was a task I've never solved before, and it was both educational and a lot of fun trying different approaches!


# Summary
(image)


# Model Detail
## 3D-CNN (cv: x.xxxx)
- backbone: r3d_18 (from torchvision: https://pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html#torchvision.models.video.R3D_18_Weights)
- use 63 frames(20fps)
- predict 19 steps
- train every 9 steps
- StepLR Scheduler(~2epochs: lr=1e-3/1e-4)

## 2.5D3D-CNN (cv: x.xxxx)
- Almost same as DFL's 1st solution by Team Hydrogen (https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/359932)
- backbone: legacy_seresnet34
- use 123 frames(20fps)
- predict 3 frames
- down sampling (g: 10%, contact: 30%)
- label smoothing (0.1-0.9)

## Both 3D, 2.5D3D
- Linear layers for g and contact
```python
# x: (bs, cnn_features)
x_contact = model_contact(x)  # (bs, n_predict_frames)
x_g = model_g(x)  # (bs, n_predict_frames)
not_is_g = (is_g == 0)
x = x_contact * not_is_g + x_g * is_g  # (bs, n_predict_frames)
```
- output 3 prediction and calculate loss: only sideline, only endzone, concat sideline-endzone feature.
```python
# pseudo code
def forward(self, x_sideline_image, x_endzone_image):
  x_sideline = cnn(x_sideline_image)  # 
  x_endzone = cnn(x_endzone_image)
  return fc(torch.cat([x_sideline, x_endzone])), fc_sideline(x_sideline), fc_endzone(x_endzone)
```

## LGBM (cv: x.xxxx)
- about 1100 features
- feautres
    - player's distance (tracking, helmet)
    - lag, diff
    - top_n nearest player's distance (n: parameters)
    - number of people within distance n (n: parameters)
- groupby
    - game_play
    - is_g
    - is_same_team
    - number of people within distance n 

## ensemble
I optimized G and contact respectively.

### weight
```python
def ensemble(scores, weights, vote_th, cols):
    weights_ = np.array([weights[f"{col}_weights"] for col in cols])
    weights_ = weights_ / weights_.sum()
    votes = np.sum(scores * weights_, axis=1)
    return votes > vote_th
cols = [
    "score_3d",  
    "score_2.5d", 
    "score_lgbm"
]
cols_ = [
    "score_3d", 
    "score_2.5d", 
    "score_lgbm"
]
def objective(trial):
    weights = {
        f"{col}_weights": trial.suggest_float(f"{col}_weights", low=0, high=1) 
        for col in cols if col != "lgbm"
    }
    weights.update({
        f"{col}_weights": trial.suggest_float(f"{col}_weights", low=0, high=2) 
        for col in cols if col == "lgbm"        
    })
    vote_th = trial.suggest_float("vote_th", low=0, high=1)
    preds = ensemble(scores, weights, vote_th, cols)
    
    mcc = matthews_corrcoef(contacts, preds)
    return mcc
```

### vote
```python
def vote(scores, thresholds, vote_th, cols):
    thresholds_ = np.array([thresholds[f"{col}_th"] for col in cols])
    votes = np.sum(scores > thresholds_, axis=1)
    return votes >= vote_th
cols = [
    "score_3d",  "score_3d", "score_sideline_3d", "score_endzone_3d", 
    "score_2.5d", "score_2.5d", "score_sideline_2.5d", "score_endzone_2.5d", 
    "score_lgbm", "score_lgbm", "score_lgbm"
]
cols_ = [
    "score_3d", "score_3d2", "score_sideline_3d", "score_endzone_3d", 
    "score_2.5d", "score_2.5d2", "score_sideline_2.5d", "score_endzone_2.5d", 
    "score_lgbm", "score_lgbm2", "score_lgbm3"
]
def objective(trial):
    thresholds = {f"{col}_th": trial.suggest_float(f"{col}_th", low=0, high=1, step=0.001) for col in cols_}
    vote_th = trial.suggest_int("vote_th", low=1, high=len(cols), step=1)
    preds = vote(scores, thresholds, vote_th, cols_)
    
    mcc = matthews_corrcoef(contacts, preds)
    return mcc
```

## What worked for me
- image preprocessing
  - draw bbox -> draw bbox and paint out
  - use 2 colors(g, contact) -> use 3 colors(g, same team contact, different team contact)
  - crop the image with keeping the aspect ratio
  ```python
  bbox_left_ratio = 4.5
  bbox_right_ratio = 4.5
  bbox_top_ratio = 4.5
  bbox_down_ratio = 2.25
  for col in ["x", "y", "width", "height"]:
      df[col] = df[[f"{col}_1", f"{col}_2"]].mean(axis=1)
  df["bbox_size"] = df[["width", "height"]].mean(axis=1)
  df["bbox_size"] = df.groupby(["view", "step", "game_play"])["bbox_size"].transform("mean")

  series = df.iloc[0]  # sample
  left = int(series["x"] - series["bbox_size"] * bbox_left_ratio)
  right = int(series["x"] + series["bbox_size"] * bbox_right_ratio)
  top = int(series["y"] + series["bbox_size"] * bbox_top_ratio)
  down = int(series["y"] - series["bbox_size"] * bbox_down_ratio)
  img = img[down:top, left:right]
  img = cv2.resize(img, (128, 96))
  ```
  
- StepLR with warmup scheduler
- label smoothing (worked for 2.5D3D, but not worked for 3D)

## What not worked for me
- Transformers
  - use top 100~400 features of lgbm feature importances
  - tuned hard but got cv 0.02 lower than lgbm.
- 2D->1D CNN
  - contact score is same as 2.5D3D, 3D but very poor G score in my work.
- interpolate bbox
- 

## Other
- tools
  (add movie)