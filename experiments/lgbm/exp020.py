import pandas as pd
import numpy as np
import tqdm
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import copy
import os
import shutil
from datetime import datetime as dt
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from logging import Logger, StreamHandler, Formatter, FileHandler
import logging
import gc
from typing import List
try:
    import mlflow
except Exception as e:
    print(e)
import pickle
from sklearn.metrics import euclidean_distances
import warnings
import json
warnings.filterwarnings("ignore")

pd.set_option("max_row", 1000)
pd.set_option("max_column", 200)

debug = False

def get_near_player(df_distance, w_df, distance_matrix, name):
    for distance_th in [1, 2, 3, 5, 7.5, 10, 15]:
        df_distance[f"n_player_distance_{name}_in_{distance_th}"] = (distance_matrix < distance_th).sum(axis=1)
    for top_n in [1, 2, 3, 5, 7]:
        if len(distance_matrix) <= top_n:
            continue
        col_name = f"distance_top{top_n}"
        df_distance[col_name] = distance_matrix[:, top_n]
        df_distance[f"diff_from_{col_name}"] = w_df["distance"].values - df_distance[col_name].values
    return df_distance


def sin(x):
    ans = np.sin(x / 180 * np.pi)
    return ans

def cos(x):
    ans = np.cos(x / 180 * np.pi)
    return ans


def reduce_mem_usage(df, verbose=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def get_logger(output_dir=None, logging_level=logging.INFO):
    formatter = Formatter("%(asctime)s|%(levelname)s| %(message)s")
    logger = Logger(name="log")
    handler = StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging_level)
    logger.addHandler(handler)
    if output_dir is not None:
        now = dt.now().strftime("%Y%m%d%H%M%S")
        file_handler = FileHandler(f"{output_dir}/{now}.txt")
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class LGBMModel:
    def __init__(self,
                 output_dir: str,
                 logger: Logger,
                 exp_name: str,
                 debug: bool = False,
                 use_features: List[str] = None,
                 params: dict = None,
                 include_helmet_info: bool = True,
                 fast_mode: bool = True):
        self.logger = logger
        if params is None:
            self.params = {
                'objective': 'binary',
                'metrics': 'auc',
                'num_leaves': 32,
                'max_depth': -1,
                'bagging_fraction': 0.7,  # 0.5,
                'feature_fraction': 0.7,
                'bagging_seed': 0,
                'reg_alpha': 1,
                'reg_lambda': 1,
                'random_state': 0,
                'verbosity': -1,
                "n_estimators": 20000,
                "early_stopping_rounds": 100,
                "learning_rate": 0.1,
            }
        else:
            self.params = params
        self.include_helmet_info = include_helmet_info
        self.debug = debug
        self.output_dir = output_dir
        self.exp_name = exp_name
        self.feature_dir = "../../output/preprocess/feature"
        self.model_dir = f"{output_dir}/model.txt"
        self.drop_columns = [
            "contact_id", "game_play", "datetime", "team_1", "position_1", "team_2", "position_2",
            "nfl_player_id_1", "nfl_player_id_2",
            "game_key"
        ]
        self.use_features = use_features
        self.fast_mode = fast_mode
        self.agg_dict = {
            "game_play": {},
            "is_same_team": {},
            "is_g": {},
        }

    def feature_engineering(self,
                            df: pd.DataFrame,
                            inference: bool = True):
        # all
        # df = df[df["distance"].fillna(0) <= 2]
        feature_path = f"{self.feature_dir}/{os.path.basename(__file__).replace('.py', '')}/feature_len{len(df)}_helmet_info_{self.include_helmet_info}.feather"
        json_path = f"{self.feature_dir}/{os.path.basename(__file__).replace('.py', '')}/feature_len{len(df)}_helmet_info_{self.include_helmet_info}.json"
        if os.path.isfile(feature_path) and not inference and not self.debug:
            self.logger.info("load from feature_dir")
            return pd.read_feather(feature_path)
        self.logger.info("Reduce memory usage")
        df = reduce_mem_usage(df)

        df["contact_id"] = df["game_play"] + "_" + df["step"].astype(str) + "_" + df["nfl_player_id_1"].astype(
            str) + "_" + df["nfl_player_id_2"].astype(str)

        self.logger.info("FE1: all features")
        helmet_columns = [
            "left_1", "width_1", "top_1", "height_1", "x_1", "y_1",
            "left_2", "width_2", "top_2", "height_2", "x_2", "y_2",
        ]
        if self.include_helmet_info:
            self.logger.info(f"[aggregate view]before: {len(df)}")
            helmet_columns = [
                "left_1", "width_1", "top_1", "height_1", "x_1", "y_1",
                "left_2", "width_2", "top_2", "height_2", "x_2", "y_2",
            ]
            df_endzone = df[df["view"] == "Endzone"].drop("view", axis=1)
            df_sideline = df[df["view"] == "Sideline"].drop("view", axis=1)
            df_endzone.columns = [f"Endzone_{col}" if col in helmet_columns else col for col in df_endzone.columns]
            df_sideline.columns = [f"Sideline_{col}" if col in helmet_columns else col for col in df_sideline.columns]
            df = df[["contact_id"]].drop_duplicates()
            df = pd.merge(df, df_endzone, how="left")
            df = pd.merge(df, df_sideline, how="left")
        else:
            self.logger.info(f"remove helmet info")
            df = df.drop_duplicates("contact_id")
            df = df.drop(helmet_columns + ["view"], axis=1)
        df = df.sort_values(["game_play", "step", "nfl_player_id_1", "nfl_player_id_2"])
        self.logger.info(f"[aggregate view]after: {df.shape}")

        self.logger.info("nearest_n_player")

        df_distances = []
        for key, w_df in tqdm.tqdm(df.drop_duplicates(["game_play", "nfl_player_id_1", "step"]).groupby(["game_play", "step"])):
            if np.isnan(w_df["x_position_1"].iloc[0]):
                continue
            df_distance = w_df[["game_play", "nfl_player_id_1", "step"]]
            distance_matrix_org = euclidean_distances(w_df[["x_position_1", "y_position_1"]].values)
            distance_matrix = distance_matrix_org.copy()
            distance_matrix.sort(axis=1)
            df_distance = get_near_player(df_distance, w_df, distance_matrix, name="all")

            distance_matrix = distance_matrix_org.copy()
            distance_matrix *= w_df["team_1"].values.reshape(-1, 1) == w_df["team_1"].values.reshape(1, -1)
            distance_matrix[distance_matrix==0] = 999
            df_distance = get_near_player(df_distance, w_df, distance_matrix, name="sameteam")

            distance_matrix = distance_matrix_org.copy()
            distance_matrix *= w_df["team_1"].values.reshape(-1, 1) == w_df["team_1"].values.reshape(1, -1)
            distance_matrix[distance_matrix==1] = 999
            df_distance = get_near_player(df_distance, w_df, distance_matrix, name="notsameteam")

            df_distances.append(df_distance)
        df_distances = pd.concat(df_distances)
        df = pd.merge(df, df_distances, how="left")
        self.logger.info(f"nearest_n_player end: {df.shape}")

        player_features = [
            "x_position_1",
            "y_position_1",
            "x_position_2",
            "y_position_2",
            "distance_1",
            "distance_2",
            "speed_1",
            "speed_2",
            "acceleration_1",
            "acceleration_2",
        ]

        for col in ["orientation", "direction"]:
            for player_id in [1, 2]:
                col_name = f"{col}_{player_id}"
                df[f'{col_name}_sin'] = df[col_name].apply(lambda x: sin(x))
                df[f'{col_name}_cos'] = df[col_name].apply(lambda x: cos(x))
                # player_features.append(f"{col_name}_sin")
                # player_features.append(f"{col_name}_cos")
            for col2 in ["acceleration", "speed"]:
                for col3 in ["sin", "cos"]:
                    for player_id in [1, 2]:
                        col_name = f"{col}_{col2}_{col3}_{player_id}"
                        df[col_name] = df[f"{col2}_{player_id}"] * df[f"{col}_{player_id}_{col3}"]
                        # player_features.append(col_name)

        df["distance"] = np.sqrt(
            (df["x_position_1"].values - df["x_position_2"]) ** 2 + \
            (df["y_position_1"].values - df["y_position_2"]) ** 2
        )

        if self.include_helmet_info:
            for view in ["Endzone", "Sideline"]:
                df[f"{view}_distance_helmet"] = np.sqrt(
                    (df[f"{view}_x_1"].values - df[f"{view}_x_2"]) ** 2 + \
                    (df[f"{view}_y_1"].values - df[f"{view}_y_2"]) ** 2
                )
            df[f"distance_helmet_mean"] = df[["Endzone_distance_helmet", "Sideline_distance_helmet"]].mean(axis=1)
            df[f"distance_helmet_min"] = df[["Endzone_distance_helmet", "Sideline_distance_helmet"]].min(axis=1)
            df[f"distance_helmet_max"] = df[["Endzone_distance_helmet", "Sideline_distance_helmet"]].max(axis=1)

        df["move_sensor"] = df["distance_1"] + df["distance_2"]

        df["is_same_team"] = df["team_1"] == df["team_2"]
        df["is_g"] = df["nfl_player_id_2"] == "G"

        for col in ["orientation", "direction"]:
            for col2 in ["acceleration", "speed"]:
                for col3 in ["sin", "cos"]:
                    col_name = f"{col}_{col2}_{col3}"
                    df[f"{col_name}_diff"] = df[f"{col_name}_1"] - df[f"{col_name}_2"]

        # group by pair
        useful_cols = [
            "distance",
            "move_sensor",
            "distance_top1",
            "distance_top3",
            "distance_top5",
            "distance_top7",
            "diff_from_distance_top1",
            "diff_from_distance_top3",
            "diff_from_distance_top5",
            "diff_from_distance_top7",
        ]
        if self.include_helmet_info:
            useful_cols += [
                "distance_helmet_mean",
                "distance_helmet_min",
                "distance_helmet_max",
            ]

        # G精度向上のために player_1 のlag情報追加
        useful_cols += player_features

        if self.include_helmet_info:
            for view in ["Endzone", "Sideline"]:
                useful_cols.extend([
                    f"{view}_x_1",
                    f"{view}_x_2",
                    f"{view}_y_1",
                    f"{view}_y_2",
                    f"{view}_distance_helmet",
                ])
        self.logger.info(f"groupby features: {df.shape}")

        # TODO: speedup
        for lag in tqdm.tqdm([1, 3, 5, 10, 20, -1, -3, -5, -10, -20]):
            lag_cols = [f"{lag_column}_lag{lag}" for lag_column in useful_cols]
            df[lag_cols] = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])[useful_cols].shift(lag)
            diff_cols = [f"{lag_column}_diff{lag}" for lag_column in useful_cols]
            df[diff_cols] = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])[useful_cols].diff(lag)

        for window in tqdm.tqdm([5, 15]):
            rolling_cols = [f"{lag_column}_window{lag}_mean" for lag_column in useful_cols]
            df[rolling_cols] = df.groupby(
                ["game_play", "nfl_player_id_1", "nfl_player_id_2"]
            )[useful_cols].rolling(window=window, center=True).mean().values
            diff_cols = [f"{col}_diff{lag}" for col in rolling_cols]
            df[diff_cols] = df[useful_cols].values - df[rolling_cols].values

        mean_cols = [f"{col}_step_mean" for col in useful_cols]
        df[mean_cols] = df.groupby(["game_play", "step"])[useful_cols].transform("mean")
        std_cols = [f"{col}_step_std" for col in useful_cols]
        df[std_cols] = df.groupby(["game_play", "step"])[useful_cols].transform("std")
        diff_cols = [f"diff_{col}" for col in mean_cols]
        df[diff_cols] = df[useful_cols].values - df[mean_cols].values

        # 2nd groupby -> memory leakなったら面倒なので後で。。
        self.logger.info(f"groupby features 2nd: {df.shape}")

        df = reduce_mem_usage(df)

        if self.include_helmet_info:
            for view in ["Endzone", "Sideline"]:
                df[f"{view}_move_helmet_1"] = np.sqrt(
                    df[f"{view}_x_1_diff1"] ** 2 + df[f"{view}_y_1_diff1"] ** 2)
                df[f"{view}_move_helmet_2"] = np.sqrt(
                    df[f"{view}_x_2_diff1"] ** 2 + df[f"{view}_y_2_diff1"] ** 2)
                df[f"{view}_move_helmet"] = df[[f"{view}_move_helmet_1", f"{view}_move_helmet_2"]].mean(
                    axis=1)
            df["move_helmet"] = df[["Endzone_move_helmet", "Sideline_move_helmet"]].mean(axis=1)

            lag_columns2 = [
                "move_helmet",
                "Endzone_move_helmet_1", "Endzone_move_helmet_2",
                "Sideline_move_helmet_1", "Sideline_move_helmet_2",
            ]

            for lag in tqdm.tqdm([1, 3, 5, 10, 20, 60, -1, -3, -5, -10, -20, -60]):
                lag_cols = [f"{lag_column}_lag{lag}" for lag_column in lag_columns2]
                df[lag_cols] = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])[lag_columns2].shift(lag)
                diff_cols = [f"{lag_column}_diff{lag}" for lag_column in lag_columns2]
                df[diff_cols] = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])[lag_columns2].diff(lag)

        agg_cols = [
            "distance",
            "move_sensor",
            "distance_lag1", "distance_lag20",
            "move_sensor_lag1", "move_sensor_lag20",
            "distance_lag-1", "distance_lag-20",
            "move_sensor_lag-1", "move_sensor_lag-20",
            "distance_top1",
            "distance_top3",
            "distance_top5",
            "distance_top7",
            "diff_from_distance_top1",
            "diff_from_distance_top3",
            "diff_from_distance_top5",
            "diff_from_distance_top7",
        ]
        if self.include_helmet_info:
            agg_cols += [
                "Endzone_move_helmet_1", "Endzone_move_helmet_2",
                "Sideline_move_helmet_1", "Sideline_move_helmet_2",
                "Endzone_move_helmet_1_lag1", "Endzone_move_helmet_2_lag1",
                "Sideline_move_helmet_1_lag1", "Sideline_move_helmet_2_lag1",
                "Endzone_move_helmet_1_lag5", "Endzone_move_helmet_2_lag5",
                "Sideline_move_helmet_1_lag5", "Sideline_move_helmet_2_lag5",
                "move_helmet",
            ]

        agg_cols += player_features

        agg_cols += [f"{f}_lag1" for f in player_features]
        agg_cols += [f"{f}_lag5" for f in player_features]
        agg_cols += [f"{f}_lag-1" for f in player_features]
        agg_cols += [f"{f}_lag-5" for f in player_features]
        self.logger.info("aggregate features")
        for agg_col in tqdm.tqdm(agg_cols):
            col_name = f"{agg_col}_groupby_gameplay_mean"
            mean = df.groupby("game_play")[agg_col].transform("mean")
            df[f"diff_{col_name}"] = df[agg_col] - mean

        agg_col2 = [
            "distance",
            "move_sensor",
            "distance_top1",
            "distance_top3",
            "distance_top5",
            "distance_top7",
            "diff_from_distance_top1",
            "diff_from_distance_top3",
            "diff_from_distance_top5",
            "diff_from_distance_top7",
        ]
        if self.include_helmet_info:
            agg_col2 += [
                "distance_helmet_mean",
            ]

        for agg_col in tqdm.tqdm(agg_col2):
            col_name = f"{agg_col}_groupby_is_same_team"
            if not inference:
                self.agg_dict["is_same_team"][agg_col] = df.groupby("is_same_team")[agg_col].mean().to_dict()
            mean = df["is_same_team"].map(self.agg_dict["is_same_team"][agg_col])
            df[f"diff_{col_name}"] = df[agg_col] - mean

        agg_col3 = [
            "distance_1",
            "speed_1",
            "acceleration_1",
            "distance_1_lag1",
            "distance_1_lag5",
            "distance_1_lag10",
            "distance_1_lag-1",
            "distance_1_lag-5",
            "distance_1_lag-10",
            "speed_1_lag1",
            "speed_1_lag5",
            "speed_1_lag10",
            "speed_1_lag-1",
            "speed_1_lag-5",
            "speed_1_lag-10",
            "distance_top1",
            "distance_top3",
            "distance_top5",
            "distance_top7",
            "diff_from_distance_top1",
            "diff_from_distance_top3",
            "diff_from_distance_top5",
            "diff_from_distance_top7",
        ]
        if self.include_helmet_info:
            agg_col3 += [
                "move_helmet",
                "move_helmet_lag1",
                "move_helmet_lag5",
                "move_helmet_lag-1",
                "move_helmet_lag-5",
                "Endzone_move_helmet_1",
                "Sideline_move_helmet_1",
                "Endzone_move_helmet_1_lag1",
                "Sideline_move_helmet_1_lag1",
                "Endzone_move_helmet_1_lag5",
                "Sideline_move_helmet_1_lag5",
            ]
        for groupby_col in ["is_g", "n_player_distance_all_in_3",
                            "n_player_distance_sameteam_in_3", "n_player_distance_notsameteam_in_3",]:
            for agg_col in tqdm.tqdm(agg_col3):
                col_name = f"{agg_col}_groupby_{groupby_col}"
                if not inference:
                    if groupby_col not in self.agg_dict:
                        self.agg_dict[groupby_col] = {}
                    self.agg_dict[groupby_col][agg_col] = df.groupby(groupby_col)[agg_col].mean().to_dict()
                mean = df[groupby_col].map(self.agg_dict[groupby_col][agg_col])
                df[f"diff_{col_name}"] = df[agg_col] - mean

        self.logger.info("Reduce memory usage")
        df = reduce_mem_usage(df)

        self.logger.info(f"feature engineering end! {df.shape}")
        df = df[df["contact"].notnull()].reset_index(drop=True)
        self.logger.info(f"drop contact=null {df.shape}")
        if not inference:
            self.logger.info("save feather")
            os.makedirs(os.path.dirname(feature_path), exist_ok=True)
            df.to_feather(feature_path)
            with open(json_path, "w") as  f:
                json.dump(self.agg_dict, f)

        return df

    def train(self,
              df: pd.DataFrame,
              df_label: pd.DataFrame = None,
              negative_ratio: float = 1,
              use_half_data: bool = False):

        gkfold = GroupKFold(5)
        df_fe = self.feature_engineering(df, inference=False)
        if df_label is None:
            if self.debug:
                df_label = df
            else:
                df_label = pd.read_csv("../../input/nfl-player-contact-detection/train_labels.csv")
        df_label["game_key"] = [int(x.split("_")[0]) for x in df_label["contact_id"].values]

        self.logger.info((df_fe.isnull().sum() / len(df_fe)).sort_values())

        for train_idx, val_idx in gkfold.split(df_label, groups=df_label["game_key"].values):
            df_label_train = df_label.iloc[train_idx]
            df_label_val = df_label.iloc[val_idx]
            df_train = df_fe[df_fe["game_key"].isin(df_label_train["game_key"].values)]
            df_val = df_fe[df_fe["game_key"].isin(df_label_val["game_key"].values)]
            df_test = df[df["game_key"].isin(df_label_val["game_key"].values)]
            break
        del df_fe; gc.collect()

        if negative_ratio < 1:
            pos = df_train[df_train["contact"] == 0]
            pos = pos.sample(int(pos * negative_ratio))
            df_train = pd.concat([
                pos,
                df_train[df_train["contact"] == 1],
            ])
            del pos; gc.collect()

        df_merge = pd.merge(
            df_label_val[["contact_id", "contact"]],
            df_val[["contact_id", "contact"]].rename(columns={"contact": "pred"}),
            how="left"
        ).fillna(0).sort_values("contact", ascending=False).drop_duplicates("contact_id")
        possible_score_all = matthews_corrcoef(df_merge["contact"].values, df_merge["pred"].values == 1)
        self.logger.info(f"possible MCC score: {possible_score_all}")

        if self.use_features is None:
            self.use_features = df_train.drop(self.drop_columns + ["contact"], axis=1).columns
        if use_half_data:
            df_train = df_train.iloc[::2].reset_index(drop=True)
        dataset_train = lgb.Dataset(df_train[self.use_features], label=df_train["contact"])
        dataset_val = lgb.Dataset(df_val[self.use_features], label=df_val["contact"])
        del df_train; gc.collect()

        lgb.register_logger(self.logger)
        mlflow.set_tracking_uri('../../mlruns/')

        with mlflow.start_run(experiment_id=1, run_name=self.exp_name):
            self.model = lgb.train(
                copy.copy(self.params),
                dataset_train,
                valid_sets=[dataset_train, dataset_val],
                verbose_eval=100,
            )

            self.model.save_model(self.model_dir)

            # inference
            self.model = lgb.Booster(model_file=self.model_dir)
            if self.fast_mode:
                pred = self.model.predict(df_val[self.model.feature_name()])
                contact_id = df_val["contact_id"].values
            else:
                pred, contact_id = self.predict(df_test)

            df_pred = pd.DataFrame({
                "contact_id": contact_id,
                "pred": pred
            })

            df_merge = pd.merge(
                df_label_val, df_pred, how="left"
            )
            df_merge["pred"] = df_merge["pred"].fillna(0)

            pred = df_merge["pred"].values
            contact = df_merge["contact"].values
            auc = roc_auc_score(contact, pred)
            self.logger.info(f"auc: {auc}")
            self.logger.info("------- MCC -------")

            best_th = None
            best_score = -1
            for th in np.arange(0, 1, 0.05):
                score = matthews_corrcoef(contact, pred > th)
                self.logger.info(f"th={th}: {score}")
                if score > best_score:
                    best_th = th
                    best_score = score

            for k, v in self.params.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("best_th", best_th)
            mlflow.log_metric("best_score", best_score)

            pd.DataFrame({
                "col": self.model.feature_name(),
                "imp": self.model.feature_importance("gain") / self.model.feature_importance("gain").sum()
            }).sort_values("imp", ascending=False).to_csv(f"{self.output_dir}/feature_importance.csv", index=False)
            pd.DataFrame({
                "contact_id": df_merge["contact_id"].values,
                "score": pred
            }).to_csv(f"{self.output_dir}/pred.csv", index=False)

    def predict(self,
                df: pd.DataFrame):
        df = self.feature_engineering(df)
        return self.model.predict(df[self.model.feature_name()]), df["contact_id"].values


def main():
    output_dir = f"../../output/lgbm/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(__file__, output_dir)
    logger = get_logger(output_dir)
    df = pd.read_feather("../../output/preprocess/master_data_v4/gps.feather")
    if debug:
        df = df.head(300000)

    params = {
        'objective': 'binary',
        'metrics': 'auc',
        'num_leaves': 128,
        'max_depth': -1,
        'bagging_fraction': 0.9,  # 0.5,
        'feature_fraction': 0.3,
        'bagging_seed': 0,
        'reg_alpha': 10,
        'reg_lambda': 5,
        'min_data_in_leaf': 10000,
        'random_state': 0,
        'verbosity': -1,
        "n_estimators": 20000,
        "early_stopping_rounds": 100,
        "learning_rate": 0.01,
        "n_jobs": 28
    }

    model = LGBMModel(output_dir=output_dir, logger=logger, exp_name="exp006_bugfix", debug=debug, fast_mode=True,
                      params=params, include_helmet_info=False)
    model.train(df, use_half_data=True)
    del model.logger

    with open(f"{output_dir}/model.pickle", "wb") as f:
        pickle.dump(model, f)

    # for _ in range(1000):
    #     output_dir = f"../../output/lgbm/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}"
    #     os.makedirs(output_dir, exist_ok=True)
    #     shutil.copy(__file__, output_dir)
    #     logger = get_logger(output_dir)
    #
    #     df = pd.read_feather("../../output/preprocess/master_data_v3/gps.feather")
    #     if debug:
    #         df = df.head(300000)
    #
    #     params = {
    #         'objective': 'binary',
    #         'metrics': 'auc',
    #         'num_leaves': np.random.choice([16, 32, 64, 128, 256]),
    #         'max_depth': -1,
    #         'bagging_fraction': np.random.choice([0.7, 0.9]),  # 0.5,
    #         'feature_fraction': np.random.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
    #         'bagging_seed': 0,
    #         'reg_alpha': np.random.choice([0, 0.1, 0.5, 1, 3, 5, 10]),
    #         'reg_lambda': np.random.choice([0, 0.1, 0.5, 1, 3, 5, 10]),
    #         'min_data_in_leaf': np.random.choice([1, 10, 50, 100, 500, 1000, 5000, 10000]),
    #         'random_state': 0,
    #         'verbosity': -1,
    #         "n_estimators": 20000,
    #         "early_stopping_rounds": 100,
    #         "learning_rate": 0.1,
    #         "n_jobs": 28
    #     }
    #     model = LGBMModel(output_dir=output_dir, logger=logger, exp_name="exp006_bugfix", debug=debug, fast_mode=True,
    #                       params=params, use_features=use_features)
    #     model.train(df)

if __name__ == "__main__":
    main()
