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

pd.set_option("max_row", 1000)
pd.set_option("max_column", 200)

debug = False

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
        feature_path = f"{self.feature_dir}/{os.path.basename(__file__).replace('.py', '')}/feature_len{len(df)}.feather"
        if os.path.isfile(feature_path) and not inference and not self.debug:
            self.logger.info("load from feature_dir")
            return pd.read_feather(feature_path)
        self.logger.info("Reduce memory usage")
        df = reduce_mem_usage(df)

        self.logger.info("FE1: all features")

        self.logger.info(f"[aggregate view]before: {len(df)}")
        helmet_columns = [
            "left_1", "width_1", "top_1", "height_1", "x_1", "y_1",
            "left_2", "width_2", "top_2", "height_2", "x_2", "y_2",
        ]
        df["contact_id"] = df["game_play"] + "_" + df["step"].astype(str) + "_" + df["nfl_player_id_1"].astype(
            str) + "_" + df["nfl_player_id_2"].astype(str)
        df_endzone = df[df["view"] == "Endzone"].drop("view", axis=1)
        df_sideline = df[df["view"] == "Sideline"].drop("view", axis=1)
        df_endzone.columns = [f"Endzone_{col}" if col in helmet_columns else col for col in df_endzone.columns]
        df_sideline.columns = [f"Sideline_{col}" if col in helmet_columns else col for col in df_sideline.columns]
        df = df[["contact_id"]].drop_duplicates()
        df = pd.merge(df, df_endzone, how="left")
        df = pd.merge(df, df_sideline, how="left")
        df = df.sort_values(["game_play", "step", "nfl_player_id_1", "nfl_player_id_2"])
        self.logger.info(f"[aggregate view]after: {len(df)}")

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
                player_features.append(f"{col_name}_sin")
                player_features.append(f"{col_name}_cos")
            for col2 in ["acceleration", "speed"]:
                for col3 in ["sin", "cos"]:
                    for player_id in [1, 2]:
                        col_name = f"{col}_{col2}_{col3}_{player_id}"
                        df[col_name] = df[f"{col2}_{player_id}"] * df[f"{col}_{player_id}_{col3}"]
                        player_features.append(col_name)

        df["distance"] = np.sqrt(
            (df["x_position_1"].values - df["x_position_2"]) ** 2 + \
            (df["y_position_1"].values - df["y_position_2"]) ** 2
        )

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
        df_rets = []
        lag_columns = [
            "distance",
            "move_sensor",
            "distance_helmet_mean",
            "distance_helmet_min",
            "distance_helmet_max",
            "orientation_acceleration_sin_diff",
            "orientation_acceleration_cos_diff",
            "orientation_speed_sin_diff",
            "orientation_speed_cos_diff",
            "direction_acceleration_sin_diff",
            "direction_acceleration_cos_diff",
            "direction_speed_sin_diff",
            "direction_speed_cos_diff",
        ]

        # G精度向上のために player_1 のlag情報追加
        lag_columns += player_features

        for view in ["Endzone", "Sideline"]:
            lag_columns.extend([
                f"{view}_x_1",
                f"{view}_x_2",
                f"{view}_y_1",
                f"{view}_y_2",
                f"{view}_distance_helmet",
            ])
        self.logger.info("groupby features")

        # TODO: speedup
        for lag in tqdm.tqdm([1, 5, 10, 20, -1, -5, -10, -20]):
            cols = [f"{lag_column}_lag{lag}" for lag_column in lag_columns]
            df[cols] = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])[lag_columns].diff(lag)

        # 2nd groupby -> memory leakなったら面倒なので後で。。
        self.logger.info("groupby features 2nd")
        lag_columns2 = [
            "move_helmet",
            "Endzone_move_helmet_1", "Endzone_move_helmet_2",
            "Sideline_move_helmet_1", "Sideline_move_helmet_2",
        ]
        df_rets2 = []
        df = reduce_mem_usage(df)

        for _, w_df in tqdm.tqdm(df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])):
            # lag
            for view in ["Endzone", "Sideline"]:
                w_df[f"{view}_move_helmet_1"] = np.sqrt(
                    w_df[f"{view}_x_1_lag1"].diff() ** 2 + w_df[f"{view}_y_1_lag1"].diff() ** 2)
                w_df[f"{view}_move_helmet_2"] = np.sqrt(
                    w_df[f"{view}_x_2_lag1"].diff() ** 2 + w_df[f"{view}_y_2_lag1"].diff() ** 2)
                w_df[f"{view}_move_helmet"] = w_df[[f"{view}_move_helmet_1", f"{view}_move_helmet_2"]].mean(
                    axis=1)
            w_df["move_helmet"] = w_df[["Endzone_move_helmet", "Sideline_move_helmet"]].mean(axis=1)

            df_rets2.append(w_df)
        df_rets2 = pd.concat(df_rets2).sort_index().reset_index(drop=True)
        del df; gc.collect()

        for lag in [1, 5, 10, 20, 60]:
            cols = [f"{lag_column}_lag{lag}" for lag_column in lag_columns2]
            df_rets2[cols] = df_rets2.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])[lag_columns2].diff(lag)

        agg_cols = [
            "distance",
            "Endzone_move_helmet_1", "Endzone_move_helmet_2",
            "Sideline_move_helmet_1", "Sideline_move_helmet_2",
            "Endzone_move_helmet_1_lag1", "Endzone_move_helmet_2_lag1",
            "Sideline_move_helmet_1_lag1", "Sideline_move_helmet_2_lag1",
            "Endzone_move_helmet_1_lag5", "Endzone_move_helmet_2_lag5",
            "Sideline_move_helmet_1_lag5", "Sideline_move_helmet_2_lag5",
            "move_helmet", "move_sensor",
            "distance_lag1", "distance_lag20",
            "move_sensor_lag1", "move_sensor_lag20",
        ]

        agg_cols += player_features

        agg_cols += [f"{f}_lag1" for f in player_features]
        agg_cols += [f"{f}_lag5" for f in player_features]
        self.logger.info("aggregate features")
        for agg_col in tqdm.tqdm(agg_cols):
            col_name = f"{agg_col}_groupby_gameplay_mean"
            mean = df_rets2.groupby("game_play")[agg_col].transform("mean")
            df_rets2[f"diff_{col_name}"] = df_rets2[agg_col] - mean

        agg_col2 = [
            "distance",
            "move_sensor",
            "distance_helmet_mean",
            "distance_helmet_min",
            "distance_helmet_max",
        ]

        for agg_col in tqdm.tqdm(agg_col2):
            col_name = f"{agg_col}_groupby_is_same_team"
            if not inference:
                self.agg_dict["is_same_team"][agg_col] = df_rets2.groupby("is_same_team")[agg_col].mean().to_dict()
            mean = df_rets2["is_same_team"].map(self.agg_dict["is_same_team"][agg_col])
            df_rets2[f"diff_{col_name}"] = df_rets2[agg_col] - mean

        agg_col3 = [
            "distance_1",
            "speed_1",
            "acceleration_1",
            "Endzone_move_helmet_1",
            "Sideline_move_helmet_1",
            "Endzone_move_helmet_1_lag1",
            "Sideline_move_helmet_1_lag1",
            "Endzone_move_helmet_1_lag5",
            "Sideline_move_helmet_1_lag5",
            "distance_1_lag1",
            "distance_1_lag5",
            "distance_1_lag10",
            "speed_1_lag1",
            "speed_1_lag5",
            "speed_1_lag10",
            "acceleration_1_lag1",
            "acceleration_1_lag5",
            "acceleration_1_lag10",
        ]
        for agg_col in tqdm.tqdm(agg_col3):
            col_name = f"{agg_col}_groupby_is_g"
            if not inference:
                self.agg_dict["is_g"][agg_col] = df_rets2.groupby("is_g")[agg_col].mean().to_dict()
            mean = df_rets2["is_g"].map(self.agg_dict["is_g"][agg_col])
            df_rets2[f"diff_{col_name}"] = df_rets2[agg_col] - mean

        self.logger.info("Reduce memory usage")
        df_rets2 = reduce_mem_usage(df_rets2)

        self.logger.info(f"feature engineering end! {df_rets2.shape}")
        df_rets2 = df_rets2[df_rets2["contact"].notnull()].reset_index(drop=True)
        self.logger.info(f"drop contact=null {df_rets2.shape}")
        if not inference:
            self.logger.info("save feather")
            os.makedirs(os.path.dirname(feature_path), exist_ok=True)
            df_rets2.to_feather(feature_path)

        return df_rets2

    def train(self,
              df: pd.DataFrame,
              df_label: pd.DataFrame = None):

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

        df_merge = pd.merge(
            df_label_val[["contact_id", "contact"]],
            df_val[["contact_id", "contact"]].rename(columns={"contact": "pred"}),
            how="left"
        ).fillna(0).sort_values("contact", ascending=False).drop_duplicates("contact_id")
        possible_score_all = matthews_corrcoef(df_merge["contact"].values, df_merge["pred"].values == 1)
        self.logger.info(f"possible MCC score: {possible_score_all}")

        if self.use_features is None:
            self.use_features = df_train.drop(self.drop_columns + ["contact"], axis=1).columns
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
        "learning_rate": 0.1,
        "n_jobs": 8
    }
    use_features = pd.read_csv("../../output/lgbm/exp010/20230115181755/feature_importance.csv")["col"].values[:400]

    model = LGBMModel(output_dir=output_dir, logger=logger, exp_name="exp006_bugfix", debug=debug, fast_mode=True,
                      params=params)
    model.train(df)
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
    #         "n_jobs": 8
    #     }
    #     model = LGBMModel(output_dir=output_dir, logger=logger, exp_name="exp006_bugfix", debug=debug, fast_mode=True,
    #                       params=params, use_features=use_features)
    #     model.train(df)

if __name__ == "__main__":
    main()
