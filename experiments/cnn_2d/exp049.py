import torch
from torch import nn
import timm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime as dt
from logging import Logger, StreamHandler, Formatter, FileHandler
from sklearn.model_selection import GroupKFold
import logging
import dataclasses
import tqdm
import numpy as np
from typing import List
from transformers import get_linear_schedule_with_warmup
import cv2
from torchvision.io.video import read_video
try:
    from torchvision.models.video import r3d_18, R3D_18_Weights, mc3_18, MC3_18_Weights, mvit_v2_s, MViT_V2_S_Weights, r2plus1d_18, R2Plus1D_18_Weights
    import mlflow
    import wandb
except Exception as e:
    print(e)
    from torchvision.models.video import r3d_18
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
import shutil
import torch.nn.functional as F
import pickle
from torch.optim.lr_scheduler import StepLR, LambdaLR
from typing import Tuple
import albumentations as A
import random
import glob
import gc
import dgl
from dgl.dataloading import GraphDataLoader
import copy
from dgl.nn import EGATConv

debug = False
torch.backends.cudnn.benchmark = True

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


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

@dataclasses.dataclass
class ConfigForGNN:
    exp_name: str
    debug: bool = debug
    epochs: int = 1
    if debug:
        epochs: int = 1
    lr: float = 1e-3
    lr_fc: float = 1e-3
    weight_decay: float = 0.001
    model_name: str = "egat"
    scheduler: str = "StepLRWithWarmUp"
    step_size_ratio: float = 2
    gamma: float = 0.1
    warmup_ratio: float = 0.01
    criterion: str = "smoothfocalloss"
    gradient_clipping: float = 10

    base_dir: str = "../../output/preprocess/images"
    data_dir: str = f"../../output/preprocess/master_data_v3"

    n_predict_frames: int = 1
    node_feats: int = 64
    edge_feats: int = 64
    batch_size: int = 32

    data_per_epoch: float = 1
    calc_single_view_loss: bool = False


@dataclasses.dataclass
class ConfigForTransformer:
    exp_name: str
    debug: bool = debug

    epochs: int = 5
    if debug:
        epochs: int = 1

    lr: float = 1e-3
    lr_fc: float = 1e-3
    weight_decay: float = 0.01
    n_frames: int = 31
    n_predict_frames: int = 1

    if n_frames % 2 == 0 or n_predict_frames % 2 == 0:
        raise ValueError
    data_dir: str = f"../../output/preprocess/master_data_v3"
    base_dir: str = "../../output/preprocess/images"
    image_path: str = "a"
    model_name: str = "transformer"
    gradient_clipping: float = 10
    data_per_epoch: float = 1
    batch_size: int = 128
    use_data_step: int = 1
    criterion: str = "smoothfocalloss"
    scheduler: str = "linear"
    step_size_ratio: float = 2
    gamma: float = 0.1
    warmup_ratio: float = 0.1
    calc_single_view_loss: bool = False  # 固定
    fc_sideend: str = "concat"

    # transformer
    hidden_size_transformer: int = 128  # TODO: exp002に合わせて
    num_layer_transformer: int = 2
    nhead: int = 8
    num_layer_rnn: int = 1
    feature_dir: str = "../../output/preprocess/feature/exp026/feature_len13876756.feather"
    max_length: int = 256
    apply_log1p: bool = True
    apply_norm: bool = False
    apply_ffn: bool = True
    apply_lstm: bool = True

    try:
        use_features: List[str] = pd.read_csv(
            "../../output/lgbm/exp026/20230216223836/feature_importance.csv"
        )["col"].values[:100]
    except:
        use_features: List[str] = ""

    submission_mode: bool = False
    transformer: str = "only_encoder"
    dropout_transformer: float = 0.1
    residual: bool = False

    fold: int = 0
    gk_key: str = "game_key"
    smooth: float = 0.1
    g_emb: bool = False


@dataclasses.dataclass
class Config:
    exp_name: str
    debug: bool = debug
    epochs: int = 1
    if debug:
        epochs: int = 1

    lr: float = 1e-4
    lr_fc: float = 1e-3
    weight_decay: float = 0.1
    n_frames: int = 31
    n_predict_frames: int = 1

    if n_frames % 2 == 0 or n_predict_frames % 2 == 0:
        raise ValueError
    step: int = 3
    extention: str = ".jpg"
    negative_sample_ratio_close: float = 0.2
    negative_sample_ratio_far: float = 0.2
    negative_sample_ratio_g: float = 0.2
    base_dir: str = "../../output/preprocess/images"
    image_path: str = "images_128x96_v21"
    data_dir: str = f"../../output/preprocess/master_data_v3"
    img_shape: Tuple[int, int] = (96, 128)
    gradient_clipping: float = 0.2
    exist_image_threshold: float = 0.1
    data_per_epoch: float = 1
    grayscale: bool = False
    batch_size: int = 32
    use_data_step: int = 1

    model_name: str = "cnn_3d_r3d_18"
    seq_model: str = "flatten"
    dropout_seq: float = 0.2
    activation: nn.Module = nn.ReLU

    kernel_size_conv1d: int = 13
    stride_conv1d: int = 1
    hidden_size_1d: int = 300

    submission_mode: bool = False

    distance_threshold: float = 1.75
    calc_single_view_loss: bool = True
    calc_single_view_loss_weight: float = 1
    exist_center_image_threshold: float = 0

    criterion: str = "smoothfocalloss"
    fc: str = "simple"
    fc_dropout: float = 0

    scheduler: str = "StepLRWithWarmUp"
    step_size_ratio: float = 0.3
    gamma: float = 0.1
    warmup_ratio: float = 0.01

    pretrained: bool = True
    fc_sideend: str = "concat"
    hidden_size_3d: int = 128
    kernel_size_3d: Tuple[int, int, int] = (7, 5, 5)
    stride_3d: Tuple[int, int, int] = (2, 1, 1)
    dilation_3d: Tuple[int, int, int] = (1, 1, 1)

    dropout_3d: float = 0.2

    transforms_train: A.Compose = A.Compose([
        A.HorizontalFlip(p=0.5),
    ])
    transforms_eval: A.Compose = A.Compose([
    ])

    p_drop_frame: float = 0
    frame_adjust: int = 0
    interpolate_image: bool = True

    g_embedding: bool = False
    custom_3d: bool = False
    kernel_custom_3d: Tuple[int, int, int] = (2, 3, 3)

    feature_window: int = 0
    try:
        feature_cols: List[str] = pd.read_csv(
            "../../output/lgbm/exp026/20230216223836/feature_importance.csv"
        )["col"].values[:100]
    except:
        feature_cols: List[str] = ""
    feature_hidden_size: int = 128
    feature_mean_dim: str = "window"
    nhead: int = 8
    num_layer_transformer: int = 2
    feature_dir: str = "../../output/preprocess/feature/exp026/feature_len13876756.feather"

    save_feature: bool = False
    interpolate_outside: bool = True
    image_feature_dir: str = ""

    pretrained_feature_dim: int = 512
    channel_3d: int = 3
    seq_model_short: bool = False
    only_g: bool = False

    stack_dilations: Tuple[int] = (1, 2, 3, 4, 5)
    pooling: str = "avgpool"

    fold: int = 0
    gk_key: str = "game_key"
    soft_label_range: tuple = None
    channel_6: bool = False

    pad_image_values: int = 0
    smooth: float = 0.1
    focal_gamma: float = 2.0
    max_frames: int = 300
    predict_all_frames: bool = False

class FocalLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1, gamma=2):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1. - pt)**self.gamma * bce_loss
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class SmoothFocalLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1, gamma=2, smoothing=0.0):
        super().__init__()
        self.reduction = reduction
        self.focal_loss = FocalLoss(reduction='mean', alpha=alpha, gamma=gamma)
        self.smoothing = smoothing

    @staticmethod
    def _smooth(targets:torch.Tensor, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothFocalLoss._smooth(targets, self.smoothing)
        loss = self.focal_loss(inputs, targets)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class MCCLoss(nn.Module):
    """
    Calculates the proposed Matthews Correlation Coefficient-based loss.
    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    """

    def __init__(self):
        super(MCCLoss, self).__init__()

    def forward(self, inputs, targets):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.
        """
        inputs = torch.sigmoid(inputs)
        targets = torch.sigmoid(targets)
        tp = torch.sum(torch.mul(inputs, targets))
        tn = torch.sum(torch.mul((1 - inputs), (1 - targets)))
        fp = torch.sum(torch.mul(inputs, (1 - targets)))
        fn = torch.sum(torch.mul((1 - inputs), targets))

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, 1, fp)
            * torch.add(tp, 1, fn)
            * torch.add(tn, 1, fp)
            * torch.add(tn, 1, fn)
        )

        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = torch.div(numerator.sum(), denominator.sum() + 1.0)
        return 1 - mcc


class NFLDatasetForFeatureExtraction(Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        filelist_path = f"{self.base_dir}/filelist.pickle"
        with open(filelist_path, "rb") as f:
            self.files = list(pickle.load(f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        img = cv2.imread(file).transpose(2, 0, 1)

        return torch.Tensor(img), [file]


class NFLTransformerDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 base_dir: str,
                 logger: Logger,
                 config: ConfigForTransformer,
                 test: bool):
        self.base_dir = base_dir
        self.config = config
        self.test = test
        self.exist_files = set()
        self._get_item_information(df, logger)

    def _get_base_dir(self,
                      game_play: str,
                      view: str,
                      id_1: str,
                      id_2: str):
        return f"{self.base_dir}/{game_play}/{view}/{id_1}_{id_2}"

    def _get_key(self,
                 game_play,
                 view,
                 id_1,
                 id_2,
                 frame):
        if self.image_dict is not None:
            # for submission
            return f"{game_play}_{view}_{id_1}_{id_2}_{frame}"
        else:
            # for local training
            base_dir = self._get_base_dir(game_play, view, id_1, id_2)
            return f"{base_dir}_{frame}{self.config.extention}"

    def _get_item_information(self, df: pd.DataFrame, logger: Logger):
        self.items = []
        logger.info("_get_item_information start")

        failed_count = 0
        np.random.seed(0)

        label_sum = 0

        df = df[df["contact"].notnull()]
        if self.config.apply_log1p:
            df[self.config.use_features] = np.log1p(df[self.config.use_features]).fillna(0).replace(np.inf, 0).replace(-np.inf, 0)
        else:
            df[self.config.use_features] = df[self.config.use_features].fillna(0).replace(np.inf, 0).replace(-np.inf, 0)

        for key, w_df in tqdm.tqdm(df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])):
            id_2 = key[2]

            is_g = int(id_2 == "G")
            contacts_ = w_df["contact"].values

            contact_ids = [""] * self.config.max_length
            contact_ids[:len(w_df)] = w_df["contact_id"].values.tolist()
            # w_df = w_df.drop(drop_columns, axis=1).fillna(0).replace(np.inf, 0).replace(-np.inf, 0)
            # w_df = w_df[use_features].fillna(0).replace(np.inf, 0).replace(-np.inf, 0)
            feature_ary = np.zeros((self.config.max_length, len(self.config.use_features)))
            feature_ary[:len(w_df), :] = w_df[self.config.use_features].values

            mask = np.ones(self.config.max_length)
            mask[:len(contacts_)] = 0
            mask = mask == 1

            contacts = np.zeros(self.config.max_length)
            contacts[:len(contacts_)] = contacts_
            label_sum += contacts.sum()

            self.items.append({
                "contact_id": contact_ids,
                "contact": contacts,
                "feature": feature_ary,
                "is_g": is_g,
                "mask": mask
            })
            if self.config.debug and len(self.items) > 300:
                break

        logger.info(f"finished. extracted={len(self.items)} (total_df={len(df)}, label_sum={label_sum})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        contact_id = item["contact_id"]
        feature = item["feature"]
        mask = item["mask"]
        labels = item["contact"]
        is_g = item["is_g"]

        return contact_id, torch.Tensor(feature), torch.Tensor(labels), torch.Tensor(mask), torch.LongTensor([is_g])


class NFLGraphDataset(dgl.data.DGLDataset):
    def __init__(self, df: pd.DataFrame, config: Config):
        self.df = df
        self.config = config
        self.max_len = 22*22
        super().__init__(name="nfl")

    def _load_graph(self, df):
        graphs = []
        contact_ids = []
        df["is_same_team"] = (df["team_1"] == df["team_2"]).astype(int)
        for k, w_df in tqdm.tqdm(
            df.drop_duplicates(["game_play", "step", "nfl_player_id_1", "nfl_player_id_2"]).groupby(["game_play", "step"])
        ):
            w_df = w_df.sort_values(["nfl_player_id_1", "nfl_player_id_2"])
            w_df["nfl_player_id_2"] = [ary[0] if ary[1] == "G" else ary[1] for ary in
                                       w_df[["nfl_player_id_1", "nfl_player_id_2"]].values]
            id_dict = {p_id: i for i, p_id in
                       enumerate(np.unique(w_df[["nfl_player_id_1", "nfl_player_id_2"]].values.flatten()))}
            g = dgl.graph([
                (id_dict[x[0]], id_dict[x[1]]) for x in w_df[["nfl_player_id_1", "nfl_player_id_2"]].values
            ])
            g.ndata["feature"] = torch.Tensor(w_df.drop_duplicates(["game_play", "step", "nfl_player_id_1"])[
                                                  ["speed_1", "distance_1", "acceleration_1"]].values)
            g.edata["distance"] = torch.Tensor(w_df["distance"].fillna(0).values.reshape(-1, 1))
            g.edata["is_same_team"] = torch.LongTensor(w_df["is_same_team"].fillna(0).values.reshape(-1, 1))
            g.edata["label"] = torch.Tensor(w_df["contact"].values)
            graphs.append(g)
            contact_ids.append(w_df["contact_id"].values)
            if len(graphs) > 100 and config.debug:
                break
        return graphs, contact_ids

    def process(self):
        self.graphs, self.contact_ids = self._load_graph(self.df)
        del self.df

    def __getitem__(self, idx):
        dummy = torch.Tensor([0])
        pad = [""]
        contact_ids = self.contact_ids[idx].tolist()
        if len(contact_ids) < self.max_len:
            contact_ids += pad * (self.max_len - len(contact_ids))

        return contact_ids, self.graphs[idx], dummy, dummy, dummy

    def __len__(self):
        return len(self.graphs)


class NFLDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 base_dir: str,
                 logger: Logger,
                 config: Config,
                 test: bool,
                 use_filelist: bool = True,
                 submission_mode: bool = True,
                 image_dict: dict = None):
        self.base_dir = base_dir
        self.config = config
        self.test = test
        self.exist_files = set()
        self.image_dict = image_dict
        self.submission_mode = submission_mode

        filelist_path = f"{self.base_dir}/filelist.pickle"
        if use_filelist:
            if os.path.isfile(filelist_path):
                logger.info("load filelist...")
                with open(filelist_path, "rb") as f:
                    filelist = pickle.load(f)
            else:
                logger.info("make filelist...")
                filelist = glob.glob(f"{self.base_dir}/*/*/*{self.config.extention}")
                filelist = [f.replace("\\", "/") for f in filelist]
                filelist = set(filelist)
                logger.info("save filelist...")
                with open(filelist_path, "wb") as f:
                    pickle.dump(filelist, f)
        else:
            filelist = None

        if not self.submission_mode:
            self.filelist = filelist
        if self.config.channel_6:
            C = 6
        else:
            C = 3
        if self.config.model_name == "cnn_2d3d":
            logger.info("load features...")
            self.img_shape = np.load(list(filelist)[0]).shape
        elif "2.5d" not in self.config.model_name:
            self.img_shape = (self.config.img_shape[0], self.config.img_shape[1], C)
        else:
            self.img_shape = (self.config.img_shape[0], self.config.img_shape[1], 1)
        self._get_item_information(df, logger, filelist)

    def _get_base_dir(self,
                      game_play: str,
                      view: str,
                      id_1: str,
                      id_2: str):
        return f"{self.base_dir}/{game_play}/{view}/{id_1}_{id_2}"

    def _get_key(self,
                 game_play,
                 view,
                 id_1,
                 id_2,
                 frame,
                 prefix=""):
        if self.image_dict is not None:
            return f"{game_play}_{view}_{id_1}_{id_2}_{frame}"
        else:
            base_dir = self._get_base_dir(game_play, view, id_1, id_2)
            if self.config.channel_6:
                return f"{base_dir}_{frame}{prefix}{self.config.extention}"
            else:
                return f"{base_dir}_{frame}{self.config.extention}"

    def _exist_file(self,
                    game_play,
                    view,
                    id_1,
                    id_2,
                    frame,
                    filelist):
        if self.config.channel_6:
            prefix = "_original"
        else:
            prefix = ""
        if self.image_dict is not None:
            # for pretrained features
            return self._get_key(game_play, view, id_1, id_2, frame, prefix) in self.image_dict
        else:
            # for local training
            return self._get_key(game_play, view, id_1, id_2, frame, prefix) in filelist

    def _exist_files(self,
                     game_play: str,
                     id_1: str,
                     id_2: str,
                     frames: List[int],
                     filelist: set,
                     n_frames: int,
                     exist_image_threshold: float):
        count = 0

        for view in ["Sideline", "Endzone"]:
            base_dir = self._get_base_dir(game_play, view, id_1, id_2)
            for frame in frames:
                if self._exist_file(game_play, view, id_1, id_2, frame, filelist):
                    count += 1
                if count >= n_frames * 2 * exist_image_threshold:
                    return True
        return False

    def _get_item_information(self, df: pd.DataFrame, logger: Logger, filelist: set):
        self.items = []
        logger.info("_get_item_information start")

        failed_count = 0
        is_g_count = 0
        np.random.seed(0)

        contacts_all = []
        df = df[df["contact"].notnull()]

        if self.config.only_g:
            df = df[df["nfl_player_id_2"] == "G"]
        if self.config.feature_window > 0:
            cols_log = [f"{col}_log" for col in self.config.feature_cols]
            df[cols_log] = np.log1p(df[self.config.feature_cols].fillna(0)).replace(np.inf, 0).replace(-np.inf, 0).fillna(0)
        for key, w_df in tqdm.tqdm(
            df.drop_duplicates(
                ["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
            ).groupby(
                ["game_play", "nfl_player_id_1", "nfl_player_id_2"]
            )
        ):
            game_play = key[0]
            id_1 = key[1]
            id_2 = key[2]
            w_df = w_df.reset_index(drop=True)

            contact_ids = w_df["contact_id"].values
            frames = w_df["frame"].values
            contacts = w_df["contact"].values
            distances = w_df["distance"].fillna(0).values

            if self.config.feature_window > 0:
                features = w_df[cols_log].values

            is_g = int(id_2 == "G")
            if not self.config.predict_all_frames:
                for i in range(len(w_df)):

                    if not self.test and i % self.config.use_data_step != 0:
                        continue

                    min_frame_idx = frames[i] - self.config.n_frames // 2 * self.config.step
                    max_frame_idx = frames[i] + self.config.n_frames // 2 * self.config.step + 1  # frames数は偶数にする(conv1dメンドイので)

                    frame_indice = np.arange(min_frame_idx, max_frame_idx, self.config.step)

                    if self.config.frame_adjust > 0 and not self.test:
                        frame_index_adjust = int(np.random.choice(np.arange(-self.config.frame_adjust, self.config.frame_adjust+1)))
                        frame_indice += frame_index_adjust
                    window = self.config.n_predict_frames // 2

                    if i - window < 0 or i + window + 1 > len(w_df):
                        continue
                    predict_frames_indice = np.arange(
                        i - window,
                        i + window + 1,
                    )

                    if self.config.feature_window > 0:
                        feature_window = self.config.feature_window // 2
                        feature_indices = np.arange(
                            max(i - feature_window, 0),
                            min(i + feature_window + 1, len(w_df))
                        )
                        feature = features[feature_indices]

                        if i - feature_window < 0:
                            feature = np.concatenate([
                                np.zeros((feature_window - i, len(self.config.feature_cols))) - 1,
                                feature
                            ], axis=0)
                        if i + feature_window + 1 > len(w_df):
                            feature = np.concatenate([
                                feature,
                                np.zeros((feature_window + i + 1 - len(w_df), len(self.config.feature_cols))) - 1
                            ], axis=0)
                        assert feature.shape == (self.config.feature_window, len(self.config.feature_cols))
                    else:
                        feature = [0]

                    assert len(predict_frames_indice) == self.config.n_predict_frames
                    if distances[predict_frames_indice].min() > self.config.distance_threshold:
                        continue

                    if not self.test and contacts[predict_frames_indice].sum() == 0:
                        # down sampling (only negative)
                        if not is_g:
                            if distances[i] < 0.75 and np.random.random() > self.config.negative_sample_ratio_close:
                                continue
                            if distances[i] >= 0.75 and np.random.random() > self.config.negative_sample_ratio_far:
                                continue
                        elif is_g and np.random.random() > self.config.negative_sample_ratio_g:
                            continue

                    if not self._exist_files(game_play=game_play, id_1=id_1, id_2=id_2, frames=frame_indice,
                                             filelist=filelist, n_frames=self.config.n_frames,
                                             exist_image_threshold=self.config.exist_image_threshold):
                        failed_count += 1
                        continue
                    if not self._exist_files(game_play=game_play, id_1=id_1, id_2=id_2,
                                             frames=frames[predict_frames_indice], filelist=filelist,
                                             n_frames=self.config.n_predict_frames,
                                             exist_image_threshold=self.config.exist_center_image_threshold):
                        failed_count += 1
                        continue

                    contacts_ = contacts[predict_frames_indice]
                    if self.config.soft_label_range is not None and not self.test:
                        contacts_ = np.clip(contacts_, self.config.soft_label_range[0], self.config.soft_label_range[1])

                    contacts_all.append({
                        "is_g": is_g,
                        "contact": contacts_.mean()
                    })

                    self.items.append({
                        "contact_id": contact_ids[predict_frames_indice],
                        "game_play": game_play,
                        "id_1": id_1,
                        "id_2": id_2,
                        "contact": contacts_,
                        "frames": frame_indice,
                        "features": feature,
                        "is_g": is_g,
                    })
                    is_g_count += is_g
            else:
                contact_ids_ = [""] * self.config.max_frames
                contact_ids_[:len(contact_ids)] = contact_ids
                if self.config.soft_label_range is not None and not self.test:
                    contacts = np.clip(contacts, self.config.soft_label_range[0], self.config.soft_label_range[1])
                contacts_ = np.zeros(self.config.max_frames) - 1
                contacts_[:len(contacts)] = contacts
                frames_ = np.arange(frames[0], frames[0]+self.config.max_frames*self.config.step, self.config.step)
                self.items.append({
                    "contact_id": np.array(contact_ids_),
                    "game_play": game_play,
                    "id_1": id_1,
                    "id_2": id_2,
                    "contact": contacts_,
                    "frames": frames_,
                    "features": [0],
                    "is_g": is_g,
                })
                is_g_count += is_g
                contacts_all.append({
                    "is_g": is_g,
                    "contact": contacts_[contacts_ != -1].mean()
                })
        logger.info(f"finished. extracted={len(self.items)} (total={len(df)}, is_g={is_g_count}, failed={failed_count})")
        logger.info(f"contacts_distribution: \n {pd.DataFrame(contacts_all).groupby(['is_g', 'contact']).size()}")

    def __len__(self):
        return len(self.items)

    def imread_6channel(self, game_play, view, id_1, id_2, frame):
        if self.imread(game_play, view, id_1, id_2, frame, prefix="_original") is None:
            return None
        return np.concatenate([
            self.imread(game_play, view, id_1, id_2, frame, prefix="_original"),
            self.imread(game_play, view, id_1, id_2, frame, prefix="_filter"),
        ], axis=2)

    def imread(self, game_play, view, id_1, id_2, frame, prefix=""):
        key = self._get_key(game_play, view, id_1, id_2, frame, prefix)
        # random drop frame
        if np.random.random() < self.config.p_drop_frame and not self.test:
            return None
        if not self.submission_mode:
            isfile = key in self.filelist
        elif self.image_dict is not None:
            isfile = key in self.image_dict
        else:
            isfile = os.path.isfile(key)
        if isfile:
            if self.image_dict is not None:
                return self.image_dict[key]
            if self.config.extention == ".npy":
                img = np.load(key)
            if self.config.extention == ".jpg":
                img = cv2.imread(key)
        else:
            return None
        if self.config.grayscale or "2.5d" in self.config.model_name:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
            if "2.5d" not in self.config.model_name:
                img = np.concatenate([img, img, img], axis=2)
        return img

    def aug_video(self, frames):
        seed = random.randint(0, 99999)
        aug_vid = []
        for frame in frames:
            random.seed(seed)
            if not self.test:
                aug_vid.append((self.config.transforms_train(image=frame))['image'])
            else:
                aug_vid.append((self.config.transforms_eval(image=frame))['image'])
        return np.stack(aug_vid)

    def __getitem__(self, index):
        item = self.items[index]  # {movie_id}/{start_time}

        game_play = item["game_play"]
        contact_id = item["contact_id"]
        frames = item["frames"]
        labels = item["contact"]
        id_1 = item["id_1"]
        id_2 = item["id_2"]
        is_g = item["is_g"]
        feature = item["features"]

        imgs_all = []
        for view in ["Endzone", "Sideline"]:
            if self.config.channel_6:
                imgs = [self.imread_6channel(game_play, view, id_1, id_2, frame) for frame in frames]
            else:
                imgs = [self.imread(game_play, view, id_1, id_2, frame) for frame in frames]

            if self.config.interpolate_image:
                # 外挿
                img_indices = [i for i, img in enumerate(imgs) if img is not None]

                if len(img_indices) == 0:
                    imgs_all.extend(
                        [np.zeros(self.img_shape, dtype=np.uint8) + self.config.pad_image_values for _ in
                         range(len(imgs))])
                    continue
                first_img_idx = img_indices[0]

                # 最初の画像まで
                for idx in range(first_img_idx):
                    if self.config.interpolate_outside:
                        imgs[idx] = imgs[first_img_idx].copy()
                    else:
                        imgs[idx] = np.zeros(self.img_shape, dtype=np.uint8) + self.config.pad_image_values

                # 最初の画像から最後まで
                if self.config.interpolate_outside:
                    for i in range(len(imgs) - 1):
                        if imgs[i + 1] is None:
                            imgs[i + 1] = imgs[i].copy()
                else:
                    for i in range(img_indices[-1]):
                        if imgs[i + 1] is None:
                            imgs[i + 1] = imgs[i].copy()
                    for i in range(img_indices[-1] + 1, len(imgs)):
                        imgs[i] = np.zeros(self.img_shape, dtype=np.uint8) + self.config.pad_image_values
            else:
                imgs = [img if img is not None else np.zeros(self.img_shape, dtype=np.uint8) + self.config.pad_image_values
                        for img in imgs]
            imgs_all.extend(imgs)

        if len(self.img_shape) == 3:
            frames = np.stack(imgs_all, axis=0).transpose(3, 0, 1, 2)  # shape = (C, n_frame*n_view, H, W)
            frames = self.aug_video(frames.transpose(1, 2, 3, 0)) # shape = (n_frame, H, W, C)
            frames = frames.transpose(3, 0, 1, 2)
        elif len(self.img_shape) == 1:
            frames = np.stack(imgs_all, axis=0)  # shape = (n_frame*n_view, feature)

        return contact_id.tolist(), torch.Tensor(frames), torch.Tensor(labels), torch.LongTensor([is_g]), torch.Tensor(feature)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch, config):
    model.train()
    loss_meter = AverageMeter()
    loss_concat_meter = AverageMeter()
    loss_endzone_meter = AverageMeter()
    loss_sideline_meter = AverageMeter()

    data_length = int(len(dataloader) * config.data_per_epoch)
    tk0 = tqdm.tqdm(enumerate(dataloader), total=data_length)

    scaler = torch.cuda.amp.GradScaler()
    count = 0
    loss_100 = []
    for bi, data in tk0:
        count += 1
        batch_size = len(data)

        x = data[1].to(device)
        label = data[2].to(device)
        is_g = data[3].to(device)
        feature = data[4].to(device)
        optimizer.zero_grad()

        enabled = True
        gnn = type(config) == ConfigForGNN
        if gnn:
            enabled = False
            label = x.edata["label"]
        with torch.cuda.amp.autocast(enabled=enabled):
            pred, pred_endzone, pred_sideline = model(x, is_g, feature)
            if type(config) == ConfigForTransformer:
                mask = is_g
                mask_flat = mask.flatten()
                label = label.flatten()
                label = label[mask_flat == 0]
                pred = pred.flatten()
                pred = pred[mask_flat == 0]
                loss = criterion(pred, label)
            else:
                if config.predict_all_frames:
                    label = label.flatten()
                    mask = label != -1
                    pred = pred.flatten()[mask]
                    pred_endzone = pred_endzone.flatten()[mask]
                    pred_sideline = pred_sideline.flatten()[mask]
                    label = label[mask]

                loss_concat = criterion(pred.flatten(), label.flatten())
                if type(config) == Config and config.calc_single_view_loss:
                    loss_endzone = criterion(pred_endzone.flatten(), label.flatten())
                    loss_sideline = criterion(pred_sideline.flatten(), label.flatten())
                    loss = loss_concat + (loss_endzone + loss_sideline) * config.calc_single_view_loss_weight
                else:
                    loss = loss_concat
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_meter.update(loss.detach().item(), batch_size)
        if config.calc_single_view_loss:
            loss_concat_meter.update(loss_concat.detach().item(), batch_size)
            loss_endzone_meter.update(loss_endzone.detach().item(), batch_size)
            loss_sideline_meter.update(loss_sideline.detach().item(), batch_size)
            wandb.log({
                "loss": loss_meter.avg,
                "loss_concat": loss_concat_meter.avg,
                "loss_endzone": loss_endzone_meter.avg,
                "loss_sideline": loss_sideline_meter.avg,
                "lr": optimizer.param_groups[0]['lr']
            })
        else:
            wandb.log({
                "loss": loss_meter.avg,
                "lr": optimizer.param_groups[0]['lr']
            })
        tk0.set_postfix(Loss=loss_meter.avg,
                        LossSnap=np.mean(loss_100),
                        LossCat=loss_concat_meter.avg,
                        LossSide=loss_sideline_meter.avg,
                        LossEnd=loss_endzone_meter.avg,
                        Epoch=epoch,
                        LR=optimizer.param_groups[0]['lr'])

        if count > data_length:
            break

    return loss_meter.avg


def eval_fn(data_loader, model, criterion, device, config: Config):
    loss_score = AverageMeter()

    model.eval()
    tk0 = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    preds = []
    preds_endzone = []
    preds_sideline = []
    contact_ids = []
    labels = []
    features = []

    gnn = type(config) == ConfigForGNN
    transformer = type(config) == ConfigForTransformer
    with torch.no_grad():
        for bi, data in tk0:
            batch_size = len(data)

            contact_id = data[0]
            x = data[1].to(device)
            label = data[2].to(device)
            is_g = data[3].to(device)
            feature = data[4].to(device)

            enabled = True
            if gnn:
                enabled = False
                label = x.edata["label"]
            with torch.cuda.amp.autocast(enabled=enabled):
                pred, pred_endzone, pred_sideline = model(x, is_g, feature)
                if type(config) == Config:
                    if config.predict_all_frames:
                        label = label.flatten()
                        mask = label != -1
                        pred = pred.flatten()[mask]
                        pred_endzone = pred_endzone.flatten()[mask]
                        pred_sideline = pred_sideline.flatten()[mask]
                        label = label[mask]
                    loss = criterion(pred.flatten(), label.flatten())

                elif type(config) == ConfigForTransformer:
                    mask = is_g
                    mask_flat = mask.flatten()
                    label = label.flatten()
                    label = label[mask_flat == 0]
                    pred = pred.flatten()
                    pred = pred[mask_flat == 0]
                    loss = criterion(pred, label)

            loss_score.update(loss.detach().item(), batch_size)
            tk0.set_postfix(Eval_Loss=loss_score.avg)

            contact_id = np.array(contact_id).transpose(1, 0).flatten()
            contact_id = contact_id[contact_id != ""]
            contact_ids.extend(np.array(contact_id))

            preds.extend(torch.sigmoid(pred.flatten()).detach().cpu().numpy())
            if config.calc_single_view_loss:
                preds_endzone.extend(torch.sigmoid(pred_endzone.flatten()).detach().cpu().numpy())
                preds_sideline.extend(torch.sigmoid(pred_sideline.flatten()).detach().cpu().numpy())
            labels.extend(label.flatten().detach().cpu().numpy())

            del x, label, pred

    preds = np.array(preds).astype(np.float16)
    if config.calc_single_view_loss:
        preds_endzone = np.array(preds_endzone).astype(np.float16)
        preds_sideline = np.array(preds_sideline).astype(np.float16)

    labels = np.array(labels).astype(np.float16)

    idx = np.arange(config.n_predict_frames) - config.n_predict_frames // 2
    indices = np.tile(idx, len(preds) // config.n_predict_frames)

    if type(config) == ConfigForTransformer:
        df_ret = pd.DataFrame({
            "contact_id": contact_ids,
            "score": preds.tolist(),
            "label": labels.tolist(),
        })
    elif config.predict_all_frames:
        df_ret = pd.DataFrame({
            "contact_id": contact_ids,
            "score": preds.tolist(),
            "score_endzone": preds_endzone.tolist(),
            "score_sideline": preds_sideline.tolist(),
            "label": labels.tolist(),
        })
    elif config.calc_single_view_loss:
        df_ret = pd.DataFrame({
            "contact_id": contact_ids,
            "score": preds.tolist(),
            "score_endzone": preds_endzone.tolist(),
            "score_sideline": preds_sideline.tolist(),
            "label": labels.tolist(),
            "index": indices.tolist(),
        })
    else:
        df_ret = pd.DataFrame({
            "contact_id": contact_ids,
            "score": preds,
            "label": labels,
            "index": indices,
            "view": ["total"] * len(contact_ids)
        })
    return df_ret, loss_score.avg


def get_key(fname):
    game_play = os.path.dirname(fname).split("/")[-2]
    id_1 = os.path.basename(fname).split("_")[0]
    id_2 = os.path.basename(fname).split("_")[1]
    frame = os.path.basename(fname).split("_")[2].split(".")[0]
    view = os.path.dirname(fname).split("/")[-1]
    return f"{game_play}_{view}_{id_1}_{id_2}_{frame}"


def save_feature(model, device: str, config: Config):
    model.eval()
    preds = []
    preds_2d = []
    files = []

    dataset = NFLDatasetForFeatureExtraction(
        base_dir=f"{config.base_dir}/{config.image_path}",
    )
    if config.debug:
        dataset.files = dataset.files[:1000]
        num_workers = 0
    else:
        num_workers = 8
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers
    )
    tk0 = tqdm.tqdm(enumerate(loader), total=len(loader))

    with torch.no_grad():
        for bi, data in tk0:
            x = data[0].to(device)
            file = data[1]

            file = np.array(file).flatten().tolist()
            with torch.cuda.amp.autocast():
                pred_2d = model.forward_features(x)
                pred = F.adaptive_avg_pool2d(pred_2d, 1).squeeze(3).squeeze(2)
                pred = pred.detach().cpu().numpy().astype(np.float16)
            for i in range(len(file)):
                f_structure = file[i].split("/")[-3:]
                output_dir = f"{config.output_dir}/2d/" + "/".join(f_structure[:-1])
                os.makedirs(output_dir, exist_ok=True)

                np.save(f"{output_dir}/{f_structure[-1].replace('.jpg', '.npy')}", pred[i])

class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.ln1 = nn.LayerNorm(state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.ln2 = nn.LayerNorm(state_size)

    def forward(self, x):
        x = self.lr1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.lr2(x)
        x = self.ln2(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self,
                 config: ConfigForTransformer):
        super().__init__()
        self.config = config
        self.fc1 = nn.LazyLinear(self.config.hidden_size_transformer)
        self.ln1 = nn.LayerNorm(self.config.hidden_size_transformer)
        self.bn = nn.BatchNorm1d(self.config.max_length)

        if self.config.transformer == "only_encoder":
            transformer_encoder = nn.TransformerEncoderLayer(
                d_model=self.config.hidden_size_transformer,
                nhead=self.config.nhead,
                dropout=self.config.dropout_transformer,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                num_layers=self.config.num_layer_transformer,
                encoder_layer=transformer_encoder,
                # norm=nn.LayerNorm(self.config.num_layer_transformer),
            )
        elif self.config.transformer == "transformer":
            self.transformer = nn.Transformer(
                d_model=self.config.hidden_size_transformer,
                nhead=self.config.nhead,
                num_decoder_layers=self.config.num_layer_transformer,
                num_encoder_layers=self.config.num_layer_transformer,
                dropout=self.config.dropout_transformer,
                batch_first=True,
            )

        self.rnn = nn.LSTM(
            input_size=self.config.hidden_size_transformer,
            hidden_size=self.config.hidden_size_transformer,
            num_layers=self.config.num_layer_rnn,
            batch_first=True,
        )
        if self.config.g_emb:
            self.g_embedding = nn.Embedding(2, 32)
            self.ffn = FFN(self.config.hidden_size_transformer + 32)
        else:
            self.ffn = FFN(self.config.hidden_size_transformer)
        self.layer_norm = nn.LayerNorm(self.config.hidden_size_transformer)
        self.fc = nn.LazyLinear(1)

    def forward(self, x, mask, is_g):
        if self.config.apply_norm:
            x = self.bn(x)
        x = self.fc1(x)
        x_feat = self.ln1(x)
        mask_ = (mask[:, None, :] * mask[:, :, None])
        mask_ = torch.cat([m[np.newaxis].tile(self.config.nhead, 1, 1) for m in mask_], dim=0)
        if self.config.transformer == "only_encoder":
            x = self.transformer(x_feat, mask_)  # (bs, seq_len, n_feature) -> (bs, seq_len, n_feature)
        elif self.config.transformer == "transformer":
            x = self.transformer(x_feat, x_feat, mask_, mask_)  # (bs, seq_len, n_feature) -> (bs, seq_len, n_feature)

        if self.config.apply_lstm:
            x = self.rnn(x)[0]  # (bs, seq_len, n_feature) -> (bs, seq_len, n_feature)
        x = self.layer_norm(x)

        if self.config.residual:
            x += x_feat
        if self.config.g_emb:
            g_emb = self.g_embedding(is_g)
            g_emb = g_emb.tile((1, x.shape[1], 1))
            x = torch.cat([x, g_emb], dim=2)
        if self.config.apply_ffn:
            x = self.ffn(x)  # (bs, seq_len, n_feature) -> (bs, seq_len, n_feature)
        x = self.fc(x)  # (bs, seq_len, n_feature) -> (bs, seq_len, 1)
        x = x.squeeze(2)  # (bs, seq_len, 1) -> (bs, seq_len)
        return x, None, None


class NFLGraphModel(nn.Module):
    def __init__(self, config: ConfigForGNN):
        super().__init__()

        self.egat1 = EGATConv(in_node_feats=3,
                              in_edge_feats=64,
                              out_node_feats=config.node_feats,
                              out_edge_feats=config.edge_feats,
                              num_heads=4)
        self.egat2 = EGATConv(in_node_feats=config.node_feats,
                              in_edge_feats=config.edge_feats,
                              out_node_feats=1,
                              out_edge_feats=1,
                              num_heads=4)
        self.fc_distance = nn.Linear(1, 32)
        self.emb_sameteam = nn.Embedding(2, 32)

    def forward(self, graph, *args, **kwargs):
        edge_feats = torch.cat([
            self.fc_distance(graph.edata["distance"]),
            self.emb_sameteam(graph.edata["is_same_team"].squeeze(1))
        ], axis=1)
        node_feats, edge_feats = self.egat1(graph, graph.ndata["feature"], edge_feats)

        node_feats = node_feats.mean(dim=1)
        edge_feats = edge_feats.mean(dim=1)

        node_feats, edge_feats = self.egat2(graph, node_feats, edge_feats)

        return edge_feats.mean(dim=1), None, None


class SimpleConv3d(nn.Module):
    def __init__(self,
                 config: Config):
        super(SimpleConv3d, self).__init__()
        hid = config.hidden_size_3d
        self.fc = nn.LazyConv3d(hid,
                                config.kernel_size_3d,
                                bias=False,
                                stride=config.stride_3d,
                                padding=(0, 1, 1),
                                dilation=config.dilation_3d)
        self.bn = nn.BatchNorm3d(hid)
        self.do = nn.Dropout(config.dropout_3d)
        self.num_features = hid
        self.activation = config.activation()

    def forward(self, x):
        x = self.do(self.activation(self.bn(self.fc(x))))
        return x


class StackConv3d(nn.Module):
    def __init__(self,
                 config: Config):
        super(StackConv3d, self).__init__()
        hid = config.hidden_size_3d

        models = []

        for dilation in config.stack_dilations:
            models.append(nn.Sequential(
                nn.LazyConv3d(hid,
                              (config.kernel_size_3d[0] // dilation, config.kernel_size_3d[1], config.kernel_size_3d[2]),
                              bias=False,
                              stride=(max(config.stride_3d[0] // dilation, 1), config.stride_3d[1], config.stride_3d[2]),
                              padding=(0, 1, 1),
                              dilation=(dilation, 1, 1)),
                nn.BatchNorm3d(hid),
                nn.Dropout(config.dropout_3d),
            ))
        self.models = nn.ModuleList(models)
        self.num_features = hid
        self.activation = config.activation()

    def forward(self, x):

        rets = []
        for model in self.models:
            ret = self.activation(model(x))
            ret = F.adaptive_avg_pool3d(ret, 1).squeeze(4).squeeze(3).squeeze(2)
            rets.append(ret)
        rets = torch.cat(rets, dim=1)
        return rets


class SlowFastConv3d(nn.Module):
    def __init__(self,
                 config: Config):
        super(SlowFastConv3d, self).__init__()
        hid = config.hidden_size_3d
        self.fc_slow = nn.LazyConv3d(hid,
                                     config.kernel_size_3d,
                                     bias=False,
                                     stride=config.stride_3d,
                                     padding=(0, 1, 1),
                                     dilation=(1, 1, 1))
        self.fc_fast = nn.LazyConv3d(hid,
                                     config.kernel_size_3d,
                                     bias=False,
                                     stride=config.stride_3d,
                                     padding=(0, 1, 1),
                                     dilation=(2, 1, 1))
        self.bn = nn.BatchNorm3d(hid)
        self.do = nn.Dropout(config.dropout_3d)
        self.num_features = hid

    def forward(self, x):
        x_slow = self.do(F.relu(self.bn(self.fc_slow(x))))
        x_fast = self.do(F.relu(self.bn(self.fc_fast(x))))
        return x_slow, x_fast


class ThreeLayerConv3d(nn.Module):
    def __init__(self,
                 config: Config):
        super(ThreeLayerConv3d, self).__init__()
        hid = config.hidden_size_3d
        self.fc = nn.LazyConv3d(hid, config.kernel_size_3d, bias=False, stride=config.stride_3d, padding=(0, 1, 1))
        self.bn = nn.BatchNorm3d(hid)
        self.do = nn.Dropout(0.2)
        self.pool = nn.MaxPool3d((3, 1, 1))
        self.fc2 = nn.LazyConv3d(hid*2, (3, 1, 1), bias=False, stride=(2, 1, 1))
        self.bn2 = nn.BatchNorm3d(hid*2)
        self.do2 = nn.Dropout(0.25)
        self.fc3 = nn.LazyConv3d(hid*2, (2, 1, 1), bias=False, stride=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(hid*2)
        self.do3 = nn.Dropout(0.25)
        self.num_features = hid

    def forward(self, x):
        x = self.pool(self.do(F.relu(self.bn(self.fc(x)))))
        x = self.do2(F.relu(self.bn2(self.fc2(x))))
        x = self.do3(F.relu(self.bn3(self.fc3(x))))
        return x


class TwoLayerConv3d(nn.Module):
    def __init__(self,
                 config: Config):
        super(TwoLayerConv3d, self).__init__()
        hid = config.hidden_size_3d
        self.fc = nn.LazyConv3d(hid, config.kernel_size_3d[0], bias=False, stride= config.stride_3d[0], padding=(0, 1, 1))
        self.bn = nn.BatchNorm3d(hid)
        self.do = nn.Dropout(0.2)
        self.pool = nn.MaxPool3d((3, 1, 1))
        self.fc2 = nn.LazyConv3d(hid*2, config.kernel_size_3d[1], bias=False, stride=config.stride_3d[1])
        self.bn2 = nn.BatchNorm3d(hid*2)
        self.do2 = nn.Dropout(0.25)
        self.activation = config.activation()

    def forward(self, x):
        x = self.pool(self.do(self.activation(self.bn(self.fc(x)))))
        x = self.do2(self.activation(self.bn2(self.fc2(x))))
        return x


class SimpleConv1D(nn.Module):
    def __init__(self,
                 config: Config):
        super(SimpleConv1D, self).__init__()
        self.fc = nn.LazyConv1d(config.hidden_size_1d, config.kernel_size_conv1d, stride=config.stride_conv1d, bias=False)
        self.bn = nn.LazyBatchNorm1d()
        self.do = nn.Dropout(config.dropout_seq)
        self.pl = nn.MaxPool2d(config.stride_conv1d)
        self.activation = config.activation()

    def forward(self, x):
        x = self.pl(self.do(self.activation(self.bn(self.fc(x)))))
        return x

class ThreeLayerConv1DUnit(nn.Module):
    def __init__(self,
                 config: Config):
        super(ThreeLayerConv1DUnit, self).__init__()
        self.fc = nn.LazyConv1d(config.hidden_size_1d, 13, bias=False, stride=2)
        self.bn = nn.LazyBatchNorm1d()
        self.do = nn.Dropout(config.dropout_seq)
        self.pl = nn.MaxPool2d(3)
        self.fc2 = nn.LazyConv1d(config.hidden_size_1d * 2, 7, bias=False, stride=1)
        self.bn2 = nn.LazyBatchNorm1d()
        self.do2 = nn.Dropout(config.dropout_seq)
        self.fc3 = nn.LazyConv1d(config.hidden_size_1d * 2, 5, bias=False, stride=1)
        self.bn3 = nn.LazyBatchNorm1d()
        self.do3 = nn.Dropout(config.dropout_seq)
        self.num_features = config.hidden_size_1d * 2
        self.activation = config.activation()

    def forward(self, x):
        x = self.pl(self.do(self.activation(self.bn(self.fc(x)))))
        x = self.do2(self.activation(self.bn2(self.fc2(x))))
        x = self.do3(self.activation(self.bn3(self.fc3(x))))
        return x

class SequenceModel(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 config: Config):
        super().__init__()
        self.config = config
        self.n_frames = self.config.n_frames

        if config.seq_model == "lstm":
            self.model = nn.LSTM(
                hidden_size,
                hidden_size // 2,
                bidirectional=True,
                batch_first=True,
                dropout=self.config.dropout_seq
            )
        elif config.seq_model == "gru":
            self.model = nn.GRU(
                hidden_size,
                hidden_size // 2,
                bidirectional=True,
                batch_first=True,
                dropout=self.config.dropout_seq
            )
        elif config.seq_model == "1dcnn":
            self.model = nn.LazyConv1d(
                out_channels=config.hidden_size_1d,
                kernel_size=config.kernel_size_conv1d,
                stride=config.stride_conv1d,
                bias=False
            )
        elif config.seq_model == "1dcnn_simple":
            self.model = SimpleConv1D(config=config)
        elif config.seq_model == "1dcnn_3layers":
            self.model = ThreeLayerConv1DUnit(config)
        elif config.seq_model =="3dcnn_simple":
            self.model = SimpleConv3d(config=config)
        elif config.seq_model == "3dcnn_stack":
            self.model = StackConv3d(config=config)
        elif config.seq_model == "3dcnn_2layers":
            self.model = TwoLayerConv3d(config=config)
        elif config.seq_model == "3dcnn_3layers":
            self.model = ThreeLayerConv3d(config=config)
        else:
            raise ValueError(config.seq_model)

    def forward(self, x):
        x = self.model(x)
        if self.config.seq_model in ["lstm", "gru"]:
            return x[0]
        else:
            return x


class Model2DTo1D(nn.Module):
    def __init__(self,
                 config: Config):
        super().__init__()
        self.config = config
        pretrained = not config.submission_mode
        if "cnn_2d_" in self.config.model_name:
            self.cnn_2d = timm.create_model(config.model_name.replace("cnn_2d_", ""), num_classes=0, pretrained=pretrained)

        self.seq_model = SequenceModel(
            hidden_size=self.config.pretrained_feature_dim,
            config=config
        )
        if self.config.fc == "simple":
            self.fc = nn.LazyLinear(config.n_predict_frames)
        elif self.config.fc == "2layers":
            self.fc = nn.Sequential(
                nn.LazyLinear(32),
                nn.Dropout(self.config.fc_dropout),
                nn.GELU(),
                nn.Linear(32, config.n_predict_frames)
            )
        self.fc_contact = FC(config)
        self.fc_g = FC(config)
        self.fc_endzone_contact = FC(config)
        self.fc_sideline_contact = FC(config)
        self.fc_endzone_g = FC(config)
        self.fc_sideline_g = FC(config)

    def _forward_g_contact(self, model_g, model_contact, x, is_g):
        x_contact = model_contact(x)  # (bs, n_predict_frames)
        x_g = model_g(x)  # (bs, n_predict_frames)

        not_is_g = (is_g == 0)
        x = x_contact * not_is_g + x_g * is_g  # (bs, n_predict_frames)
        return x

    def forward(self, x, is_g, feature):
        if "cnn_2d_" in self.config.model_name:
            bs, C, seq_len, W, H = x.shape
            x = x.permute(0, 2, 1, 3, 4)  # (bs, seq_len*n_view, C, W, H)
            x = x.reshape(bs * seq_len, C, W, H)  # (bs*seq_len*n_view, C, W, H)
            x = self.cnn_2d(x)  # (bs*seq_len*n_view, features)
            x = x.reshape(bs, seq_len, -1)

        bs, seq_len, f_dim = x.shape  # shape = (bs, seq_len*n_view, features)
        x = x.reshape(bs*2, seq_len//2, f_dim)  # shape = (bs*n_view, seq_len, features)

        x = self.seq_model(x) # (bs*n_view, seq_len, features) -> (bs*n_view, hidden)
        if self.config.pooling == "avgpool":
            x = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        elif self.config.pooling == "maxpool":
            x = F.adaptive_max_pool1d(x, 1).squeeze(2)

        x_endzone = self._forward_g_contact(self.fc_endzone_g, self.fc_endzone_contact, x[::2], is_g)
        x_sideline = self._forward_g_contact(self.fc_sideline_g, self.fc_sideline_contact, x[1::2], is_g)
        if self.config.fc_sideend == "concat":
            x = x.reshape(bs, -1)
            x = self._forward_g_contact(self.fc_g, self.fc_contact, x, is_g)
        return x, x_endzone, x_sideline

    def forward_features(self, x):
        return self.cnn_2d.forward_features(x)


class Model2DTo3D(nn.Module):
    def __init__(self,
                 config: Config):
        super().__init__()
        self.config = config
        if "cnn_2d_" in self.config.model_name:
            self.cnn_2d = timm.create_model(config.model_name.replace("cnn_2d_", ""), num_classes=0, pretrained=True)

        self.seq_model = SequenceModel(
            hidden_size=self.config.pretrained_feature_dim,
            config=config
        )
        if self.config.fc == "simple":
            self.fc = nn.LazyLinear(config.n_predict_frames)
        elif self.config.fc == "2layer":
            self.fc = nn.Sequential(
                nn.LazyLinear(32),
                nn.Dropout(self.config.fc_dropout),
                nn.GELU(),
                nn.Linear(32, config.n_predict_frames)
            )
        self.fc_weight = nn.LazyLinear(1)
        self.fc_contact = nn.LazyLinear(config.n_predict_frames)
        self.fc_g = nn.LazyLinear(config.n_predict_frames)
        self.fc_endzone_contact = nn.LazyLinear(config.n_predict_frames)
        self.fc_sideline_contact = nn.LazyLinear(config.n_predict_frames)
        self.fc_endzone_g = nn.LazyLinear(config.n_predict_frames)
        self.fc_sideline_g = nn.LazyLinear(config.n_predict_frames)

    def _forward_g_contact(self, model_g, model_contact, x, is_g):
        x_contact = model_contact(x)  # (bs, n_predict_frames)
        x_g = model_g(x)  # (bs, n_predict_frames)

        not_is_g = (is_g == 0)
        x = x_contact * not_is_g + x_g * is_g  # (bs, n_predict_frames)
        return x

    def forward(self, x, is_g, feature):
        if "cnn_2d_" in self.config.model_name:
            bs, C, seq_len, W, H = x.shape
            x = x.permute(0, 2, 1, 3, 4)  # (bs, seq_len*n_view, C, W, H)
            x = x.reshape(bs * seq_len, C, W, H)  # (bs*seq_len*n_view, C, W, H)
            x = self.cnn_2d.forward_features(x)  # (bs, n_view*seq_len, C, W, H)
            x = x.reshape(bs, -1, x.shape[1], x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1, 3, 4)

        bs, C, seq_len, W, H = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (bs, n_view*seq_len, C, W, H)
        x = x.reshape(bs*2, seq_len//2, C, W, H)   # (bs*n_view, seq_len, C, W, H)
        x = x.permute(0, 2, 1, 3, 4)

        x = self.seq_model(x)  # (bs*n_view, seq_len, features) -> (bs*n_view, hidden)

        if len(x.shape) == 5:
            x = F.adaptive_avg_pool3d(x, 1).squeeze(4).squeeze(3).squeeze(2)

        x_endzone = self._forward_g_contact(self.fc_endzone_g, self.fc_endzone_contact, x[::2], is_g)
        x_sideline = self._forward_g_contact(self.fc_sideline_g, self.fc_sideline_contact, x[1::2], is_g)
        if self.config.fc_sideend == "concat":
            x = x.reshape(bs, -1)
            x = self._forward_g_contact(self.fc_g, self.fc_contact, x, is_g)
        return x, x_endzone, x_sideline

    def forward_features(self, x):
        return self.cnn_2d.forward_features(x)


class Model2D(nn.Module):
    def __init__(self,
                 config: Config):
        super().__init__()
        self.config = config
        if config.pretrained and not config.submission_mode:
            pretrained = True
        else:
            pretrained = False
        self.cnn_2d = timm.create_model(config.model_name.replace("cnn_2d_", ""), num_classes=0, pretrained=pretrained)
        self.seq_model = SequenceModel(
            hidden_size=self.cnn_2d.num_features,
            config=config
        )
        if self.config.fc == "simple":
            self.fc = nn.LazyLinear(config.n_predict_frames)
        elif self.config.fc == "2layer":
            self.fc = nn.Sequential(
                nn.LazyLinear(32),
                nn.Dropout(self.config.fc_dropout),
                nn.GELU(),
                nn.Linear(32, config.n_predict_frames)
            )
        self.fc_contact = nn.LazyLinear(config.n_predict_frames)
        self.fc_g = nn.LazyLinear(config.n_predict_frames)
        self.fc_endzone_contact = nn.LazyLinear(config.n_predict_frames)
        self.fc_sideline_contact = nn.LazyLinear(config.n_predict_frames)
        self.fc_endzone_g = nn.LazyLinear(config.n_predict_frames)
        self.fc_sideline_g = nn.LazyLinear(config.n_predict_frames)

    def _forward_g_contact(self, model_g, model_contact, x, is_g):
        x_contact = model_contact(x)  # (bs, n_predict_frames)
        x_g = model_g(x)  # (bs, n_predict_frames)

        not_is_g = (is_g == 0)
        x = x_contact * not_is_g + x_g * is_g  # (bs, n_predict_frames)
        return x

    def forward(self, x, is_g, feature):
        bs, C, seq_len, W, H = x.shape  # seq_len = 1
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(bs*seq_len, C, W, H)

        x_endzone = None
        x_sideline = None
        x = self.cnn_2d(x)  # (bs*n_view, features)
        x_endzone = self._forward_g_contact(self.fc_endzone_g, self.fc_endzone_contact, x[::2], is_g)
        x_sideline = self._forward_g_contact(self.fc_sideline_g, self.fc_sideline_contact, x[1::2], is_g)
        x = torch.cat([x[::2], x[1::2]], dim=1)  # (bs, n_view*features)
        x = self._forward_g_contact(self.fc_g, self.fc_contact, x, is_g)

        return x, x_endzone, x_sideline

    def forward_features(self, x):
        return self.cnn_2d.forward_features(x)


class FC(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        if config.fc == "simple":
            self.fc = nn.LazyLinear(config.n_predict_frames)
        elif config.fc == "2layers":
            self.fc1 = nn.LazyLinear(32)
            self.dout1 = nn.Dropout(config.fc_dropout)
            self.act1 = nn.ReLU()
            self.fc2 = nn.Linear(32, config.n_predict_frames)

    def forward(self, x):
        if self.config.fc == "simple":
            return self.fc(x)
        elif self.config.fc == "2layers":
            x = self.fc1(x)
            x = self.dout1(x)
            x = self.act1(x)
            x = self.fc2(x)
            return x


class Model2p5D(nn.Module):
    def __init__(self,
                 config: Config):
        super().__init__()

        if config.pretrained and not config.submission_mode:
            pretrained = True
        else:
            pretrained = False
        self.config = config
        self.cnn_2d = timm.create_model(config.model_name.replace("cnn_2.5d_", ""), num_classes=0, pretrained=pretrained,
                                        in_chans=self.config.n_frames)

        self.fc_contact = FC(config)
        self.fc_g = FC(config)
        self.fc_endzone_contact = FC(config)
        self.fc_sideline_contact = FC(config)
        self.fc_endzone_g = FC(config)
        self.fc_sideline_g = FC(config)

    def _forward_g_contact(self, model_g, model_contact, x, is_g):
        x_contact = model_contact(x)  # (bs, n_predict_frames)
        x_g = model_g(x)  # (bs, n_predict_frames)

        not_is_g = (is_g == 0)
        x = x_contact * not_is_g + x_g * is_g  # (bs, n_predict_frames)
        return x

    def forward(self, x, is_g, feature):
        bs, _, seq_len, W, H = x.shape  # C = 1
        x = x.squeeze(1)  # (bs, n_view*seq_len, W, H)
        x = x.reshape(bs*2, seq_len//2, W, H)  # (bs*n_view, seq_len, W, H)
        x = self.cnn_2d(x)  # (bs*n_view, features)

        x_endzone = self._forward_g_contact(self.fc_endzone_g, self.fc_endzone_contact, x[::2], is_g)
        x_sideline = self._forward_g_contact(self.fc_sideline_g, self.fc_sideline_contact, x[1::2], is_g)
        x = torch.cat([x[::2], x[1::2]], dim=1) # (bs, n_view*features)
        x = self._forward_g_contact(self.fc_g, self.fc_contact, x, is_g)
        return x, x_endzone, x_sideline


class Model2p5DTo3D(nn.Module):
    def __init__(self,
                 config: Config):
        super().__init__()
        self.config = config
        if config.n_frames % config.channel_3d != 0:
            raise ValueError(f"n_frames must be able to n_frames % channel_3d == 0")

        if config.pretrained and not config.submission_mode:
            pretrained = True
        else:
            pretrained = False
        if config.channel_3d != 3:
            self.cnn_2d = timm.create_model(config.model_name.replace("cnn_2.5d3d_", ""), num_classes=0,
                                            pretrained=pretrained, in_chans=config.channel_3d)
        else:
            self.cnn_2d = timm.create_model(config.model_name.replace("cnn_2.5d3d_", ""), num_classes=0, pretrained=pretrained)
        if config.seq_model == "3dcnn_simple":
            self.seq_model = SimpleConv3d(config=config)
        elif config.seq_model == "3dcnn_3layers":
            self.seq_model = ThreeLayerConv3d(config=config)
        elif config.seq_model == "3dcnn_2layers":
            self.seq_model = TwoLayerConv3d(config=config)
        elif config.seq_model in ["3dcnn_slowfast", "3dcnn_slowfast_add"]:
            self.seq_model = SlowFastConv3d(config=config)
        if config.seq_model_short:
            config_ = copy.copy(config)
            config_.kernel_size_3d = (config.kernel_size_3d[0]//2, 1, 1)
            config_.stride_3d = (config.stride_3d[0]//2, 1, 1)
            self.seq_model_short = SimpleConv3d(config=config_)

        self.fc_contact = FC(config)
        self.fc_g = FC(config)
        self.fc_endzone_contact = FC(config)
        self.fc_sideline_contact = FC(config)
        self.fc_endzone_g = FC(config)
        self.fc_sideline_g = FC(config)
        if self.config.g_embedding:
            self.emb_g = nn.Embedding(2, 16)
        if self.config.feature_window > 0:
            self.fc_feature = nn.LazyLinear(self.config.feature_hidden_size)
            transformer_encoder = nn.TransformerEncoderLayer(
                d_model=self.config.feature_hidden_size,
                nhead=self.config.nhead,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                num_layers=self.config.feature_hidden_size,
                encoder_layer=transformer_encoder,
                # norm=nn.LayerNorm(self.config.num_layer_transformer),
            )
            self.ln = nn.LayerNorm(self.config.feature_hidden_size)

    def _forward_g_contact(self, model_g, model_contact, x, is_g):
        x_contact = model_contact(x)  # (bs, n_predict_frames)
        x_g = model_g(x)  # (bs, n_predict_frames)

        not_is_g = (is_g == 0)
        x = x_contact * not_is_g + x_g * is_g  # (bs, n_predict_frames)
        return x

    def _forward_sep_sideend(self, x, is_g, feature):
        bs, _, seq_len, W, H = x.shape  # C = 1
        x = x.squeeze(1)  # (bs, n_view*seq_len, W, H)
        x = x.reshape(bs*(seq_len//self.config.channel_3d), self.config.channel_3d, W, H)  # (bs*n_view*seq_len//channel_3d, channel_3d, W, H)
        x = self.cnn_2d.forward_features(x)  # (bs*n_view*seq_len//3, C, W, H)
        bs_, C_, W_, H_ = x.shape

        x_3d = x.reshape(bs*2, seq_len//(self.config.channel_3d*2), C_, W_, H_)  # (bs*n_view, seq_len//3, C, W, H)
        x = self.seq_model(x_3d)  # (bs*n_view*seq_len//3, feature)

        if self.config.feature_window > 0:
            feature = self.fc_feature(feature)  # (bs, feature_window, feature_hidden_size)
            feature = self.transformer(feature)  # (bs, feature_window, feature_hidden_size)
            feature = self.ln(feature)

            if self.config.feature_mean_dim == "feature":
                feature = feature.mean(dim=2)  # (bs, feature_window)
            elif self.config.feature_mean_dim == "window":
                feature = feature.mean(dim=1)  # (bs, feature_hidden_size)

        if self.config.seq_model == "3dcnn_slowfast":
            if self.config.pooling == "avgpool":
                x_slow = F.adaptive_avg_pool3d(x[0], 1).squeeze(4).squeeze(3).squeeze(2)
                x_fast = F.adaptive_avg_pool3d(x[1], 1).squeeze(4).squeeze(3).squeeze(2)
            elif self.config.pooling == "maxpool":
                x_slow = F.adaptive_max_pool3d(x[0], 1).squeeze(4).squeeze(3).squeeze(2)
                x_fast = F.adaptive_max_pool3d(x[1], 1).squeeze(4).squeeze(3).squeeze(2)
            x = torch.cat([x_slow, x_fast], dim=1)

        elif self.config.seq_model == "3dcnn_slowfast_add":
            if self.config.pooling == "avgpool":
                x_slow = F.adaptive_avg_pool3d(x[0], 1).squeeze(4).squeeze(3).squeeze(2)
                x_fast = F.adaptive_avg_pool3d(x[1], 1).squeeze(4).squeeze(3).squeeze(2)
            elif self.config.pooling == "maxpool":
                x_slow = F.adaptive_max_pool3d(x[0], 1).squeeze(4).squeeze(3).squeeze(2)
                x_fast = F.adaptive_max_pool3d(x[1], 1).squeeze(4).squeeze(3).squeeze(2)
            x = x_slow + x_fast
        else:
            if self.config.pooling == "avgpool":
                x = F.adaptive_avg_pool3d(x, 1).squeeze(4).squeeze(3).squeeze(2)
            elif self.config.pooling == "maxpool":
                x = F.adaptive_max_pool3d(x, 1).squeeze(4).squeeze(3).squeeze(2)
        if self.config.seq_model_short:
            center_min = (seq_len // (self.config.channel_3d * 2)) // 4
            center_max = (seq_len // (self.config.channel_3d * 2) * 3) // 4
            x_short = self.seq_model_short(x_3d[:, center_min:center_max])
            x_short = F.adaptive_avg_pool3d(x_short, 1).squeeze(4).squeeze(3).squeeze(2)
            x = torch.cat([x, x_short], dim=1)

        x_endzone = self._forward_g_contact(self.fc_endzone_g, self.fc_endzone_contact, x[::2], is_g)
        x_sideline = self._forward_g_contact(self.fc_sideline_g, self.fc_sideline_contact, x[1::2], is_g)

        if self.config.fc_sideend == "add_weight":
            weight_end = torch.sigmoid(self.fc_weight(x[::2]))
            weight_side = torch.sigmoid(self.fc_weight(x[1::2]))
            x = self.fc(x[::2] * weight_end + x[1::2] * weight_side)
        elif self.config.fc_sideend == "add":
            x = x_endzone + x_sideline
        elif self.config.fc_sideend == "concat":
            x = torch.cat([x[::2], x[1::2]], dim=1)
            if self.config.g_embedding:
                x_emb_g = self.emb_g(is_g).squeeze(1)
                x = torch.cat([x, x_emb_g], dim=1)
            if self.config.feature_window > 0:
                x = torch.cat([x, feature], dim=1)
            x = self._forward_g_contact(self.fc_g, self.fc_contact, x, is_g)
        return x, x_endzone, x_sideline

    def _forward_concat_sideend(self, x, is_g, feature):
        bs, _, seq_len, W, H = x.shape
        x = x.squeeze(1)  # (bs, n_view*seq_len, W, H)
        x = torch.cat([x[:, :seq_len//2], x[:, seq_len//2:]], dim=2) # (bs, seq_len, W*2, H)

        x = x.reshape(-1, self.config.channel_3d, W*2, H)  # (bs*n_view*seq_len//channel_3d, channel_3d, W, H)
        x = self.cnn_2d.forward_features(x)  # (bs*seq_len//3, channel_3d, W*2, H)
        bs_, C_, W_, H_ = x.shape

        x = x.reshape(bs, -1, C_, W_, H_)  # (bs*n_view, seq_len//3, C, W, H)
        x = self.seq_model(x)  # (bs*n_view*seq_len//3, feature)
        if self.config.pooling == "avgpool":
            x = F.adaptive_avg_pool3d(x, 1).squeeze(4).squeeze(3).squeeze(2)
        elif self.config.pooling == "maxpool":
            x = F.adaptive_max_pool3d(x, 1).squeeze(4).squeeze(3).squeeze(2)

        if self.config.feature_window > 0:
            feature = self.fc_feature(feature)  # (bs, feature_window, feature_hidden_size)
            feature = self.transformer(feature)  # (bs, feature_window, feature_hidden_size)
            feature = self.ln(feature)

            if self.config.feature_mean_dim == "feature":
                feature = feature.mean(dim=2)  # (bs, feature_window)
            elif self.config.feature_mean_dim == "window":
                feature = feature.mean(dim=1)  # (bs, feature_hidden_size)

        if self.config.g_embedding:
            x_emb_g = self.emb_g(is_g).squeeze(1)
            x = torch.cat([x, x_emb_g], dim=1)
        if self.config.feature_window > 0:
            x = torch.cat([x, feature], dim=1)
        x = self._forward_g_contact(self.fc_g, self.fc_contact, x, is_g)

        return x, None, None

    def forward(self, x, is_g, feature):
        if self.config.fc_sideend != "image_concat":
            return self._forward_sep_sideend(x, is_g, feature)
        else:
            return self._forward_concat_sideend(x, is_g, feature)


class Model3D(nn.Module):
    def __init__(self,
                 config: Config):
        super().__init__()
        self.config = config
        if self.config.model_name == "cnn_3d_r3d_18":
            if self.config.submission_mode:
                self.model = r3d_18()
                self.model.fc = nn.Identity()
            else:
                weights = R3D_18_Weights.DEFAULT
                self.model = r3d_18(weights=weights)
                self.model.fc = nn.Identity()
            if self.config.custom_3d:
                self.model.stem[0] = nn.Conv3d(
                    3, 64,
                    kernel_size=self.config.kernel_custom_3d,
                    stride=(1, 2, 2),
                    padding=(1, 3, 3)
                )
            if self.config.channel_6:
                self.model.stem[0] = nn.Conv3d(
                    6, 64,
                    kernel_size=self.model.stem[0].kernel_size,
                    stride=self.model.stem[0].stride,
                    padding=self.model.stem[0].padding
                )
        elif self.config.model_name == "cnn_3d_mc3_18":
            if self.config.submission_mode:
                self.model = mc3_18()
                self.model.fc = nn.Identity()
            else:
                weights = MC3_18_Weights.DEFAULT
                self.model = mc3_18(weights=weights)
                self.model.fc = nn.Identity()
        elif self.config.model_name == "cnn_3d_mvit_v2_s":
            if self.config.submission_mode:
                self.model = mvit_v2_s()
                self.model.fc = nn.Identity()
            else:
                weights = MViT_V2_S_Weights.DEFAULT
                self.model = mvit_v2_s(weights=weights)
                self.model.fc = nn.Identity()
        elif self.config.model_name == "cnn_3d_r2plus1d_18":
            if self.config.submission_mode:
                self.model = r2plus1d_18()
                self.model.fc = nn.Identity()
            else:
                weights = R2Plus1D_18_Weights.DEFAULT
                self.model = r2plus1d_18(weights=weights)
                self.model.fc = nn.Identity()

        self.fc_contact = FC(config)
        self.fc_g = FC(config)
        if config.fc_sideend != "image_concat":
            self.fc_endzone_contact = FC(config)
            self.fc_sideline_contact = FC(config)
            self.fc_endzone_g = FC(config)
            self.fc_sideline_g = FC(config)

        if self.config.fc_sideend == "add_weight":
            self.fc_weight = nn.LazyLinear(1)
        if self.config.g_embedding:
            self.emb_g = nn.Embedding(2, 16)
        if self.config.feature_window > 0:
            self.fc_feature = nn.LazyLinear(self.config.feature_hidden_size)
            transformer_encoder = nn.TransformerEncoderLayer(
                d_model=self.config.feature_hidden_size,
                nhead=self.config.nhead,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                num_layers=self.config.feature_hidden_size,
                encoder_layer=transformer_encoder,
                # norm=nn.LayerNorm(self.config.num_layer_transformer),
            )
            self.ln = nn.LayerNorm(self.config.feature_hidden_size)

    def _forward_g_contact(self, model_g, model_contact, x, is_g):
        x_contact = model_contact(x)  # (bs, n_predict_frames)
        x_g = model_g(x)  # (bs, n_predict_frames)

        not_is_g = (is_g == 0)
        x = x_contact * not_is_g + x_g * is_g  # (bs, n_predict_frames)
        return x

    def _forward_sep_sideend(self, x, is_g, feature):
        bs, C, seq_len, W, H = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (bs, n_view*seq_len, C, W, H)
        x = x.reshape(bs*2, seq_len//2, C, W, H)   # (bs*n_view, seq_len, C, W, H)
        x = x.permute(0, 2, 1, 3, 4)

        # x[0] = EndZone[0], x[1] = SideLine[0]...  x[i] = EndZone[i//2], x[i+1] = SideLine[i//2]
        x = self.model(x)  # (bs*n_view, fc)

        if self.config.feature_window > 0:
            feature = self.fc_feature(feature)  # (bs, feature_window, feature_hidden_size)
            feature = self.transformer(feature)  # (bs, feature_window, feature_hidden_size)
            feature = self.ln(feature)

            if self.config.feature_mean_dim == "feature":
                feature = feature.mean(dim=2)  # (bs, feature_window)
            elif self.config.feature_mean_dim == "window":
                feature = feature.mean(dim=1)  # (bs, feature_hidden_size)

        x_endzone = self._forward_g_contact(self.fc_endzone_g, self.fc_endzone_contact, x[::2], is_g)
        x_sideline = self._forward_g_contact(self.fc_sideline_g, self.fc_sideline_contact, x[1::2], is_g)
        if self.config.fc_sideend == "add":
            x = x[::2] + x[1::2]
            x = self.fc(x)
        elif self.config.fc_sideend == "add_weight":
            weight_end = F.sigmoid(self.fc_weight(x[::2]))
            weight_side = F.sigmoid(self.fc_weight(x[1::2]))
            if self.config.feature_window > 0:
                x = torch.cat([x, feature], dim=1)
            x = self.fc(x[::2] * weight_end + x[1::2] * weight_side)
        elif self.config.fc_sideend == "add":
            x = x_endzone + x_sideline
        elif self.config.fc_sideend == "concat":
            x = x.reshape(bs, -1)
            if self.config.g_embedding:
                x_emb_g = self.emb_g(is_g).squeeze(1)
                x = torch.cat([x, x_emb_g], dim=1)
            if self.config.feature_window > 0:
                x = torch.cat([x, feature], dim=1)
            x = self._forward_g_contact(self.fc_g, self.fc_contact, x, is_g)

        return x, x_endzone, x_sideline

    def _forward_concat_sideend(self, x, is_g, feature):
        bs, C, seq_len, W, H = x.shape
        x = torch.cat([x[:, :, :seq_len//2], x[:, :, seq_len//2:]], dim=3) # (bs, seq_len, C, W*2, H)

        # x[0] = EndZone[0], x[1] = SideLine[0]...  x[i] = EndZone[i//2], x[i+1] = SideLine[i//2]
        x = self.model(x)  # (bs*n_view, fc)

        if self.config.feature_window > 0:
            feature = self.fc_feature(feature)  # (bs, feature_window, feature_hidden_size)
            feature = self.transformer(feature)  # (bs, feature_window, feature_hidden_size)
            feature = self.ln(feature)

            if self.config.feature_mean_dim == "feature":
                feature = feature.mean(dim=2)  # (bs, feature_window)
            elif self.config.feature_mean_dim == "window":
                feature = feature.mean(dim=1)  # (bs, feature_hidden_size)

        if self.config.g_embedding:
            x_emb_g = self.emb_g(is_g).squeeze(1)
            x = torch.cat([x, x_emb_g], dim=1)
        if self.config.feature_window > 0:
            x = torch.cat([x, feature], dim=1)
        x = self._forward_g_contact(self.fc_g, self.fc_contact, x, is_g)

        return x, None, None


    def forward(self, x, is_g, feature):
        if self.config.fc_sideend != "image_concat":
            return self._forward_sep_sideend(x, is_g, feature)
        else:
            return self._forward_concat_sideend(x, is_g, feature)


def get_df_from_item(item):
    df = pd.DataFrame({
        "contact_id": item["contact_id"],
        "contact": item["contact"] == 1,
    })
    df["contact"] = df["contact"].astype(int)
    return df

def calc_best(label, pred, logger, epoch, name):
    best_th = -1
    best_score = -1
    auc = roc_auc_score(label, pred)
    logger.info(f"\nauc: {auc}")
    wandb.log({f"auc_{name}": auc})
    for th in np.arange(0, 1, 0.05):
        score = matthews_corrcoef(label, pred > th)

        logger.info(f"th={th}: score={score}")
        # logger.info(f"counfusion_matrix: \n{confusion_matrix(label, pred > th)}")
        if best_score < score:
            best_th = th
            best_score = score

    return auc, best_th, best_score


def main(config):
    try:
        seed_everything()
        output_dir = f"../../output/cnn_3d/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}_{config.exp_name}"
        config.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(__file__, output_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(f"{output_dir}/cfg.pickle", "wb") as f:
            pickle.dump(config, f)

        base_dir = config.base_dir
        logger = get_logger(output_dir)
        logger.info("start!")
        df = pd.read_feather(config.feature_dir)
        gkfold = GroupKFold(5)

        df_label = pd.read_csv("../../input/nfl-player-contact-detection/train_labels.csv")
        if config.debug:
            df_label = df_label.iloc[:300000]

        if "cnn_3d_" in config.model_name:
            model = Model3D(config=config)
        elif "cnn_2.5d3d_" in config.model_name:
            model = Model2p5DTo3D(config=config)
        elif "cnn_2.5d_" in config.model_name:
            model = Model2p5D(config=config)
        elif "cnn_2d_" in config.model_name and config.n_frames == 1:
            model = Model2D(config=config)
        elif (config.model_name == "cnn_2d1d") or ("cnn_2d_" in config.model_name and "1dcnn_" in config.seq_model):
            model = Model2DTo1D(config=config)
        elif (config.model_name == "cnn_2d3d") or ("cnn_2d_" in config.model_name and "3dcnn_" in config.seq_model):
            model = Model2DTo3D(config=config)
        elif type(config) == ConfigForGNN:
            model = NFLGraphModel(config=config)
        elif type(config) == ConfigForTransformer:
            model = TransformerModel(config=config)
        else:
            raise ValueError("モデルの指定が変です")

        df_label["game_key"] = [int(x.split("_")[0]) for x in df_label["contact_id"].values]
        for fold, (train_idx, val_idx) in enumerate(gkfold.split(df_label, groups=df_label[config.gk_key].values)):
            if fold != config.fold:
                continue
            df_label_train = df_label.iloc[train_idx]
            df_label_val = df_label.iloc[val_idx]
            if type(config) == ConfigForTransformer:
                df_feature = pd.read_feather(config.feature_dir)
                df_train = df_feature[df_feature["game_play"].isin(df_label_train["game_play"].values)]
                df_val = df_feature[df_feature["game_play"].isin(df_label_val["game_play"].values)]
            else:
                df_train = df[df[config.gk_key].isin(df_label_train[config.gk_key].values)]
                df_val = df[df[config.gk_key].isin(df_label_val[config.gk_key].values)]
            break
        df_merge = pd.merge(
            df_label[["contact_id", "contact"]],
            df[["contact_id", "contact"]].rename(columns={"contact": "pred"}),
            how="left"
        ).fillna(0).sort_values("contact", ascending=False).drop_duplicates("contact_id")
        possible_score_all = matthews_corrcoef(df_merge["contact"].values, df_merge["pred"].values == 1)
        logger.info(f"possible MCC score: {possible_score_all}")
        del df; gc.collect()

        if config.debug:
            num_workers = 0
        elif type(config) == ConfigForGNN:
            num_workers = 0
        elif "cnn_2d_" in config.model_name and config.n_frames == 1:
            num_workers = 2
        elif config.model_name == "cnn_2d1d":
            num_workers = 0
        elif type(config) == ConfigForTransformer:
            num_workers = 0
        else:
            num_workers = 12
        use_filelist = True
        if config.model_name == "cnn_2d1d":
            use_filelist = False

        if type(config) != ConfigForGNN:
            if type(config) == Config:
                train_dataset = NFLDataset(
                    df=df_train,
                    base_dir=f"{base_dir}/{config.image_path}",
                    logger=logger,
                    config=config,
                    test=False,
                    use_filelist=use_filelist
                )

                val_dataset = NFLDataset(
                    df=df_val,
                    base_dir=f"{base_dir}/{config.image_path}",
                    logger=logger,
                    config=config,
                    test=True,
                    use_filelist=use_filelist
                )
            elif type(config) == ConfigForTransformer:
                train_dataset = NFLTransformerDataset(
                    df=df_train,
                    base_dir=f"{base_dir}/{config.image_path}",
                    logger=logger,
                    config=config,
                    test=False,
                )

                val_dataset = NFLTransformerDataset(
                    df=df_val,
                    base_dir=f"{base_dir}/{config.image_path}",
                    logger=logger,
                    config=config,
                    test=True,
                )
            df_val_dataset = pd.concat([get_df_from_item(item) for item in val_dataset.items])
            df_merge = pd.merge(
                df_label_val[["contact_id", "contact"]],
                df_val_dataset[["contact_id", "contact"]].rename(columns={"contact": "pred"}),
                how="left"
            ).fillna(0).sort_values("contact", ascending=False).drop_duplicates("contact_id")
            possible_score_extracted = matthews_corrcoef(df_merge["contact"].values, df_merge["pred"].values)
            logger.info(f"possible MCC score: {possible_score_extracted}")

            if config.debug:
                train_dataset.items = train_dataset.items[:200]
                val_dataset.items = val_dataset.items[:200]
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                num_workers=num_workers
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                num_workers=num_workers
            )
        else:
            train_dataset = NFLGraphDataset(
                df=df_train,
                config=config,
            )

            val_dataset = NFLGraphDataset(
                df=df_val,
                config=config,
            )

            train_loader = GraphDataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                num_workers=num_workers,
            )

            val_loader = GraphDataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                num_workers=num_workers,
            )
            possible_score_extracted = possible_score_all

        del df_val; gc.collect()
        del df_merge, df_label, df_label_val; gc.collect()

        model = model.to(device)
        def get_params(k, params):
            if k in ["cnn_2d", "model"]:
                logger.info(f"{k}: lr={config.lr}")
                return {"params": params.parameters(), "lr": config.lr, "weight_decay": config.weight_decay}
            else:
                logger.info(f"{k}: lr={config.lr_fc}")
                return {"params": params.parameters(), "lr": config.lr_fc, "weight_decay": config.weight_decay}
        params = [get_params(k, params) for k, params in model._modules.items()]
        optimizer = torch.optim.AdamW(params)

        if config.criterion == "bcewithlogitsloss":
            criterion = nn.BCEWithLogitsLoss()
        elif config.criterion == "l1loss":
            criterion = nn.L1Loss()
        elif config.criterion == "mseloss":
            criterion = nn.MSELoss()
        elif config.criterion == "focalloss":
            criterion = FocalLoss()
        elif config.criterion == "smoothfocalloss":
            criterion = SmoothFocalLoss(smoothing=config.smooth, gamma=config.focal_gamma)
        elif config.criterion == "mccloss":
            criterion = MCCLoss()

        if config.fc_sideend == "image_concat":
            logger.info("single_view_loss: False")
            config.calc_single_view_loss = False

        if config.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=len(train_loader) * config.warmup_ratio * config.epochs,
                num_training_steps=len(train_loader) * config.epochs
            )
        if config.scheduler == "StepLR":
            scheduler = StepLR(
                optimizer=optimizer, step_size=int(len(train_loader) * config.step_size_ratio), gamma=config.gamma,
            )
        if config.scheduler == "StepLRWithWarmUp":
            num_warmup_steps = int(config.warmup_ratio * len(train_loader))
            step_size = int(len(train_loader) * config.step_size_ratio)

            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                reduce = config.gamma ** (current_step // max(1, step_size))
                return reduce
            scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

        results = []
        wandb.init(project="nfl_contact", name=config.exp_name, reinit=True)

        for k, v in config.__dict__.items():
            wandb.config.update({k: v})

        wandb.log({
            "MCC_all": possible_score_all,
            "MCC_extracted": possible_score_extracted,
        })
        wandb.config.update({"output_dir": output_dir})
        total_best_score = {}
        for epoch in range(config.epochs):
            logger.info(f"===============================")
            logger.info(f"epoch {epoch + 1}")
            logger.info(f"===============================")

            if type(config) == Config:
                train_dataset = NFLDataset(
                    df=df_train,
                    base_dir=f"{base_dir}/{config.image_path}",
                    logger=logger,
                    config=config,
                    test=False,
                    use_filelist=use_filelist
                )
                if config.debug:
                    train_dataset.items = train_dataset.items[:200]
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=True,
                    num_workers=num_workers
                )

            train_loss = train_fn(
                train_loader,
                model,
                criterion,
                optimizer,
                device,
                scheduler,
                epoch,
                config,
            )

            df_pred, valid_loss = eval_fn(
                val_loader,
                model,
                criterion,
                device,
                config
            )

            df_label = pd.read_csv("../../input/nfl-player-contact-detection/train_labels.csv")
            df_label["game_key"] = [int(x.split("_")[0]) for x in df_label["contact_id"].values]
            for fold, (train_idx, val_idx) in enumerate(gkfold.split(df_label, groups=df_label[config.gk_key].values)):
                if fold != config.fold:
                    continue
                df_label_val = df_label.iloc[val_idx]
                break

            logger.info(f"loss: train {train_loss}, val {valid_loss}")
            logger.info(f"------ MCC ------")

            if config.calc_single_view_loss:
                cols = ["score_endzone", "score_sideline", "score"]
            else:
                cols = ["score"]
            if type(config) == ConfigForTransformer:
                if len(df_label_val) != len(df_pred):
                    logger.info("df length が違う")
                df_score = df_pred.copy()
                df_score["contact"] = df_score["label"]
            else:
                df_merge = pd.merge(df_label_val, df_pred, how="left")
                df_score = df_merge.groupby(["contact_id", "contact"], as_index=False)[cols].mean()

            results.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": valid_loss,
            })

            wandb.log({
                "val_loss": valid_loss,
                "epoch": epoch,
            })

            for col in cols:
                logger.info(f"-------------- {col} ----------------")
                logger.info(f"\n[G]")
                w_df = df_score[df_score["contact_id"].str.contains("G")]
                label_g = w_df["contact"].values
                pred_g = w_df[col].fillna(0).values
                _, best_th_g, best_score_g = calc_best(label_g, pred_g, logger, epoch, name="g")

                logger.info(f"\n[Contact]")
                w_df = df_score[~df_score["contact_id"].str.contains("G")]
                label_contact = w_df["contact"].values
                pred_contact = w_df[col].fillna(0).values
                _, best_th_contact, best_score_contact = calc_best(label_contact, pred_contact, logger, epoch, name="contact")

                best_score = matthews_corrcoef(
                    np.concatenate([label_g, label_contact]),
                    np.concatenate([pred_g > best_th_g, pred_contact > best_th_contact]),
                )

                logger.info(f"***************** epoch {epoch} *****************")
                logger.info(f"best: {best_score}")
                logger.info(f"******************************************")
                if type(config) == ConfigForTransformer:
                    df_pred.to_csv(f"{output_dir}/pred_{epoch}.csv", index=False)
                else:
                    pd.merge(df_label_val, df_pred, how="left").to_csv(f"{output_dir}/pred_{epoch}.csv", index=False)
                if not config.debug:
                    torch.save(model.state_dict(), f"{output_dir}/epoch{epoch}.pth")

                results.append({
                    col: best_score,
                })
                wandb.log({
                    f"{col}": best_score,
                    f"{col}_g": best_score_g,
                    f"{col}_contact": best_score_contact,
                    "epoch": epoch,
                })
                if col not in total_best_score or total_best_score[col] < best_score:
                    total_best_score[col] = best_score
                    wandb.log({
                        f"best_{col}": total_best_score[col],
                        f"best_{col}_g": best_score_g,
                        f"best_{col}_contact": best_score_contact,
                        "epoch": epoch,
                    })

                    if col == "score":
                        logger.info("save best!")
                        if type(config) == ConfigForTransformer:
                            df_pred.to_csv(f"{output_dir}/pred_best.csv", index=False)
                        else:
                            pd.merge(df_label_val, df_pred, how="left").to_csv(f"{output_dir}/pred_best.csv",
                                                                               index=False)
                        if not config.debug:
                            torch.save(model.state_dict(), f"{output_dir}/best.pth")

            pd.DataFrame(results).to_csv(f"{output_dir}/results.csv", index=False)

        if "cnn_2d_" in config.model_name and config.save_feature:
            logger.info("save feature")
            save_feature(model, device, config)

        wandb.finish()
    except Exception as e:
        print(e)
        raise


if __name__ == "__main__":
    v = 44
    n_predict_frames = 5

    exp_name = f"2.5d3d_v{v}_smoothbce(0.1, 0.9)_wd0.1_gd0.2_pred_all_frames"
    config = Config(exp_name=exp_name, seq_model="3dcnn_simple",
                    model_name=f"cnn_2.5d3d_legacy_seresnet18", epochs=2, step=3,
                    step_size_ratio=1.5,
                    fc="2layers",
                    hidden_size_3d=512,
                    weight_decay=0.1,
                    warmup_ratio=0.2,
                    gradient_clipping=0.2,
                    interpolate_image=False,
                    image_path=f"images_128x96_v{v}",
                    criterion="bcewithlogitsloss",
                    soft_label_range=(0.1, 0.9),
                    lr_fc=1e-3,
                    lr=1e-4,
                    batch_size=24,
                    gk_key="game_play",
                    calc_single_view_loss_weight=1,
                    predict_all_frames=True,
                    max_frames=297,
                    n_predict_frames=297,
                    n_frames=297,
                    )
    main(config)

    exp_name = f"3d_v{v}_n_predict_frames{n_predict_frames}_pred_all_frames"
    config = Config(exp_name=exp_name, seq_model="flatten",
                    step=3, epochs=2,
                    step_size_ratio=1,
                    negative_sample_ratio_g=1,
                    negative_sample_ratio_close=1,
                    negative_sample_ratio_far=1,
                    model_name="cnn_3d_r3d_18",
                    lr_fc=1e-3,
                    lr=1e-4,
                    batch_size=24,
                    interpolate_image=False,
                    criterion="smoothfocalloss",
                    smooth=0.2,
                    transforms_train=A.Compose([
                        A.CenterCrop(96, 96, p=1.0),
                        A.HorizontalFlip(p=0.5),
                    ]),
                    transforms_eval=A.Compose([
                        A.CenterCrop(96, 96, p=1.0),
                    ]),
                    gk_key="game_play",
                    pooling="maxpool",
                    image_path=f"images_128x96_v{v}",
                    calc_single_view_loss_weight=1,
                    predict_all_frames=True,
                    max_frames=297,
                    n_predict_frames=297,
                    n_frames=297,
                    )
    main(config)

