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
    from torchvision.models.video import r3d_18, R3D_18_Weights
    import mlflow
except Exception as e:
    print(e)
    from torchvision.models.video import r3d_18
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
import shutil
import torch.nn.functional as F
import pickle
from typing import Tuple

torch.backends.cudnn.benchmark = True

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
class Config:
    exp_name: str
    debug: bool = True

    epochs: int = 1
    if debug:
        epochs: int = 1

    lr: float = 1e-4
    weight_decay: float = 0.001
    n_frames: int = 31
    n_predict_frames: int = 1

    if n_frames % 2 == 0 or n_predict_frames % 2 == 0:
        raise ValueError
    step: int = 4
    extention: str = ".npy"
    negative_sample_ratio_close: float = 0.2
    negative_sample_ratio_far: float = 0.2
    base_dir: str = "../../output/preprocess/images"
    data_dir: str = f"../../output/preprocess/master_data_v2"
    image_path: str = "images_128x96"
    img_shape: Tuple[int, int] = (96, 128)
    gradient_clipping: float = 1
    exist_image_threshold: float = 0.1
    data_per_epoch: float = 1
    grayscale: bool = False
    batch_size: int = 32
    use_data_step: int = 1

    num_training_steps: int = 100000

    # 2d_cnn
    model_name: str = "cnn_3d_r3d_18"
    seq_model: str = "flatten"
    dropout_seq: float = 0.2
    activation: nn.Module = nn.Identity

    kernel_size_conv1d: int = 3
    stride_conv1d: int = 1
    hidden_size_1d: int = 32

    submission_mode: bool = False

    distance_threshold: float = 1.75
    sep_is_g: bool = False
    calc_single_view_loss: bool = True

    calc_single_view_loss_weight: float = 0.5


class NFLDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 base_dir: str,
                 logger: Logger,
                 config: Config,
                 test: bool,
                 image_dict: dict = None):
        self.base_dir = base_dir
        self.config = config
        self.test = test
        self.exist_files = set()
        self.image_dict = image_dict
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

    def _exist_file(self,
                    game_play,
                    view,
                    id_1,
                    id_2,
                    frame):
        if self.image_dict is not None:
            # for submission
            return self._get_key(game_play, view, id_1, id_2, frame) in self.image_dict
        else:
            # for local training
            return os.path.isfile(self._get_key(game_play, view, id_1, id_2, frame))

    def _exist_files(self,
                     game_play: str,
                     id_1: str,
                     id_2: str,
                     frames: List[int]):
        count = 0

        for view in ["Sideline", "Endzone"]:
            base_dir = self._get_base_dir(game_play, view, id_1, id_2)
            for frame in frames:
                if self._exist_file(game_play, view, id_1, id_2, frame):
                    count += 1
                if count > self.config.n_frames * 2 * self.config.exist_image_threshold:
                    return True
        return False

    def _exist_center_files(self,
                            game_play: str,
                            id_1: str,
                            id_2: str,
                            frames: List[int]):
        # 予測対象のframeはSideline / Endline どっちかにはファイルがいてほしい
        for frame in frames:
            ret = False
            for view in ["Sideline", "Endzone"]:
                if self._exist_file(game_play, view, id_1, id_2, frame):
                    ret = True
            if not ret:
                return False
        return True

    def _get_item_information(self, df: pd.DataFrame, logger: Logger):
        self.items = []
        logger.info("_get_item_information start")

        failed_count = 0
        is_g_count = 0
        np.random.seed(0)

        contacts_all = []
        for key, w_df in tqdm.tqdm(
            df.drop_duplicates(
                ["game_play", "nfl_player_id_1", "nfl_player_id_2", "frame"]
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
            steps = w_df["step"].fillna(0).values

            for i in range(len(w_df)):
                if not self.test and i % self.config.use_data_step != 0:
                    continue

                min_frame_idx = frames[i] - self.config.n_frames // 2 * self.config.step
                max_frame_idx = frames[i] + self.config.n_frames // 2 * self.config.step + 1  # frames数は偶数にする(conv1dメンドイので)

                frame_indice = np.arange(min_frame_idx, max_frame_idx, self.config.step)
                window = self.config.n_predict_frames // 2

                if i - window < 0 or i + window + 1 > len(w_df):
                    continue
                predict_frames_indice = np.arange(
                    i - window,
                    i + window + 1,
                )

                expected_steps = np.arange(
                    steps[i] - window,
                    steps[i] + window + 1
                )

                # contactsが途中で切れているデータは除去する (ex. [5, 10, 11] centerが10だけど, 6~9が抜けている)
                if (steps[predict_frames_indice] != expected_steps).sum() > 0:
                    failed_count += 1
                    continue

                assert len(predict_frames_indice) == self.config.n_predict_frames
                if distances[i] > self.config.distance_threshold:
                    continue

                if not self.test and contacts[predict_frames_indice].sum() == 0:
                    # down sampling (only negative)
                    if distances[i] < 0.75 and np.random.random() > self.config.negative_sample_ratio_close and id_2 != "G":
                        continue
                    if distances[i] >= 0.75 and np.random.random() > self.config.negative_sample_ratio_far:
                        continue
                    if id_2 == "G" and np.random.random() > self.config.negative_sample_ratio_far:
                        continue

                if not self._exist_files(game_play=game_play, id_1=id_1, id_2=id_2, frames=frame_indice):
                    failed_count += 1
                    continue
                # if not self._exist_center_files(game_play=game_play, id_1=id_1, id_2=id_2, frames=frames[predict_frames_indice]):
                #     failed_count += 1
                #     continue

                contacts_all.append(contacts[predict_frames_indice].mean())
                self.items.append({
                    "contact_id": contact_ids[predict_frames_indice],
                    "game_play": game_play,
                    "id_1": id_1,
                    "id_2": id_2,
                    "contact": contacts[predict_frames_indice],
                    "frames": frame_indice,
                    "is_g": int(id_2 == "G"),
                })
                is_g_count += int(id_2 == "G")

        logger.info(f"finished. extracted={len(self.items)} (total={len(df)}, is_g={is_g_count}, failed={failed_count})")
        logger.info(f"contacts_distribution: \n {pd.Series(contacts_all).value_counts()}")

    def __len__(self):
        return len(self.items)

    def imread(self, game_play, view, id_1, id_2, frame):
        key = self._get_key(game_play, view, id_1, id_2, frame)
        if self.image_dict is None:
            if os.path.isfile(key):
                img = np.load(key)
            else:
                return None
        else:
            if key in self.image_dict:
                img = self.image_dict[key]
            else:
                return None
        if self.config.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        return img

    def __getitem__(self, index):
        item = self.items[index]  # {movie_id}/{start_time}

        game_play = item["game_play"]
        contact_id = item["contact_id"]
        frames = item["frames"]
        labels = item["contact"]
        id_1 = item["id_1"]
        id_2 = item["id_2"]
        is_g = item["is_g"]

        imgs_all = []
        for view in ["Endzone", "Sideline"]:
            imgs = [self.imread(game_play, view, id_1, id_2, frame) for frame in frames]

            # 外挿
            first_img_idx = [i for i, img in enumerate(imgs) if img is not None]

            if len(first_img_idx) == 0:
                imgs_all.extend([np.zeros((self.config.img_shape[0], self.config.img_shape[1], 3)) for _ in range(len(imgs))])
                continue
            first_img_idx = first_img_idx[0]

            for idx in range(first_img_idx):
                imgs[idx] = imgs[first_img_idx].copy()
            for i in range(len(imgs) - 1):
                if imgs[i+1] is None:
                    imgs[i+1] = imgs[i].copy()
            imgs_all.extend(imgs)
        frames = np.stack(imgs_all, axis=0).transpose(3, 0, 1, 2)  # shape = (C, n_frame*n_view, H, W)

        return contact_id.tolist(), torch.Tensor(frames), torch.Tensor(labels), torch.Tensor([is_g])

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


def log_loss(loss_history, loss_meter, loss, batch_size, name, mode):
    loss_history.append(loss)
    loss_history = loss_history[-100:]
    loss_meter.update(np.mean(loss_history), batch_size)
    mlflow.log_metric(f"{mode}_{name}", loss_meter.avg)
    mlflow.log_metric(f"{mode}_{name}_snap", np.mean(loss_history))


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
    loss_concat_100 = []
    loss_endzone_100 = []
    loss_sideline_100 = []
    for bi, data in tk0:
        count += 1
        batch_size = len(data)

        x = data[1].to(device)
        label = data[2].to(device)
        is_g = data[3].to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            pred, pred_endzone, pred_sideline = model(x, is_g)
            loss_concat = criterion(pred.flatten(), label.flatten())
            if config.calc_single_view_loss:
                loss_endzone = criterion(pred_endzone.flatten(), label.flatten())
                loss_sideline = criterion(pred_sideline.flatten(), label.flatten())
                loss = loss_concat + (loss_endzone + loss_sideline) * config.calc_single_view_loss_weight
            else:
                loss = loss_concat
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
        scheduler.step()
        scaler.step(optimizer)
        scaler.update()

        log_loss(loss_history=loss_100, loss_meter=loss_meter, loss=loss.detach().item(),
                 batch_size=batch_size, name="loss", mode="train")
        if config.calc_single_view_loss:
            log_loss(loss_history=loss_concat_100, loss_meter=loss_concat_meter, loss=loss_concat.detach().item(),
                     batch_size=batch_size, name="loss_concat", mode="train")
            log_loss(loss_history=loss_endzone_100, loss_meter=loss_endzone_meter, loss=loss_endzone.detach().item(),
                     batch_size=batch_size, name="loss_endzone", mode="train")
            log_loss(loss_history=loss_sideline_100, loss_meter=loss_sideline_meter, loss=loss_sideline.detach().item(),
                     batch_size=batch_size, name="loss_sideline", mode="train")

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


def eval_fn(data_loader, model, criterion, device):
    loss_score = AverageMeter()

    model.eval()
    tk0 = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    preds = []
    contact_ids = []
    labels = []

    with torch.no_grad():
        for bi, data in tk0:
            batch_size = len(data)

            contact_id = data[0]
            x = data[1].to(device)
            label = data[2].to(device)
            is_g = data[3].to(device)

            with torch.cuda.amp.autocast():
                pred, pred_endzone, pred_sideline = model(x, is_g)
                loss = criterion(pred.flatten(), label.flatten())

            loss_score.update(loss.detach().item(), batch_size)
            tk0.set_postfix(Eval_Loss=loss_score.avg)

            contact_ids.extend(np.array(contact_id).flatten())
            preds.extend(torch.sigmoid(pred.flatten()).detach().cpu().numpy())
            labels.extend(label.flatten().detach().cpu().numpy())

            del x, label, pred

    preds = np.array(preds).astype(np.float16)
    labels = np.array(labels).astype(np.float16)

    idx = np.arange(config.n_predict_frames) - config.n_predict_frames // 2
    indices = np.tile(idx, len(preds) // config.n_predict_frames)

    df_ret = pd.DataFrame({
        "contact_id": contact_ids,
        "score": preds,
        "label": labels,
        "index": indices,
    })
    return df_ret, loss_score.avg


class SimpleConv3d(nn.Module):
    def __init__(self,
                 hidden_size_3d,
                 config: Config):
        super(SimpleConv3d, self).__init__()
        hid = hidden_size_3d // 16 + 1
        self.fc = nn.Conv3d(hidden_size_3d, hid, (13, 2, 2), bias=False, stride=(2, 1, 1), padding=(0, 1, 1))
        self.bn = nn.BatchNorm3d(hid)
        self.do = nn.Dropout(0.2)
        self.activation = config.activation()
        self.num_features = hid

    def forward(self, x):
        x = self.activation(self.do(F.relu(self.bn(self.fc(x)))))
        return x


class ThreeLayerConv3d(nn.Module):
    def __init__(self,
                 hidden_size_3d,
                 config: Config):
        super(ThreeLayerConv3d, self).__init__()
        hid = hidden_size_3d // 16 + 1
        self.fc = nn.Conv3d(hidden_size_3d, hid, (5, 2, 2), bias=False, stride=(2, 1, 1), padding=(0, 1, 1))
        self.bn = nn.BatchNorm3d(hid)
        self.do = nn.Dropout(0.2)
        self.fc2 = nn.Conv3d(hid, hid, (3, 2, 2), bias=False, stride=(2, 1, 1), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(hid)
        self.do2 = nn.Dropout(0.25)
        self.fc3 = nn.Conv3d(hid, hid, (3, 1, 1), bias=False, stride=(2, 1, 1))
        self.bn3 = nn.BatchNorm3d(hid)
        self.do3 = nn.Dropout(0.25)
        self.num_features = hid
        self.activation = config.activation()

    def forward(self, x):
        x = self.activation(self.do(F.relu(self.bn(self.fc(x)))))
        x = self.activation(self.do2(F.relu(self.bn2(self.fc2(x)))))
        x = self.activation(self.do3(F.relu(self.bn3(self.fc3(x)))))
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
                out_channels=self.n_frames,
                kernel_size=config.kernel_size_conv1d,
                stride=config.stride_conv1d,
                bias=False
            )
        elif config.seq_model == "1dcnn_3layers":
            self.model = ThreeLayerConv1DUnit(config)
        elif config.seq_model =="3dcnn_simple":
            self.model = SimpleConv3d(hidden_size_3d=hidden_size, config=config)
        elif config.seq_model == "3dcnn_3layers":
            self.model = ThreeLayerConv3d(hidden_size_3d=hidden_size, config=config)
        else:
            raise ValueError(config.seq_model)

    def forward(self, x):
        x = self.model(x)
        if self.config.seq_model in ["lstm", "gru"]:
            return x[0]
        else:
            return x


class Model(nn.Module):
    def __init__(self,
                 config: Config):
        super().__init__()
        self.config = config
        self.cnn_2d = timm.create_model(config.model_name, num_classes=0, pretrained=True)
        self.seq_model = SequenceModel(
            hidden_size=self.cnn_2d.num_features,
            config=config
        )
        self.fc = nn.LazyLinear(config.n_predict_frames)

    def forward(self, x):
        bs, C, seq_len, W, H = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (bs, seq_len*n_view, C, W, H)
        x = x.reshape(bs*seq_len, C, W, H)  # (bs*seq_len*n_view, C, W, H)
        if "3dcnn" not in self.config.seq_model:
            x = self.cnn_2d(x)  # (bs*seq_len*n_view, features)
            x = x.reshape(bs, seq_len, -1)  # (bs, n_view*seq_len, features)

            x = self.seq_model(x)  # (bs, n_view*seq_len, features)
            x = x.mean(dim=2)  # (bs, n_view*seq_len)
            x = self.fc(x)  # (bs, n_view*seq_len, n_predict_frames)

        else:
            x = self.cnn_2d.forward_features(x)  # (bs*n_view*seq_len, C', W', H')
            _, C_, W_, H_ = x.shape
            x = x.reshape(bs*2, seq_len//2, C_, W_, H_)  # (bs*n_view, seq_len, C_, W_, H_)
            x = x.permute(0, 2, 1, 3, 4)  # (bs*n_view, C_, seq_len, W_, H_)
            x = self.seq_model(x)  # (bs*2, seq_len, features)
            x = F.adaptive_avg_pool3d(x, 1).squeeze(4).squeeze(3).squeeze(2)  # (bs*2, C)
            x = x.reshape(bs, 2, -1)  # (bs, 2, C)
            x = x.mean(dim=2)
            x = self.fc(x)  # (bs, seq_len, n_predict_frames)
        return x


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
        if self.config.sep_is_g:
            self.fc_contact = nn.LazyLinear(config.n_predict_frames)
            self.fc_g = nn.LazyLinear(config.n_predict_frames)

            if self.config.calc_single_view_loss:
                raise NotImplementedError
        else:
            if self.config.calc_single_view_loss:
                self.fc_endzone = nn.LazyLinear(config.n_predict_frames)
                self.fc_sideline = nn.LazyLinear(config.n_predict_frames)
            self.fc = nn.LazyLinear(config.n_predict_frames)

    def forward(self, x, is_g):
        bs, C, seq_len, W, H = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (bs, n_view*seq_len, C, W, H)
        x = x.reshape(bs*2, seq_len//2, C, W, H)   # (bs*n_view, seq_len, C, W, H)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)  # (bs*n_view, fc)
        x = x.reshape(bs, 2, -1)  # (bs, 2, -1)

        x_endzone = None
        x_sideline = None
        if not self.config.sep_is_g:
            if self.config.calc_single_view_loss:
                x_endzone = x[:, 0].reshape(bs, -1)
                x_sideline = x[:, 1].reshape(bs, -1)
                x = x.reshape(bs, -1)

                x_endzone = self.fc_endzone(x_endzone)
                x_sideline = self.fc_sideline(x_sideline)
                x = self.fc(x)

            else:
                x = x.reshape(bs, -1)
                x = self.fc(x)
        else:
            x = x.reshape(bs, -1)
            x_contact = self.fc_contact(x)  # (bs, n_predict_frames)
            x_g = self.fc_g(x)  # (bs, n_predict_frames)

            not_is_g = (is_g == 0)
            x = x_contact * not_is_g + x_g * is_g  # (bs, n_predict_frames)

        return x, x_endzone, x_sideline

def get_df_from_item(item):
    df = pd.DataFrame({
        "contact_id": item["contact_id"],
        "contact": item["contact"] == 1,
    })
    df["contact"] = df["contact"].astype(int)
    return df

def main(config):
    output_dir = f"../../output/cnn_3d/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}_{config.exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(__file__, output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(f"{output_dir}/cfg.pickle", "wb") as f:
        pickle.dump(config, f)

    base_dir = config.base_dir
    df = pd.read_feather(f"{config.data_dir}/gps.feather")
    logger = get_logger(output_dir)
    logger.info("start!")
    gkfold = GroupKFold(5)

    df_label = pd.read_csv("../../input/nfl-player-contact-detection/train_labels.csv")
    if config.debug:
        df_label = df_label.iloc[:150000]

    if "cnn_3d" in config.model_name:
        model = Model3D(config=config)
    else:
        if config.sep_is_g:
            raise NotImplementedError
        if config.calc_single_view_loss:
            raise NotImplementedError
        model = Model(config=config)

    for train_idx, val_idx in gkfold.split(df_label, groups=df_label["game_play"].values):
        df_label_train = df_label.iloc[train_idx]
        df_label_val = df_label.iloc[val_idx]
        df_train = df[df["game_play"].isin(df_label_train["game_play"].values)]
        df_val = df[df["game_play"].isin(df_label_val["game_play"].values)]
        break
    df_merge = pd.merge(
        df_label[["contact_id", "contact"]],
        df[["contact_id", "contact"]].rename(columns={"contact": "pred"}),
        how="left"
    ).fillna(0).sort_values("contact", ascending=False).drop_duplicates("contact_id")
    possible_score_all = matthews_corrcoef(df_merge["contact"].values, df_merge["pred"].values == 1)
    logger.info(f"possible MCC score: {possible_score_all}")

    train_dataset = NFLDataset(
        df=df_train,
        base_dir=f"{base_dir}/{config.image_path}",
        logger=logger,
        config=config,
        test=False
    )

    val_dataset = NFLDataset(
        df=df_val,
        base_dir=f"{base_dir}/{config.image_path}",
        logger=logger,
        config=config,
        test=True
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
        num_workers=8
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=8
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=50, num_training_steps=config.num_training_steps
    )

    results = []
    mlflow.set_tracking_uri('../../mlruns/')

    try:
        with mlflow.start_run(run_name=config.exp_name):
            for k, v in config.__dict__.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("MCC_all", possible_score_all)
            mlflow.log_metric("MCC_extracted", possible_score_extracted)
            mlflow.log_param("output_dir", output_dir)
            total_best_score = -1
            for epoch in range(config.epochs):
                logger.info(f"===============================")
                logger.info(f"epoch {epoch + 1}")
                logger.info(f"===============================")
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
                    device
                )
                df_merge = pd.merge(df_label_val, df_pred, how="left")
                df_merge["score"] = df_merge["score"].fillna(0)

                best_th = -1
                best_score = -1
                best_func = None
                logger.info(f"loss: train {train_loss}, val {valid_loss}")
                logger.info(f"------ MCC ------")
                for func in [np.mean, np.max, np.min]:
                    df_score = df_merge.groupby(["contact_id", "contact"], as_index=False)["score"].apply(func)

                    auc = roc_auc_score(df_score["contact"].values, df_score["score"].values)
                    logger.info(f"\nfunc={func} auc: {auc}")

                    label = df_score["contact"].values
                    pred = df_score["score"].values
                    for th in np.arange(0, 1, 0.05):
                        score = matthews_corrcoef(label, pred > th)

                        logger.info(f"th={th} func={func}: score={score}")
                        logger.info(f"counfusion_matrix: \n{confusion_matrix(label, pred > th)}")
                        if best_score < score:
                            best_th = th
                            best_score = score
                            best_func = func

                logger.info(f"***************** epoch {epoch} *****************")
                logger.info(f"best: {best_score} (th={best_th}, func={best_func})")
                logger.info(f"******************************************")
                df_merge.to_csv(f"{output_dir}/pred_{epoch}.csv", index=False)
                torch.save(model.state_dict(), f"{output_dir}/epoch{epoch}.pth")

                results.append({
                    "epoch": epoch,
                    "score": best_score,
                    "train_loss": train_loss,
                    "val_loss": valid_loss,
                    "th": best_th,
                    "func": best_func
                })
                mlflow.log_metric("val_loss", valid_loss)
                mlflow.log_metric("score", best_score)

                if total_best_score < best_score:
                    total_best_score = best_score
                    mlflow.log_metric("best_score", total_best_score)
                    mlflow.log_metric("threshold", best_th)

                pd.DataFrame(results).to_csv(f"{output_dir}/results.csv", index=False)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # exp_name = f"3d_sep_g_contact"
    # config = Config(exp_name=exp_name, n_frames=31, seq_model="flatten",
    #                 gradient_clipping=1, model_name="cnn_3d_r3d_18",
    #                 epochs=2, sep_is_g=True)
    # main(config)
    #
    # for calc_single_view_loss_weight in [0.5, 1]:
    #     exp_name = f"3d_calc_view_loss_contact_weight{calc_single_view_loss_weight}"
    #     config = Config(exp_name=exp_name, n_frames=31, seq_model="flatten",
    #                     gradient_clipping=1, model_name="cnn_3d_r3d_18",
    #                     calc_single_view_loss_weight=calc_single_view_loss_weight,
    #                     epochs=2, calc_single_view_loss=True)
    #     main(config)
    #
    # for use_data_step in [2]:
    #     exp_name = f"3d_predict_frames3_datastep{use_data_step}"
    #     config = Config(exp_name=exp_name, n_frames=31, seq_model="flatten",
    #                     gradient_clipping=1, model_name="cnn_3d_r3d_18",
    #                     n_predict_frames=3,
    #                     use_data_step=use_data_step,
    #                     epochs=2)
    #     main(config)
    #
    # for calc_single_view_loss_weight in [0.25]:
    #     exp_name = f"3d_calc_view_loss_contact_weight{calc_single_view_loss_weight}"
    #     config = Config(exp_name=exp_name, n_frames=31, seq_model="flatten",
    #                     gradient_clipping=1, model_name="cnn_3d_r3d_18",
    #                     calc_single_view_loss_weight=calc_single_view_loss_weight,
    #                     epochs=2, calc_single_view_loss=True)
    #     main(config)

    exp_name = f"3d_predict_frames3_datastep1"
    config = Config(exp_name=exp_name, n_frames=31, seq_model="flatten",
                    gradient_clipping=1, model_name="cnn_3d_r3d_18",
                    n_predict_frames=3,
                    epochs=2)
    main(config)

    for sample_ratio in [0.05, 0.1]:
        exp_name = f"3d_neg_far{sample_ratio}_close{sample_ratio}"
        config = Config(exp_name=exp_name, n_frames=31, seq_model="flatten",
                        negative_sample_ratio_close=sample_ratio,
                        negative_sample_ratio_far=sample_ratio,
                        gradient_clipping=1, model_name="cnn_3d_r3d_18",
                        epochs=2)
        main(config)

    exp_name = f"3d_num_training_step10000"
    config = Config(exp_name=exp_name, n_frames=31, seq_model="flatten",
                    num_training_steps=10000,
                    gradient_clipping=1, model_name="cnn_3d_r3d_18",
                    epochs=2)
    main(config)
