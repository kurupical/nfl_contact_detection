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
from torchvision.models.video import r3d_18, R3D_18_Weights, r2plus1d_18, R2Plus1D_18_Weights, MViT_V2_S_Weights, mvit_v2_s
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
import shutil
import mlflow
import torch.nn.functional as F

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
    debug: bool = False

    if debug:
        epochs: int = 1
    else:
        epochs: int = 1

    lr: float = 1e-4
    n_frames: int = 31
    n_predict_frames: int = 1

    if n_frames % 2 == 0 or n_predict_frames % 2 == 0:
        raise ValueError
    step: int = 2
    extention: str = ".jpg"
    negative_sample_ratio: float = 0.2
    base_dir: str = "../../notebook/20221214"
    data_dir: str = f"../../notebook/20221214/data_v2"
    image_path: str = "images_96x96_v2"

    # 2d_cnn
    model_name_2d: str = "tf_efficientnet_b0_ns"
    seq_model: str = "1dcnn"
    dropout_seq: float = 0.2
    activation: nn.Module = nn.Identity

    kernel_size_conv1d: int = 3
    stride_conv1d: int = 1
    hidden_size_1d: int = 32



class NFLDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 base_dir: str,
                 logger: Logger,
                 config: Config,
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

    def _exist_files(self,
                     game_play: str,
                     view: str,
                     id_1: str,
                     id_2: str,
                     frames: List[int]):
        base_dir = self._get_base_dir(game_play, view, id_1, id_2)
        for frame in frames:
            fname = f"{base_dir}_{frame}{self.config.extention}"
            if not os.path.isfile(fname):
                return False
        return True

    def _get_item_information(self, df: pd.DataFrame, logger: Logger):
        self.items = []
        logger.info("_get_item_information start")

        failed_count = 0
        np.random.seed(0)

        for key, w_df in tqdm.tqdm(df.groupby(["game_play", "view", "nfl_player_id_1", "nfl_player_id_2"])):
            game_play = key[0]
            view = key[1]
            id_1 = key[2]
            id_2 = key[3]

            contact_ids = w_df["contact_id"].values
            frames = w_df["frame"].values
            contacts = w_df["contact"].values
            distances = w_df["distance"].values
            ranks = w_df["rank"].values

            rank_1_indices = np.where(ranks == 1)[0]

            for idx, i in enumerate(rank_1_indices):
                min_frame_idx = i - self.config.n_frames // 2 * self.config.step
                max_frame_idx = i + self.config.n_frames // 2 * self.config.step + 1  # frames数は偶数にする(conv1dメンドイので)
                if min_frame_idx < 0 or max_frame_idx >= len(w_df):
                    continue
                indice = np.arange(min_frame_idx, max_frame_idx, self.config.step)
                window = self.config.n_predict_frames // 2
                predict_frames_indice = np.arange(
                    idx - window,
                    idx + window + 1,
                )
                predict_frames_indice = rank_1_indices[predict_frames_indice]
                assert len(predict_frames_indice) == self.config.n_predict_frames

                if distances[predict_frames_indice].min() > 200:
                    continue
                if contacts[predict_frames_indice].sum() == 0 and np.random.random() > self.config.negative_sample_ratio and not self.test:
                    continue

                if not self._exist_files(
                    game_play=game_play,
                    view=view,
                    id_1=id_1,
                    id_2=id_2,
                    frames=frames[indice]
                ):
                    failed_count += 1
                    continue
                self.items.append({
                    "contact_id": contact_ids[predict_frames_indice],
                    "game_play": game_play,
                    "view": view,
                    "id_1": id_1,
                    "id_2": id_2,
                    "contact": contacts[predict_frames_indice],
                    "frames": frames[indice]
                })

        logger.info(f"finished. extracted={len(self.items)} (total={len(df)}, failed: {failed_count})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]  # {movie_id}/{start_time}

        contact_id = item["contact_id"]
        frames = item["frames"]
        labels = item["contact"]
        base_dir = self._get_base_dir(item["game_play"],
                                      item["view"],
                                      item["id_1"],
                                      item["id_2"])
        if self.config.extention == ".jpg":
            frames = np.stack([
                cv2.imread(f"{base_dir}_{frame}.jpg") for frame in frames
            ], axis=0).transpose(3, 0, 1, 2)  # shape = (C, n_frame, H, W)

        return contact_id.tolist(), torch.Tensor(frames), torch.Tensor(labels)

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
    loss_score = AverageMeter()

    tk0 = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))

    scaler = torch.cuda.amp.GradScaler()
    for bi, data in tk0:
        batch_size = len(data)

        x = data[1].to(device)
        label = data[2].to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            pred = model(x)
            loss = criterion(pred.flatten(), label.flatten())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scheduler.step()
        scaler.step(optimizer)
        scaler.update()

        loss_score.update(loss.detach().item(), batch_size)
        mlflow.log_metric("train_loss", loss_score.avg)

        tk0.set_postfix(Loss=loss_score.avg,
                        Epoch=epoch,
                        LR=optimizer.param_groups[0]['lr'])

    return loss_score.avg


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

            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = criterion(pred.flatten(), label.flatten())

            loss_score.update(loss.detach().item(), batch_size)
            tk0.set_postfix(Eval_Loss=loss_score.avg)

            contact_ids.extend(np.array(contact_id).flatten())
            preds.extend(torch.sigmoid(pred.flatten()).detach().cpu().numpy())
            labels.extend(label.flatten().detach().cpu().numpy())

            del x, label, pred

    preds = np.array(preds).astype(np.float16)
    labels = np.array(labels).astype(np.float16)

    df_ret = pd.DataFrame({
        "contact_id": contact_ids,
        "score": preds,
        "label": labels
    })
    return df_ret, loss_score.avg


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
        x = self.activation(self.pl(self.do(F.relu(self.bn(self.fc(x))))))
        x = self.activation(self.do2(F.relu(self.bn2(self.fc2(x)))))
        x = self.activation(self.do3(F.relu(self.bn3(self.fc3(x)))))
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
            self.model = nn.Conv1d(
                in_channels=self.n_frames,
                out_channels=self.n_frames,
                kernel_size=config.kernel_size_conv1d,
                stride=config.stride_conv1d,
                bias=False
            )
        elif config.seq_model == "1dcnn_3layers":
            self.model = ThreeLayerConv1DUnit(config)
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
        self.cnn_2d = timm.create_model(config.model_name_2d, num_classes=0, pretrained=True)
        self.seq_model = SequenceModel(
            hidden_size=self.cnn_2d.num_features,
            config=config
        )
        self.fc = nn.LazyLinear(config.n_predict_frames)

    def forward(self, x):
        bs, C, seq_len, W, H = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (bs, seq_len, C, W, H)
        x = x.reshape(bs*seq_len, C, W, H)  # (bs*seq_len, C, W, H)
        x = self.cnn_2d(x)  # (bs*seq_len, features)
        x = x.reshape(bs, seq_len, -1)  # (bs, seq_len, features)

        x = self.seq_model(x)  # (bs, seq_len, features)
        x = x.mean(dim=2)  # (bs, seq_len)
        x = self.fc(x)  # (bs, seq_len, n_predict_frames)
        return x


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

    base_dir = config.base_dir
    df = pd.read_feather(f"{config.data_dir}/gps.feather")
    logger = get_logger(output_dir)
    logger.info("start!")
    gkfold = GroupKFold(5)

    df_label = pd.read_csv("../../input/nfl-player-contact-detection/train_labels.csv")
    if config.debug:
        df_label = df_label.iloc[:150000]

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
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=8
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=8
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=50, num_training_steps=100000)

    results = []
    mlflow.set_tracking_uri('../../mlruns/')

    try:
        with mlflow.start_run(run_name=config.exp_name):
            for k, v in config.__dict__.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("MCC_all", possible_score_all)
            mlflow.log_metric("MCC_extracted", possible_score_extracted)
            mlflow.log_param("output_dir", output_dir)
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

                pd.DataFrame(results).to_csv(f"{output_dir}/results.csv", index=False)
            mlflow.log_param("threshold", best_th)
            mlflow.log_param("func", best_func.__name__)
    except Exception as e:
        print(e)


if __name__ == "__main__":

    # for n_frames in [31]:
    #     for n_predict_frames in [1, 3, 5]:
    #         image_path = "images_96x96_v2"
    #         exp_name = f"2d_n_frames_{n_frames}, n_predict_frames{n_predict_frames}"
    #         config = Config(exp_name=exp_name, n_predict_frames=n_predict_frames, n_frames=n_frames)
    #         main(config)

    image_path = "images_96x96_v2"
    exp_name = f"2d_1dcnn_3layers_GELU"
    config = Config(exp_name=exp_name, n_predict_frames=1, n_frames=31, seq_model="1dcnn_3layers", activation=nn.GELU)
    main(config)