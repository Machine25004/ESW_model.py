
import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition, Element
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

MAX_ELEMENTS = 20
SEED = 42
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 128
EPOCHS = 300
PATIENCE = 35
LR = 3e-4
WEIGHT_DECAY = 1e-4

TARGET_COLUMNS = ["left", "right"]

EMBED_FILES = {
    "s64": DATA_DIR / "Mat2Vec-S64.csv",
    "s32": DATA_DIR / "Mat2Vec-S32.csv",
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ==========================================
# 1. 读取作者提供的预训练元素嵌入
# ==========================================

class ElementEmbedding:
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        # 兼容 Element、element、第一列符号等形式
        candidates = [
            c for c in df.columns
            if c.lower() in ("element", "symbol")
        ]

        symbol_col = (
            candidates[0] if candidates else df.columns[0]
        )

        symbols = df[symbol_col].astype(str).str.strip()

        if symbols.duplicated().any():
            raise ValueError("元素嵌入表含有重复元素")

        feature_df = df.drop(columns=[symbol_col])

        # 排除常见的自动保存索引
        feature_df = feature_df.loc[
            :,
            ~feature_df.columns.str.startswith("Unnamed")
        ]

        values = feature_df.to_numpy(dtype=np.float32)

        if not np.isfinite(values).all():
            raise ValueError("元素嵌入包含 NaN 或 Inf")

        self.dim = values.shape[1]
        self.lookup = {}

        for symbol, vector in zip(symbols, values):
            # 只接受标准元素符号
            try:
                Element(symbol)
            except Exception:
                raise ValueError(
                    f"无法识别元素符号: {symbol}"
                )

            if np.allclose(vector, 0.0):
                # 有些重建文件用全零表示未覆盖元素
                continue

            self.lookup[symbol] = vector

        if not self.lookup:
            raise ValueError("没有找到有效元素向量")

        print(
            f"Loaded {csv_file.name}: "
            f"{len(self.lookup)} elements, "
            f"dimension={self.dim}"
        )

    def parse_formula(self, formula):
        comp = Composition(str(formula))
        amounts = comp.get_el_amt_dict()

        total = sum(amounts.values())
        pairs = []

        for symbol, amount in amounts.items():
            if symbol not in self.lookup:
                raise ValueError(
                    f"{formula}: 元素 {symbol} 没有有效嵌入"
                )
            pairs.append((symbol, float(amount / total)))

        # 使用原子序数固定元素排列
        pairs.sort(key=lambda x: Element(x[0]).Z)

        return pairs

    def sequence(self, formula):
        pairs = self.parse_formula(formula)

        if len(pairs) > MAX_ELEMENTS:
            raise ValueError(
                f"{formula}: 元素数量超过 {MAX_ELEMENTS}"
            )

        x = np.zeros(
            (MAX_ELEMENTS, self.dim), dtype=np.float32
        )
        fractions = np.zeros(
            MAX_ELEMENTS, dtype=np.float32
        )
        mask = np.zeros(
            MAX_ELEMENTS, dtype=np.bool_
        )

        for i, (symbol, frac) in enumerate(pairs):
            x[i] = self.lookup[symbol]
            fractions[i] = frac
            mask[i] = True

        return x, fractions, mask

    def statistics(self, formula):
        pairs = self.parse_formula(formula)

        vectors = np.stack([
            self.lookup[s] for s, _ in pairs
        ])

        fractions = np.asarray(
            [f for _, f in pairs], dtype=np.float32
        )

        mean = np.sum(
            vectors * fractions[:, None], axis=0
        )
        variance = np.sum(
            fractions[:, None] *
            (vectors - mean) ** 2,
            axis=0
        )
        std = np.sqrt(variance)

        return np.concatenate([
            mean,
            std,
            vectors.max(axis=0),
            vectors.min(axis=0),
        ]).astype(np.float32)


# ==========================================
# 2. 数据清洗与划分
# ==========================================

def canonical_formula(formula):
    return Composition(str(formula)).reduced_formula


def prepare_data():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(DATA_DIR / "esw.xlsx")
    df.columns = df.columns.str.strip()

    required = ["formula", "left", "right"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Excel 缺少必要列: {missing}"
        )

    df = df.copy()
    df["formula"] = df["formula"].astype(str).str.strip()

    for col in TARGET_COLUMNS:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df.dropna(subset=required).copy()

    df["group"] = df["formula"].apply(
        canonical_formula
    )

    # 物理标签基础检查
    invalid = df["right"] <= df["left"]

    if invalid.any():
        print(
            f"警告：发现 {invalid.sum()} 条"
            f"右边界 <= 左边界的记录"
        )
        df.loc[invalid].to_csv(
            OUT_DIR / "invalid_boundaries.csv",
            index=False
        )
        df = df.loc[~invalid].copy()

    df = df.reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("有效记录数量太少")

    if df["group"].nunique() < 3:
        raise ValueError("独立化学组成数量不足")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=SEED
    )
    train_val_idx, test_idx = next(
        splitter.split(df, groups=df["group"])
    )

    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15 / 0.85,
        random_state=SEED + 1
    )
    train_idx, val_idx = next(
        splitter2.split(
            train_val,
            groups=train_val["group"]
        )
    )

    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    for name, part in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        part.to_csv(
            OUT_DIR / f"{name}.csv", index=False
        )

    print(
        f"Train={len(train)}, "
        f"Val={len(val)}, Test={len(test)}"
    )
    print(
        "Unique compositions:",
        df["group"].nunique()
    )


def load_splits():
    return {
        name: pd.read_csv(
            OUT_DIR / f"{name}.csv"
        )
        for name in ("train", "val", "test")
    }


# ==========================================
# 3. 统一评价指标
# ==========================================

def metrics(y_true, y_pred):
    results = {}

    labels = ["left", "right", "esw"]

    yt = [
        y_true[:, 0],
        y_true[:, 1],
        y_true[:, 1] - y_true[:, 0]
    ]

    yp = [
        y_pred[:, 0],
        y_pred[:, 1],
        y_pred[:, 1] - y_pred[:, 0]
    ]

    for name, a, b in zip(labels, yt, yp):
        results[name] = {
            "MAE": float(mean_absolute_error(a, b)),
            "RMSE": float(
                np.sqrt(mean_squared_error(a, b))
            ),
            "R2": float(r2_score(a, b))
        }

    results["invalid_order_count"] = int(
        np.sum(y_pred[:, 1] < y_pred[:, 0])
    )

    return results


def save_predictions(name, df, pred):
    out = df[["formula", "left", "right"]].copy()
    out = out.rename(columns={
        "left": "true_left",
        "right": "true_right"
    })

    out["pred_left"] = pred[:, 0]
    out["pred_right"] = pred[:, 1]

    out["true_esw"] = (
        out["true_right"] - out["true_left"]
    )
    out["pred_esw"] = (
        out["pred_right"] - out["pred_left"]
    )

    out.to_csv(
        OUT_DIR / name / "test_predictions.csv",
        index=False
    )


# ==========================================
# 4. 实验 A: XGBoost + S64
# ==========================================

def train_xgboost(splits, encoder):
    name = "A_XGBoost_S64"
    save_dir = OUT_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    def get_xy(df):
        X = np.stack([
            encoder.statistics(f)
            for f in df["formula"]
        ])
        Y = df[TARGET_COLUMNS].to_numpy(
            dtype=np.float32
        )
        return X, Y

    X_train, y_train = get_xy(splits["train"])
    X_val, y_val = get_xy(splits["val"])
    X_test, y_test = get_xy(splits["test"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=SEED,
            n_jobs=2
        )
    )

    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    result = {
        "val": metrics(y_val, pred_val),
        "test": metrics(y_test, pred_test)
    }

    joblib.dump(
        {"model": model, "scaler": scaler},
        save_dir / "model.joblib"
    )

    save_json(result, save_dir / "metrics.json")
    save_predictions(
        name, splits["test"], pred_test
    )

    print(name, result["test"])
    return result["test"]


# ==========================================
# 5. 训练数据集类
# ==========================================

class ESWDataset(Dataset):
    def __init__(
        self, df, encoder, mode="stats",
        y_scaler=None
    ):
        self.mode = mode
        self.formulas = df["formula"].tolist()

        self.y = None
        if set(TARGET_COLUMNS).issubset(df.columns):
            values = df[TARGET_COLUMNS].to_numpy(
                dtype=np.float32
            )

            if y_scaler is not None:
                values = y_scaler.transform(values)

            self.y = torch.tensor(
                values, dtype=torch.float32
            )

        if mode == "stats":
            features = np.stack([
                encoder.statistics(f)
                for f in self.formulas
            ])

            self.features = torch.tensor(
                features, dtype=torch.float32
            )

        else:
            seqs = [
                encoder.sequence(f)
                for f in self.formulas
            ]

            self.x = torch.tensor(
                np.stack([v[0] for v in seqs]),
                dtype=torch.float32
            )

            self.frac = torch.tensor(
                np.stack([v[1] for v in seqs]),
                dtype=torch.float32
            )

            self.mask = torch.tensor(
                np.stack([v[2] for v in seqs]),
                dtype=torch.bool
            )

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        if self.mode == "stats":
            x = self.features[idx]
            y = (
                self.y[idx] if self.y is not None
                else torch.zeros(2)
            )
            return x, y

        y = (
            self.y[idx] if self.y is not None
            else torch.zeros(2)
        )

        return (
            self.x[idx],
            self.frac[idx],
            self.mask[idx],
            y
        )


# ==========================================
# 6. 实验 B: 双任务 MLP
# ==========================================

class DualTaskMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.left_head = nn.Linear(128, 1)
        self.right_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)

        left = self.left_head(h)
        right = self.right_head(h)

        return torch.cat([left, right], dim=1)


# ==========================================
# 7. 实验 C/D: CrabNet 风格网络
# ==========================================

class FractionEncoding(nn.Module):
    """
    正弦 + 对数原子分数编码。

    这是一个简化的 CrabNet 风格实现，
    不是原仓库 FractionalEncoder 的逐行复制。
    """
    def __init__(self, d_model):
        super().__init__()
        half = d_model // 2

        frequencies = torch.exp(
            -torch.arange(half, dtype=torch.float32)
            * np.log(10000.0)
            / max(half - 1, 1)
        )

        self.register_buffer(
            "frequencies", frequencies
        )

    def forward(self, fractions):
        f = fractions.clamp(min=1e-6)

        # [B, N, D/2]
        linear_phase = (
            f.unsqueeze(-1) * self.frequencies
        )

        log_phase = (
            torch.log(f).unsqueeze(-1)
            * self.frequencies
        )

        return torch.cat([
            torch.sin(linear_phase),
            torch.cos(log_phase)
        ], dim=-1)


class ESWCrabNet(nn.Module):
    """
    组成嵌入 + 分数编码 + Transformer
    + 共享池化 + 双预测头。
    """
    def __init__(
        self,
        embedding_dim,
        d_model=256,
        num_layers=2,
        nhead=4,
        dropout=0.15
    ):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("d_model 必须为偶数")

        self.element_projection = nn.Linear(
            embedding_dim, d_model
        )

        self.fraction_encoder = FractionEncoding(
            d_model
        )

        self.fraction_scale = nn.Parameter(
            torch.tensor(1.0)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False
        )

        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers
        )

        self.shared = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.left_head = nn.Linear(128, 1)
        self.right_head = nn.Linear(128, 1)

    def forward(self, elements, fractions, mask):
        x = self.element_projection(elements)

        position = self.fraction_encoder(fractions)

        x = x + self.fraction_scale * position

        # 避免填充位置对有效元素产生影响
        x = x.masked_fill(
            ~mask.unsqueeze(-1), 0.0
        )

        x = self.transformer(
            x,
            src_key_padding_mask=~mask
        )

        # 按原子分数池化
        weights = fractions * mask.float()
        weights = weights / weights.sum(
            dim=1, keepdim=True
        ).clamp(min=1e-8)

        pooled = (
            x * weights.unsqueeze(-1)
        ).sum(dim=1)

        h = self.shared(pooled)

        left = self.left_head(h)
        right = self.right_head(h)

        return torch.cat([left, right], dim=1)


# ==========================================
# 8. PyTorch 通用训练器
# ==========================================

def forward_batch(model, batch, mode):
    if mode == "stats":
        x, y = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        return model(x), y

    elements, fractions, mask, y = batch

    elements = elements.to(DEVICE)
    fractions = fractions.to(DEVICE)
    mask = mask.to(DEVICE)
    y = y.to(DEVICE)

    return model(elements, fractions, mask), y


def run_epoch(
    model, loader, optimizer, mode,
    train=True
):
    model.train(train)

    total = 0.0
    n = 0

    for batch in loader:
        with torch.set_grad_enabled(train):
            pred, y = forward_batch(
                model, batch, mode
            )

            # 标准化目标下的平均 MSE
            loss = torch.mean((pred - y) ** 2)

            if train:
                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0
                )

                optimizer.step()

        total += loss.item() * len(y)
        n += len(y)

    return total / max(n, 1)


@torch.no_grad()
def predict_torch(model, loader, mode):
    model.eval()
    predictions = []

    for batch in loader:
        pred, _ = forward_batch(
            model, batch, mode
        )

        predictions.append(
            pred.cpu().numpy()
        )

    return np.concatenate(predictions, axis=0)


def train_torch_model(
    name, splits, encoder,
    mode, model_builder
):
    save_dir = OUT_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 目标的标准化只在训练集上拟合
    y_scaler = StandardScaler()
    y_scaler.fit(
        splits["train"][TARGET_COLUMNS].values
    )

    datasets = {
        name_: ESWDataset(
            df, encoder,
            mode=mode,
            y_scaler=y_scaler
        )
        for name_, df in splits.items()
    }

    loaders = {
        name_: DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=(name_ == "train"),
            num_workers=0,
            pin_memory=(DEVICE.type == "cuda")
        )
        for name_, ds in datasets.items()
    }

    model = model_builder().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_loss = float("inf")
    wait = 0
    best_path = save_dir / "best.pt"

    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model, loaders["train"],
            optimizer, mode, train=True
        )

        val_loss = run_epoch(
            model, loaders["val"],
            None, mode, train=False
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "target_mean": y_scaler.mean_,
                "target_scale": y_scaler.scale_,
                "mode": mode,
            }, best_path)
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{name} Epoch {epoch}: "
                f"train={train_loss:.5f}, "
                f"val={val_loss:.5f}"
            )

        if wait >= PATIENCE:
            print(name, "early stopping", epoch)
            break

    pd.DataFrame(history).to_csv(
        save_dir / "history.csv", index=False
    )

    checkpoint = torch.load(
        best_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    results = {}

    for split_name in ("val", "test"):
        scaled_pred = predict_torch(
            model, loaders[split_name], mode
        )

        pred = y_scaler.inverse_transform(
            scaled_pred
        )

        true = splits[split_name][
            TARGET_COLUMNS
        ].to_numpy(dtype=np.float32)

        results[split_name] = metrics(
            true, pred
        )

        if split_name == "test":
            save_predictions(
                name,
                splits["test"],
                pred
            )

    save_json(
        results, save_dir / "metrics.json"
    )

    print(name, results["test"])
    return results["test"]


# ==========================================
# 9. 执行四个实验
# ==========================================

def run_all(selected="all"):
    set_seed()

    if not (OUT_DIR / "train.csv").exists():
        prepare_data()

    splits = load_splits()

    experiments = (
        ["A", "B", "C", "D"]
        if selected == "all"
        else [selected]
    )

    # 只在需要时加载对应维度
    s64 = None
    s32 = None

    if any(x in experiments for x in ["A", "B", "C"]):
        s64 = ElementEmbedding(
            EMBED_FILES["s64"]
        )

    if "D" in experiments:
        s32 = ElementEmbedding(
            EMBED_FILES["s32"]
        )

    summary = {}

    if "A" in experiments:
        summary["A_XGBoost_S64"] = train_xgboost(
            splits, s64
        )

    if "B" in experiments:
        set_seed(SEED)

        summary["B_MLP_S64"] = train_torch_model(
            "B_MLP_S64",
            splits,
            s64,
            mode="stats",
            model_builder=lambda: DualTaskMLP(
                input_dim=s64.dim * 4
            )
        )

    if "C" in experiments:
        set_seed(SEED)

        summary["C_CrabNet_S64"] = train_torch_model(
            "C_CrabNet_S64",
            splits,
            s64,
            mode="sequence",
            model_builder=lambda: ESWCrabNet(
                embedding_dim=s64.dim
            )
        )

    if "D" in experiments:
        set_seed(SEED)

        summary["D_CrabNet_S32"] = train_torch_model(
            "D_CrabNet_S32",
            splits,
            s32,
            mode="sequence",
            model_builder=lambda: ESWCrabNet(
                embedding_dim=s32.dim
            )
        )

    save_json(
        summary,
        OUT_DIR / "comparison.json"
    )

    print("\n========== FINAL COMPARISON ==========")

    for name, result in summary.items():
        print(
            f"{name:22s} "
            f"Left MAE={result['left']['MAE']:.4f}, "
            f"Right MAE={result['right']['MAE']:.4f}, "
            f"ESW MAE={result['esw']['MAE']:.4f}"
        )


# ==========================================
# 10. 新材料预测
# ==========================================

def predict_new(model_name, formulas):
    configurations = {
        "A": (
            "A_XGBoost_S64", "s64", "stats"
        ),
        "B": (
            "B_MLP_S64", "s64", "stats"
        ),
        "C": (
            "C_CrabNet_S64", "s64", "sequence"
        ),
        "D": (
            "D_CrabNet_S32", "s32", "sequence"
        ),
    }

    folder, embedding_key, mode = (
        configurations[model_name]
    )

    encoder = ElementEmbedding(
        EMBED_FILES[embedding_key]
    )

    save_dir = OUT_DIR / folder
    df = pd.DataFrame({"formula": formulas})

    if model_name == "A":
        saved = joblib.load(
            save_dir / "model.joblib"
        )

        X = np.stack([
            encoder.statistics(f)
            for f in formulas
        ])

        X = saved["scaler"].transform(X)
        pred = saved["model"].predict(X)

    else:
        dataset = ESWDataset(
            df, encoder, mode=mode
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        if model_name == "B":
            model = DualTaskMLP(
                encoder.dim * 4
            )
        else:
            model = ESWCrabNet(
                embedding_dim=encoder.dim
            )

        model = model.to(DEVICE)

        ckpt = torch.load(
            save_dir / "best.pt",
            map_location=DEVICE,
            weights_only=False
        )

        model.load_state_dict(
            ckpt["model_state_dict"]
        )

        pred_scaled = predict_torch(
            model, loader, mode
        )

        pred = (
            pred_scaled * ckpt["target_scale"]
            + ckpt["target_mean"]
        )

    result = pd.DataFrame({
        "formula": formulas,
        "pred_left": pred[:, 0],
        "pred_right": pred[:, 1],
    })

    result["pred_esw"] = (
        result["pred_right"]
        - result["pred_left"]
    )

    result["valid_order"] = (
        result["pred_esw"] >= 0
    )

    print(result.to_string(index=False))

    save_path = OUT_DIR / (
        f"prediction_{model_name}.csv"
    )
    result.to_csv(save_path, index=False)

    return result


# ==========================================
# 11. 命令行入口
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "predict"],
        required=True
    )

    parser.add_argument(
        "--model",
        choices=["A", "B", "C", "D", "all"],
        default="all"
    )

    parser.add_argument(
        "--formulas",
        nargs="+",
        default=[]
    )

    args = parser.parse_args()

    if args.mode == "prepare":
        prepare_data()

    elif args.mode == "train":
        run_all(args.model)

    elif args.mode == "predict":
        if args.model == "all":
            raise ValueError(
                "预测时请选择 A、B、C 或 D"
            )
        if not args.formulas:
            raise ValueError(
                "请通过 --formulas 提供化学式"
            )
        predict_new(
            args.model,
            args.formulas
        )
