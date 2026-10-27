import os
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import ( accuracy_score)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from captum.attr import IntegratedGradients

from tqdm import tqdm

# ============================================================================
# 0. CONFIGURATION
# ============================================================================

DATA_DIR = r"MIMIC-IV"
OUT_DIR = "outputs"
RANDOM_SEED = 42
NUM_CLIENTS = 4
NUM_FED_ROUNDS = 30
LOCAL_EPOCHS = 2
GRAPH_K_NEIGHBORS = 8
LABEVENTS_CHUNK_SIZE = 200_000
PRESCRIPTIONS_CHUNK_SIZE = 200_000
out_dir= "outputs"
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PLOT_FONT = "Times New Roman"
PLOT_FONT_SIZE = 16
PLOT_TITLE_SIZE = 18
PLOT_TICK_SIZE = 14
PLOT_LEGEND_SIZE = 13

try:
    fm.findfont(PLOT_FONT, fallback_to_default=False)
except Exception:
    pass

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [PLOT_FONT, "DejaVu Serif", "serif"],
    "font.size": PLOT_FONT_SIZE,
    "axes.titlesize": PLOT_TITLE_SIZE,
    "axes.titleweight": "bold",
    "axes.labelsize": PLOT_FONT_SIZE,
    "axes.labelweight": "bold",
    "xtick.labelsize": PLOT_TICK_SIZE,
    "ytick.labelsize": PLOT_TICK_SIZE,
    "legend.fontsize": PLOT_LEGEND_SIZE,
    "figure.titlesize": PLOT_TITLE_SIZE,
    "figure.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.linewidth": 1.5,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {  "before": "#E63946", "after": "#2A9D8F" }

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def style_ax(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent bold styling to an axis."""
    if title:
        ax.set_title(title, fontname=PLOT_FONT, fontsize=PLOT_TITLE_SIZE,
                      fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontname=PLOT_FONT, fontsize=PLOT_FONT_SIZE,
                       fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontname=PLOT_FONT, fontsize=PLOT_FONT_SIZE,
                       fontweight="bold")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(PLOT_FONT)
        label.set_fontsize(PLOT_TICK_SIZE)
        label.set_fontweight("bold")


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_mimic_tables(data_dir):
    section("STEP 1 / 7  -  Loading ALL MIMIC-IV tables")
    paths = {
        "patients": os.path.join(data_dir, "patients.csv"),
        "admissions": os.path.join(data_dir, "admissions.csv"),
        "diagnoses": os.path.join(data_dir, "diagnoses_icd.csv"),
        "procedures": os.path.join(data_dir, "procedures_icd.csv"),
    }
    tables = {}
    for name, p in paths.items():
        if not os.path.exists(p):
            log(f"  ! {name}: file not found at {p} (will be handled as missing)")
            tables[name] = None
            continue
        log(f"  Loading {name} from {p} ...")
        tables[name] = pd.read_csv(p, low_memory=False)
        log(f"    -> {tables[name].shape[0]:,} rows, {tables[name].shape[1]} cols")

    lab_path = os.path.join(data_dir, "labevents.csv")
    if os.path.exists(lab_path):
        log(f"  Loading labevents.csv in chunks (size={LABEVENTS_CHUNK_SIZE:,})...")
        lab_agg_chunks = []
        usecols = ["subject_id", "hadm_id", "itemid", "valuenum", "flag"]
        for chunk in tqdm(pd.read_csv(lab_path, usecols=usecols,
                                       chunksize=LABEVENTS_CHUNK_SIZE),
                           desc="  labevents chunks"):
            chunk = chunk.dropna(subset=["subject_id"])
            chunk["subject_id"] = chunk["subject_id"].astype(int)
            chunk["abnormal_flag"] = (chunk["flag"].notna() &
                                       (chunk["flag"] != "")).astype(int)
            agg = chunk.groupby("subject_id").agg(
                lab_n_events=("itemid", "count"),
                lab_n_unique_tests=("itemid", "nunique"),
                lab_mean_valuenum=("valuenum", "mean"),
                lab_std_valuenum=("valuenum", "std"),
                lab_n_abnormal=("abnormal_flag", "sum"),
            ).reset_index()
            lab_agg_chunks.append(agg)
        tables["labevents_agg"] = pd.concat(lab_agg_chunks, ignore_index=True)

        tables["labevents_agg"] = tables["labevents_agg"].groupby("subject_id").agg(
            lab_n_events=("lab_n_events", "sum"),
            lab_n_unique_tests=("lab_n_unique_tests", "sum"),
            lab_mean_valuenum=("lab_mean_valuenum", "mean"),
            lab_std_valuenum=("lab_std_valuenum", "mean"),
            lab_n_abnormal=("lab_n_abnormal", "sum"),
        ).reset_index()
        log(f"    -> aggregated labevents for "
            f"{tables['labevents_agg'].shape[0]:,} patients")
    else:
        tables["labevents_agg"] = None
        log("  labevents.csv not found; lab features will be imputed as 0.")

    pres_path = os.path.join(data_dir, "prescriptions.csv")
    if os.path.exists(pres_path):
        log(f"  Loading prescriptions.csv in chunks (size={PRESCRIPTIONS_CHUNK_SIZE:,})...")
        total_counts = {}
        unique_drugs = {}
        required_cols = {"subject_id", "drug"}

        for chunk in tqdm(pd.read_csv(pres_path, usecols=lambda c: c in required_cols,
                                       chunksize=PRESCRIPTIONS_CHUNK_SIZE),
                           desc="  prescriptions chunks"):
            chunk = chunk.dropna(subset=["subject_id", "drug"]).copy()
            if chunk.empty:
                continue
            chunk["subject_id"] = chunk["subject_id"].astype(int)
            chunk["drug"] = chunk["drug"].astype(str)

            for subject_id, drug in chunk[["subject_id", "drug"]].itertuples(index=False, name=None):
                total_counts[subject_id] = total_counts.get(subject_id, 0) + 1
                unique_drugs.setdefault(subject_id, set()).add(drug)

        if total_counts:
            tables["prescriptions_agg"] = pd.DataFrame({
                "subject_id": list(total_counts.keys()),
                "n_prescriptions": [total_counts[k] for k in total_counts],
                "n_unique_drugs": [len(unique_drugs[k]) for k in total_counts],
            })
        else:
            tables["prescriptions_agg"] = pd.DataFrame(columns=["subject_id", "n_prescriptions", "n_unique_drugs"])
        log(f"    -> aggregated prescriptions for {tables['prescriptions_agg'].shape[0]:,} patients")
    else:
        tables["prescriptions_agg"] = None
        log("  prescriptions.csv not found; prescription features will be imputed as 0.")

    return tables


# ============================================================================
# 2. PREPROCESSING
# ============================================================================

def build_patient_feature_table(tables):
    section("STEP 2 / 7  -  Preprocessing: imputation, "
            "normalization")

    patients = tables["patients"].copy()
    admissions = tables["admissions"].copy()
    diagnoses = tables["diagnoses"]
    procedures = tables["procedures"]
    prescriptions_agg = tables.get("prescriptions_agg")
    labevents_agg = tables.get("labevents_agg")

    subj_ids = set(patients["subject_id"])
    log(f"  Total patients: {len(patients):,}")

    admissions = admissions[admissions["subject_id"].isin(subj_ids)].copy()
    log(f"  {len(admissions):,} admissions for cohort")

    log("  Aggregating admissions -> patient level...")
    adm_agg = admissions.groupby("subject_id").agg(
        n_admissions=("hadm_id", "nunique"),
        n_deaths=("hospital_expire_flag", "sum"),
        most_common_admission_type=("admission_type",
            lambda x: x.mode().iat[0] if len(x.mode()) else np.nan),
        most_common_insurance=("insurance",
            lambda x: x.mode().iat[0] if len(x.mode()) else np.nan),
        most_common_marital_status=("marital_status",
            lambda x: x.mode().iat[0] if len(x.mode()) else np.nan),
        most_common_race=("race",
            lambda x: x.mode().iat[0] if len(x.mode()) else np.nan),
        most_common_language=("language",
            lambda x: x.mode().iat[0] if len(x.mode()) else np.nan),
        most_common_admission_location=("admission_location",
            lambda x: x.mode().iat[0] if len(x.mode()) else np.nan),
        most_common_discharge_location=("discharge_location",
            lambda x: x.mode().iat[0] if len(x.mode()) else np.nan),
    ).reset_index()

    if diagnoses is not None:
        diagnoses = diagnoses[diagnoses["subject_id"].isin(subj_ids)]
        diag_agg = diagnoses.groupby("subject_id").agg(
            n_diagnoses=("icd_code", "count"),
            n_unique_diag_codes=("icd_code", "nunique"),
        ).reset_index()
    else:
        diag_agg = pd.DataFrame({"subject_id": list(subj_ids),
                                  "n_diagnoses": 0, "n_unique_diag_codes": 0})

    if procedures is not None:
        procedures = procedures[procedures["subject_id"].isin(subj_ids)]
        proc_agg = procedures.groupby("subject_id").agg(
            n_procedures=("icd_code", "count"),
        ).reset_index()
    else:
        proc_agg = pd.DataFrame({"subject_id": list(subj_ids), "n_procedures": 0})

    if prescriptions_agg is not None:
        presc_agg = prescriptions_agg[prescriptions_agg["subject_id"].isin(subj_ids)].copy()
    else:
        presc_agg = pd.DataFrame({"subject_id": list(subj_ids),
                                    "n_prescriptions": 0, "n_unique_drugs": 0})

    log("  Merging all patient-level tables...")
    df = patients.merge(adm_agg, on="subject_id", how="left")
    df = df.merge(diag_agg, on="subject_id", how="left")
    df = df.merge(proc_agg, on="subject_id", how="left")
    df = df.merge(presc_agg, on="subject_id", how="left")

    if labevents_agg is not None:
        df = df.merge(labevents_agg, on="subject_id", how="left")
    else:
        for c in ["lab_n_events", "lab_n_unique_tests", "lab_mean_valuenum",
                   "lab_std_valuenum", "lab_n_abnormal"]:
            df[c] = 0.0

    df["deceased"] = df["dod"].notna().astype(int) if "dod" in df.columns else 0

    numeric_cols = [
        "anchor_age", "n_admissions", "n_deaths", "n_diagnoses",
        "n_unique_diag_codes", "n_procedures", "n_prescriptions",
        "n_unique_drugs", "lab_n_events", "lab_n_unique_tests",
        "lab_mean_valuenum", "lab_std_valuenum", "lab_n_abnormal",
    ]
    categorical_cols = [
        "gender", "anchor_year_group", "most_common_admission_type",
        "most_common_insurance", "most_common_marital_status",
        "most_common_race", "most_common_language",
        "most_common_admission_location", "most_common_discharge_location",
    ]

    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    log(f"  Numeric features ({len(numeric_cols)}): {numeric_cols}")
    log(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # ---- Missing-value imputation ----
    log("  Imputing missing numeric (median) and categorical (most frequent)...")
    num_imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    outlier_counts = {}
    for col in numeric_cols:
        vals = df[col].values
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((vals < lower) | (vals > upper)).sum())
        if n_outliers > 0:
            df[col] = np.clip(vals, lower, upper)
            outlier_counts[col] = n_outliers

    # Save BEFORE normalization for box plot
    before_norm = df[numeric_cols].copy()

    # ---- Min-max normalization ----
    log("  Applying min-max normalization to numeric features...")
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    after_norm = df[numeric_cols].copy()

    encoders = {}
    for c in categorical_cols:
        le = LabelEncoder()
        df[c + "_enc"] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    return df, numeric_cols, categorical_cols, before_norm, after_norm


def plot_normalization_boxplot(before_norm, after_norm, out_dir):
    before_vals = before_norm.values.flatten()
    before_vals = before_vals[~np.isnan(before_vals)]
    after_vals = after_norm.values.flatten()
    after_vals = after_vals[~np.isnan(after_vals)]

    fig, ax = plt.subplots(figsize=(7, 6))
    bp = ax.boxplot([before_vals, after_vals],
                     tick_labels=["Before Normalization", "After Normalization"],
                     patch_artist=True, widths=0.5, showfliers=True,
                     medianprops=dict(color="black", linewidth=2.5))
    for patch, c in zip(bp['boxes'], [COLORS["before"], COLORS["after"]]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
        patch.set_linewidth(2)
    style_ax(ax, title="Data Distribution Before and After Normalization",
             ylabel="Feature Values")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "normalization_boxplot.png"))
    plt.show()
    plt.close(fig)

# ============================================================================
# 3. GRAPH CONSTRUCTION + GRAPH FEDERATED LEARNING (GFL)
# ============================================================================

def build_patient_graph_edge_index(feature_matrix, k=GRAPH_K_NEIGHBORS):
    log(f"  Building patient similarity graph (k={k})...")
    n = feature_matrix.shape[0]
    k_eff = min(k, max(1, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto").fit(feature_matrix)
    _, indices = nbrs.kneighbors(feature_matrix)

    src, dst = [], []
    for i in range(n):
        for j in indices[i][1:]:
            src.append(i); dst.append(j)
            src.append(j); dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


class _GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.batch_norm1(h)
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        h = self.batch_norm2(h)
        h = self.dropout(h)
        h = F.relu(self.conv3(h, edge_index))
        logits = self.out(h)
        return logits, h


def fed_avg(state_dicts, weights):
    avg = {}
    total = sum(weights)
    for key in state_dicts[0].keys():
        avg[key] = sum(sd[key] * (w / total) for sd, w in zip(state_dicts, weights))
    return avg


def run_graph_federated_learning(df, numeric_cols, out_dir,
                                  num_clients=NUM_CLIENTS,
                                  num_rounds=NUM_FED_ROUNDS,
                                  local_epochs=LOCAL_EPOCHS):
    section("STEP 3 / 7  -  Graph Federated Learning (GFL)")

    X_all = df[numeric_cols].values.astype(np.float32)
    y_all = df["deceased"].values.astype(np.int64) if "deceased" in df.columns else \
        np.zeros(len(df), dtype=np.int64)
    n = len(df)

    log(f"  Partitioning {n:,} patients across {num_clients} clients...")
    idx = np.arange(n)
    np.random.shuffle(idx)
    client_indices = np.array_split(idx, num_clients)
    for i, ci in enumerate(client_indices):
        log(f"    Client {i+1}: {len(ci):,} patients")

    log("  Building per-client patient similarity graphs...")
    client_data = []
    for ci in tqdm(client_indices, desc="  Building client graphs"):
        Xc = X_all[ci]
        yc = y_all[ci]
        edge_index_c = build_patient_graph_edge_index(
            Xc, k=min(GRAPH_K_NEIGHBORS, max(1, len(ci) - 1)))
        client_data.append((torch.tensor(Xc), torch.tensor(yc), edge_index_c))

    in_dim = X_all.shape[1]
    global_model = _GCN(in_dim).to(DEVICE)

    round_losses = []
    round_accs = []
    log(f"  Starting {num_rounds} communication rounds of Federated Averaging...")
    for rnd in tqdm(range(1, num_rounds + 1), desc="  Federated rounds"):
        local_states, local_sizes, local_losses, local_accs = [], [], [], []
        for (Xc, yc, edge_index_c) in client_data:
            local_model = _GCN(in_dim).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            local_model.train()
            opt = torch.optim.Adam(local_model.parameters(), lr=0.01)
            Xc_d, yc_d, ei_d = Xc.to(DEVICE), yc.to(DEVICE), edge_index_c.to(DEVICE)

            last_loss = None
            for _ in range(local_epochs):
                opt.zero_grad()
                logits, _ = local_model(Xc_d, ei_d)
                loss = F.cross_entropy(logits, yc_d)
                loss.backward()
                opt.step()
                last_loss = loss.item()

            local_model.eval()
            with torch.no_grad():
                logits, _ = local_model(Xc_d, ei_d)
                preds = logits.argmax(dim=1)
                acc = (preds == yc_d).float().mean().item()

            local_states.append({k: v.detach().cpu() for k, v in
                                  local_model.state_dict().items()})
            local_sizes.append(len(yc))
            local_losses.append(last_loss)
            local_accs.append(acc)

        new_state = fed_avg(local_states, local_sizes)
        global_model.load_state_dict(new_state)

        avg_loss = float(np.average(local_losses, weights=local_sizes))
        avg_acc = float(np.average(local_accs, weights=local_sizes))
        round_losses.append(avg_loss)
        round_accs.append(avg_acc)

    log(f"  GFL complete. Final loss = {round_losses[-1]:.4f}, "
        f"final acc = {round_accs[-1]:.4f}")

    log("  Computing final patient embeddings from converged global model...")
    edge_index_full = build_patient_graph_edge_index(X_all, k=GRAPH_K_NEIGHBORS)
    global_model.eval()
    with torch.no_grad():
        _, embeddings = global_model(
            torch.tensor(X_all).to(DEVICE), edge_index_full.to(DEVICE))
    embeddings = embeddings.cpu().numpy()

    return global_model, embeddings, round_losses, round_accs


# ============================================================================
# 4. TRANSFORMER SENSITIVE ATTRIBUTE IDENTIFICATION (TSAI)
# ============================================================================

ATTRIBUTE_SENSITIVITY_LABELS = {
    "anchor_age": 0,
    "gender": 0,
    "n_admissions": 0,
    "most_common_admission_type": 0,
    "lab_n_unique_tests": 0,
    "most_common_marital_status": 1,
    "most_common_insurance": 1,
    "n_procedures": 1,
    "n_unique_drugs": 1,
    "most_common_language": 1,
    "most_common_admission_location": 1,
    "most_common_race": 2,
    "n_deaths": 2,
    "n_diagnoses": 2,
    "n_unique_diag_codes": 2,
    "n_prescriptions": 2,
    "lab_n_events": 2,
    "lab_n_abnormal": 2,
    "lab_mean_valuenum": 2,
    "lab_std_valuenum": 2,
}
SENSITIVITY_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def level_to_int(level):
    if isinstance(level, str):
        return {v: k for k, v in SENSITIVITY_NAMES.items()}[level]
    return int(level)


def infer_sensitivity_levels(df, numeric_cols, embeddings,
                             default_map=None, threshold_low=0.33,
                             threshold_high=0.66):

    score_map = {}
    n_samples, n_emb = embeddings.shape
    if n_samples != len(df):
        embeddings = embeddings[:len(df)]

    for col in numeric_cols:
        vals = df[col].values.astype(np.float32)
        vals = np.nan_to_num(vals, nan=np.nanmedian(vals))
        std = float(np.std(vals)) + 1e-8
        if std <= 1e-8:
            variance_score = 0.0
        else:
            variance_score = float(np.var(vals) / (np.var(vals) + std))

        if len(vals) > 1 and np.std(vals) > 1e-8:
            corr = np.corrcoef(vals, embeddings[:, 0])[0, 1]
            emb_score = float(np.abs(corr)) if np.isfinite(corr) else 0.0
        else:
            emb_score = 0.0

        base = 0.6 * variance_score + 0.4 * emb_score
        if default_map is not None and col in default_map:
            base = 0.7 * base + 0.3 * (level_to_int(default_map[col]) / 2.0)
        score_map[col] = float(np.clip(base, 0.0, 1.0))

    sensitivity_map = {}
    for col, score in score_map.items():
        if score < threshold_low:
            lv = "Low"
        elif score < threshold_high:
            lv = "Medium"
        else:
            lv = "High"
        sensitivity_map[col] = lv
    return sensitivity_map


class AttributeTransformer(nn.Module):
    def __init__(self, seq_len=32, d_model=32, nhead=4, num_layers=3, num_classes=3):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=4 * d_model, batch_first=True,
                                        dropout=0.1)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes),
        )
        self.last_attn_weights = None

    def forward(self, x):
        h = self.value_proj(x) + self.pos_embed
        attn_stack = []
        for layer in self.layers:
            attn_out, attn_w = layer.self_attn(h, h, h, need_weights=True,
                                                average_attn_weights=True)
            h = layer.norm1(h + layer.dropout1(attn_out))
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            h = layer.norm2(h + layer.dropout2(ff))
            attn_stack.append(attn_w)
        self.last_attn_weights = attn_stack[-1]
        pooled = h.mean(dim=1)
        return self.classifier(pooled)


def make_attribute_sequences(df, columns, seq_len=32, n_bootstrap=80,
                            sensitivity_map=None):
    log("  Building bootstrap sequences per attribute for TSAI training...")
    X_seqs, y_labels, attr_names = [], [], []
    if sensitivity_map is None:
        sensitivity_map = {c: SENSITIVITY_NAMES.get(ATTRIBUTE_SENSITIVITY_LABELS.get(c, 0), "Low")
                           for c in columns}
    for col in columns:
        if col not in sensitivity_map:
            continue
        label = level_to_int(sensitivity_map[col])
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            le = LabelEncoder()
            vals = le.fit_transform(series.astype(str)).astype(np.float32)
            vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        else:
            vals = series.values.astype(np.float32)
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            vals = (vals - vmin) / (vmax - vmin + 1e-8)
        vals = np.nan_to_num(vals, nan=0.0)

        for _ in range(n_bootstrap):
            if len(vals) >= seq_len:
                sample = np.random.choice(vals, size=seq_len, replace=False)
            else:
                sample = np.random.choice(vals, size=seq_len, replace=True)
            X_seqs.append(sample.reshape(seq_len, 1))
            y_labels.append(label)
            attr_names.append(col)

    X_seqs = np.array(X_seqs, dtype=np.float32)
    y_labels = np.array(y_labels, dtype=np.int64)
    return X_seqs, y_labels, attr_names


def run_tsai(df, numeric_cols, categorical_cols, out_dir, seq_len=32,
             sensitivity_map=None):
    section("STEP 4 / 7  -  Transformer Sensitive Attribute Identification (TSAI)")

    if sensitivity_map is None:
        sensitivity_map = {c: SENSITIVITY_NAMES.get(ATTRIBUTE_SENSITIVITY_LABELS.get(c, 0), "Low")
                           for c in (numeric_cols + categorical_cols)}
    all_cols = [c for c in numeric_cols + categorical_cols if c in sensitivity_map]

    X_seqs, y_labels, attr_names = make_attribute_sequences(
        df, all_cols, seq_len=seq_len, sensitivity_map=sensitivity_map)

    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X_seqs, y_labels, attr_names, test_size=0.25,
        random_state=RANDOM_SEED, stratify=y_labels
    )

    model = AttributeTransformer(seq_len=seq_len).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)

    X_train_t = torch.tensor(X_train).to(DEVICE)
    y_train_t = torch.tensor(y_train).to(DEVICE)
    X_test_t = torch.tensor(X_test).to(DEVICE)
    y_test_t = torch.tensor(y_test).to(DEVICE)

    class_counts = np.bincount(y_train, minlength=3)
    class_weights = np.zeros(3, dtype=np.float32)
    for c in range(3):
        if class_counts[c] > 0:
            class_weights[c] = len(y_train) / (3.0 * class_counts[c])
        else:
            class_weights[c] = 1.0
    class_weights_t = torch.tensor(class_weights).to(DEVICE)

    n_epochs = 80
    log(f"  Training TSAI transformer for {n_epochs} epochs...")
    model.train()
    for epoch in tqdm(range(n_epochs), desc="  TSAI training"):
        opt.zero_grad()
        logits = model(X_train_t)
        loss = F.cross_entropy(logits, y_train_t, weight=class_weights_t)
        loss.backward()
        opt.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        preds = test_logits.argmax(dim=1).cpu().numpy()
    y_test_np = y_test_t.cpu().numpy()

    acc = accuracy_score(y_test_np, preds)

    model.eval()
    with torch.no_grad():
        all_logits = model(torch.tensor(X_seqs).to(DEVICE))
        all_preds = all_logits.argmax(dim=1).cpu().numpy()
    attr_pred_df = pd.DataFrame({"attribute": attr_names, "pred_level": all_preds})
    final_levels = attr_pred_df.groupby("attribute")["pred_level"].agg(
        lambda x: x.value_counts().idxmax())
    final_levels_named = final_levels.map(SENSITIVITY_NAMES).to_dict()

    level_counts = {SENSITIVITY_NAMES[i]: 0 for i in range(3)}
    for v in final_levels_named.values():
        level_counts[v] += 1

    return model, final_levels_named, acc, X_seqs, attr_names, seq_len


# ============================================================================
# 5. SECRETARY TYRANNOSAURUS OPTIMIZATION (STO) 
# ============================================================================

def sanitization_fitness(params, level_order, df, numeric_cols, sensitivity_map):
    eps_by_level = dict(zip(level_order, params))
    total_utility_loss, total_privacy_gain, n_cols = 0.0, 0.0, 0
    for col in numeric_cols:
        level = sensitivity_map.get(col, level_order[len(level_order) // 2])
        eps = eps_by_level.get(level, np.mean(params))
        vals = df[col].values.astype(np.float32)
        std = np.std(vals) + 1e-8
        noise = np.random.normal(0, eps * std, size=vals.shape)
        sanitized = vals + noise
        utility_loss = np.mean(np.abs(sanitized - vals)) / std
        privacy_gain = eps
        total_utility_loss += utility_loss
        total_privacy_gain += privacy_gain
        n_cols += 1
    total_utility_loss /= max(n_cols, 1)
    total_privacy_gain /= max(n_cols, 1)
    alpha, beta = 0.6, 0.4
    fitness = beta * total_utility_loss - alpha * total_privacy_gain
    return fitness


def secretary_tyrannosaurus_optimization(df, numeric_cols, sensitivity_map,
                                          pop_size=30, iterations=100):
    section("STEP 5 / 7  -  Secretary Tyrannosaurus Optimization (STO) [novel]")

    level_order = sorted(set(sensitivity_map.values()),
                          key=lambda x: {"Low": 0, "Medium": 1, "High": 2}.get(x, 1))
    dim = len(level_order)
    log(f"  Optimizing {dim} epsilon parameters "
        f"(pop={pop_size}, iters={iterations})...")

    lb, ub = 0.01, 1.0
    population = np.random.uniform(lb, ub, size=(pop_size, dim))
    personal_best = population.copy()
    fitness = np.array([sanitization_fitness(ind, level_order, df,
                                                numeric_cols, sensitivity_map)
                         for ind in population])
    personal_best_fit = fitness.copy()
    best_idx = np.argmin(fitness)
    best_pos = population[best_idx].copy()
    best_fit = fitness[best_idx]

    convergence = [best_fit]
    phase_log = []

    for t in tqdm(range(1, iterations + 1), desc="  STO iterations"):
        d = 1.0 - (t / iterations)
        for i in range(pop_size):
            if d > 0.66:
                phase = "search"
                walk = np.random.uniform(-1, 1, size=dim) * (ub - lb) * d
                pull_to_best = np.random.rand(dim) * (best_pos - population[i]) * 0.3
                candidate = population[i] + walk + pull_to_best
            elif d > 0.25:
                phase = "pursue"
                pursuit_speed = (1 - d)
                target = 0.5 * best_pos + 0.5 * personal_best[i]
                candidate = population[i] + pursuit_speed * (target - population[i]) \
                    * random.uniform(0.8, 1.3)
            else:
                phase = "strike"
                strike_radius = max(d, 0.03) * (ub - lb) * 0.5
                candidate = best_pos + np.random.uniform(-1, 1, size=dim) * strike_radius

            candidate = np.clip(candidate, lb, ub)
            cand_fit = sanitization_fitness(candidate, level_order, df,
                                              numeric_cols, sensitivity_map)
            if cand_fit < fitness[i]:
                population[i] = candidate
                fitness[i] = cand_fit
                if cand_fit < personal_best_fit[i]:
                    personal_best[i] = candidate.copy()
                    personal_best_fit[i] = cand_fit
                if cand_fit < best_fit:
                    best_fit = cand_fit
                    best_pos = candidate.copy()
        convergence.append(best_fit)
        phase_log.append(phase)

    log(f"  Phases: search={phase_log.count('search')}, "
        f"pursue={phase_log.count('pursue')}, "
        f"strike={phase_log.count('strike')}")

    return dict(zip(level_order, best_pos)), convergence


# ============================================================================
# 6. ADAPTIVE MULTI-LEVEL DATA SANITIZATION (MLDS)
#    Perturbation + Generalization + Suppression
# ============================================================================

def apply_adaptive_sanitization(df, numeric_cols, sensitivity_map, optimal_eps, out_dir):
    section("STEP 6 / 7  -  Adaptive Multi-Level Data Sanitization (MLDS)")
    log("  Techniques per level:")
    log("    Low:    mild Gaussian perturbation")
    log("    Medium: Laplace perturbation + generalization (binning)")
    log("    High:   Laplace perturbation + generalization + suppression of rare values")

    sanitized_df = df.copy()
    distortion_records = []
    orig_sample_col, san_sample_col = None, None

    for col in tqdm(numeric_cols, desc="  Sanitizing attributes"):
        level = sensitivity_map.get(col, "Medium")
        eps = optimal_eps.get(level, np.mean(list(optimal_eps.values())))
        vals = df[col].values.astype(np.float32)
        std = np.std(vals) + 1e-8

        if level == "High":
            # 1) Perturbation (Laplace)
            noise = np.random.laplace(0, eps * std, size=vals.shape)
            sanitized = vals + noise
            # 2) Generalization (quantile binning into 5 bins)
            bins = np.quantile(sanitized, np.linspace(0, 1, 6))
            sanitized = np.digitize(sanitized, bins[1:-1]).astype(np.float32)
            sanitized = (sanitized - sanitized.min()) / \
                (sanitized.max() - sanitized.min() + 1e-8)
            # 3) Suppression: set rare outlier values (top/bottom 1%) to median
            low_pct, high_pct = np.percentile(sanitized, [1, 99])
            mask_suppress = (sanitized < low_pct) | (sanitized > high_pct)
            if mask_suppress.sum() > 0:
                sanitized[mask_suppress] = np.median(sanitized)
        elif level == "Medium":
            # 1) Perturbation (Laplace)
            noise = np.random.laplace(0, eps * std, size=vals.shape)
            sanitized = vals + noise
            # 2) Generalization (coarser binning into 3 bins)
            bins = np.quantile(sanitized, np.linspace(0, 1, 4))
            sanitized = np.digitize(sanitized, bins[1:-1]).astype(np.float32)
            sanitized = (sanitized - sanitized.min()) / \
                (sanitized.max() - sanitized.min() + 1e-8)
        else:  # Low
            # Mild perturbation only
            noise = np.random.normal(0, eps * std * 0.3, size=vals.shape)
            sanitized = vals + noise

        sanitized_df[col] = sanitized
        distortion = np.mean(np.abs(sanitized - vals))
        distortion_records.append({
            "attribute": col, "level": level, "epsilon": float(eps),
            "mean_distortion": float(distortion),
        })

        if orig_sample_col is None and level in ("Medium", "High"):
            orig_sample_col = vals.copy(), sanitized.copy()

    if orig_sample_col is None:
        col0 = numeric_cols[0]
        orig_sample_col = df[col0].values.astype(np.float32)
    return sanitized_df, distortion_records


# ============================================================================
# 7. EXPLAINABLE AI (XAI)
# ============================================================================

def run_xai(tsai_model, X_seqs, attr_names, sensitivity_map, out_dir, seq_len,
            max_seqs_per_attr=15):
    section("STEP 7 / 7  -  Explainable AI (XAI): Integrated Gradients + Attention")

    tsai_model.eval()
    ig = IntegratedGradients(tsai_model)

    attr_df = pd.DataFrame({"attribute": attr_names})
    attr_df["seq_idx"] = np.arange(len(attr_names))

    rows = []
    unique_attrs = sorted(set(attr_names))
    for col in tqdm(unique_attrs, desc="  Computing IG + attention per attribute"):
        idxs = attr_df.loc[attr_df["attribute"] == col, "seq_idx"].values
        if len(idxs) > max_seqs_per_attr:
            idxs = np.random.choice(idxs, size=max_seqs_per_attr, replace=False)

        batch = torch.tensor(X_seqs[idxs]).to(DEVICE)

        with torch.no_grad():
            logits = tsai_model(batch)
            pred_class = logits.argmax(dim=1)

        ig_attributions = []
        for b in range(batch.shape[0]):
            single = batch[b:b + 1]
            target = int(pred_class[b].item())
            attr = ig.attribute(single, target=target, n_steps=50)
            ig_attributions.append(attr.squeeze(-1).squeeze(0).detach().cpu().numpy())
        ig_attributions = np.array(ig_attributions)
        mean_ig = np.abs(ig_attributions).mean(axis=0)
        ig_concentration = float(mean_ig.max() / (mean_ig.sum() + 1e-8))

        with torch.no_grad():
            _ = tsai_model(batch)
            attn = tsai_model.last_attn_weights.cpu().numpy()
        mean_attn_received = attn.mean(axis=(0, 1))
        attn_entropy = float(
            -(mean_attn_received * np.log(mean_attn_received + 1e-8)).sum())
        max_attn_entropy = float(np.log(seq_len))
        attn_focus = 1.0 - (attn_entropy / max_attn_entropy)

        level = sensitivity_map.get(col, "Unknown")
        rows.append({
            "attribute": col,
            "sensitivity_level": level,
            "ig_top_position": int(mean_ig.argmax()),
            "ig_attribution_concentration": round(ig_concentration, 4),
            "attention_focus_score": round(attn_focus, 4),
        })

    xai_df = pd.DataFrame(rows)

    return xai_df


# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================

def main():
    section("MIMIC-IV PRIVACY-PRESERVING GFL PIPELINE (GFL-TSAI-STO-MLDS)")
    log(f"Device: {DEVICE}")
    log(f"Data directory: {DATA_DIR}")
    log(f"Output directory: {OUT_DIR}")

    tables = load_mimic_tables(DATA_DIR)

    df, numeric_cols, categorical_cols, before_norm, after_norm = \
        build_patient_feature_table(tables)
    plot_normalization_boxplot(before_norm, after_norm, OUT_DIR)

    global_model, embeddings, round_losses, round_accs = \
        run_graph_federated_learning(df, numeric_cols, OUT_DIR)

    sensitivity_map = infer_sensitivity_levels(df, numeric_cols, embeddings,
                                              default_map=ATTRIBUTE_SENSITIVITY_LABELS)

    tsai_model, sensitivity_map, tsai_acc, tsai_X_seqs, tsai_attr_names, \
        tsai_seq_len = run_tsai(df, numeric_cols, categorical_cols, OUT_DIR,
                                sensitivity_map=sensitivity_map)

    optimal_eps, sto_convergence = secretary_tyrannosaurus_optimization(
        df, numeric_cols, sensitivity_map)

    sanitized_df, distortion_records = apply_adaptive_sanitization(
        df, numeric_cols, sensitivity_map, optimal_eps, OUT_DIR)

    xai_df = run_xai(tsai_model, tsai_X_seqs, tsai_attr_names,
                      sensitivity_map, OUT_DIR, tsai_seq_len)
    import utils_plot
    section("PIPELINE COMPLETE")

if __name__ == "__main__":
    main()