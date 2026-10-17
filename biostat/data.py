"""資料讀取、欄位判斷、資料轉換與範例資料。"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

EXCEL_EXT = (".xlsx", ".xlsm", ".xls")
TEXT_EXT = (".csv", ".tsv", ".txt")
ENCODINGS = ("utf-8-sig", "cp950", "big5", "gb18030", "latin1")


# ---------------------------------------------------------------- reading
def list_sheets(content: bytes, filename: str) -> list[str]:
    if filename.lower().endswith(EXCEL_EXT):
        with pd.ExcelFile(io.BytesIO(content)) as xf:
            return [str(s) for s in xf.sheet_names]
    return []


def _read_text(content: bytes, header, skiprows: int) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ENCODINGS:
        try:
            text = content.decode(enc)
        except UnicodeDecodeError as err:
            last_err = err
            continue
        return pd.read_csv(io.StringIO(text), sep=None, engine="python", header=header, skiprows=skiprows)
    raise ValueError(f"無法辨識檔案編碼：{last_err}")


def read_table(content: bytes, filename: str, sheet: str | None = None, header_row: int | None = 0,
               skiprows: int = 0, coerce_numeric: bool = True) -> pd.DataFrame:
    """讀取 Excel / CSV / TSV。header_row=None 表示沒有標題列。"""
    name = filename.lower()
    if name.endswith(EXCEL_EXT):
        df = pd.read_excel(io.BytesIO(content), sheet_name=sheet if sheet is not None else 0,
                           header=header_row, skiprows=skiprows or None)
    elif name.endswith(TEXT_EXT) or "." not in name:
        df = _read_text(content, header_row, skiprows)
    else:
        raise ValueError("不支援的檔案格式，請上傳 .xlsx / .xls / .csv / .tsv")
    return clean_frame(df, coerce_numeric=coerce_numeric)


def clean_frame(df: pd.DataFrame, coerce_numeric: bool = True) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    cols: list[str] = []
    seen: dict[str, int] = {}
    for i, c in enumerate(df.columns):
        name = str(c).strip()
        if not name or name.startswith("Unnamed:") or name.lower() == "nan":
            name = f"欄{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    df.columns = cols
    if coerce_numeric:
        for c in df.columns:
            s = df[c]
            if s.dtype == object:
                stripped = s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "NA": np.nan, "N/A": np.nan, "-": np.nan})
                num = pd.to_numeric(stripped.str.replace(",", "", regex=False), errors="coerce")
                valid = stripped.notna()
                if valid.any() and num[valid].notna().all():
                    df[c] = num
            elif str(s.dtype) == "string":
                df[c] = s.astype(object)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- column helpers
def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def is_categorical_like(s: pd.Series, max_levels: int = 15) -> bool:
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return True
    vals = s.dropna()
    return vals.nunique() <= max_levels and np.allclose(vals, np.round(vals))


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """所有欄位，類別型優先排序 (數值欄位也可當分組用，例如 0/1)。"""
    cat = [c for c in df.columns if is_categorical_like(df[c])]
    return cat + [c for c in df.columns if c not in cat]


def levels(s: pd.Series) -> list:
    """依資料出現順序列出類別 (數值型類別則排序)。"""
    vals = s.dropna()
    if pd.api.types.is_numeric_dtype(vals):
        return sorted(vals.unique().tolist())
    if isinstance(vals.dtype, pd.CategoricalDtype):
        return [c for c in vals.cat.categories if c in set(vals)]
    return list(dict.fromkeys(vals.tolist()))


def as_category_labels(s: pd.Series) -> pd.Series:
    """把分組欄轉成字串標籤 (整數值不帶 .0)，缺失值保留為 NaN。"""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        vals = s.dropna()
        if len(vals) and np.allclose(vals, np.round(vals)):
            return s.map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    return s.map(lambda v: str(v) if pd.notna(v) else np.nan)


def guess_column(cols: list[str], patterns: list[str], exclude: tuple = ()) -> str | None:
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in cols:
            if c not in exclude and rx.search(str(c)):
                return c
    return None


def natural_key(v) -> tuple:
    parts = re.split(r"(\d+)", str(v))
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        rows.append({
            "欄位": c,
            "型別": "數值" if num else "類別/文字",
            "非缺失數": int(s.notna().sum()),
            "缺失數": int(s.isna().sum()),
            "不同值數": int(s.nunique()),
            "範例": ", ".join(map(str, s.dropna().unique()[:4])),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- transforms
QUICK_FUNCS = {
    "log2(x)": lambda x: np.log2(x.where(x > 0)),
    "log2(x+1)": lambda x: np.log2((x + 1).where(x > -1)),
    "log10(x)": lambda x: np.log10(x.where(x > 0)),
    "ln(x)": lambda x: np.log(x.where(x > 0)),
    "-log10(x)": lambda x: -np.log10(x.where(x > 0)),
    "√x": lambda x: np.sqrt(x.where(x >= 0)),
    "x²": lambda x: x ** 2,
    "1/x": lambda x: 1 / x.where(x != 0),
    "Z-score 標準化": lambda x: (x - x.mean()) / x.std(ddof=1),
    "Min-Max (0–1)": lambda x: (x - x.min()) / (x.max() - x.min()),
    "百分比 (佔總和 %)": lambda x: x / x.sum() * 100,
    "排名 (rank)": lambda x: x.rank(),
    "絕對值": lambda x: x.abs(),
}


def describe_transform(t: dict) -> str:
    from .i18n import _ as _t

    k = t["type"]
    if k == "melt":
        return _t("寬轉長：{n} 欄 → 「{var}」/「{val}」", n=len(t["value_vars"]), var=t["var_name"], val=t["value_name"])
    if k == "filter":
        return _t("篩選：{col} ∈ {vals}", col=t["column"], vals=f"{t['values'][:5]}{'…' if len(t['values']) > 5 else ''}")
    if k == "range":
        return _t("篩選：{lo} ≤ {col} ≤ {hi}", lo=t["lo"], col=t["column"], hi=t["hi"])
    if k == "query":
        return _t("條件：{expr}", expr=t["expr"])
    if k == "dropna":
        return _t("移除缺失值：{cols}", cols=", ".join(t["columns"]) or _t("全部欄位"))
    if k == "func":
        return _t("新增欄位 {new} = {f}", new=t["new"], f=t["func"].replace("x", t["column"]))
    if k == "expr":
        return _t("新增欄位 {new} = {f}", new=t["new"], f=t["expr"])
    if k == "astype":
        return _t("{col} 轉為數值", col=t["column"]) if t["to"] == "numeric" else _t("{col} 轉為類別", col=t["column"])
    if k == "rename":
        return _t("重新命名 {old} → {new}", old=t["old"], new=t["new"])
    if k == "recode":
        return _t("重新編碼 {col}（{n} 個值）", col=t["column"], n=len(t["mapping"]))
    if k == "transpose":
        return _t("轉置 (以「{idx}」為新欄名)", idx=t.get("index"))
    return k


def apply_transform(df: pd.DataFrame, t: dict) -> pd.DataFrame:
    k = t["type"]
    if k == "melt":
        id_vars = [c for c in t.get("id_vars", []) if c in df.columns]
        out = df.melt(id_vars=id_vars or None, value_vars=t["value_vars"], var_name=t["var_name"], value_name=t["value_name"])
        return out.dropna(subset=[t["value_name"]]).reset_index(drop=True)
    if k == "filter":
        keep = set(map(str, t["values"]))
        return df[df[t["column"]].map(str).isin(keep)].reset_index(drop=True)
    if k == "range":
        s = pd.to_numeric(df[t["column"]], errors="coerce")
        return df[(s >= t["lo"]) & (s <= t["hi"])].reset_index(drop=True)
    if k == "query":
        return df.query(t["expr"], engine="python").reset_index(drop=True)
    if k == "dropna":
        return df.dropna(subset=t["columns"] or None).reset_index(drop=True)
    if k == "func":
        out = df.copy()
        out[t["new"]] = QUICK_FUNCS[t["func"]](pd.to_numeric(out[t["column"]], errors="coerce"))
        return out
    if k == "expr":
        out = df.copy()
        out[t["new"]] = out.eval(t["expr"], engine="python")
        return out
    if k == "astype":
        out = df.copy()
        if t["to"] == "numeric":
            out[t["column"]] = pd.to_numeric(out[t["column"]], errors="coerce")
        else:
            out[t["column"]] = as_category_labels(out[t["column"]])
        return out
    if k == "rename":
        return df.rename(columns={t["old"]: t["new"]})
    if k == "recode":
        out = df.copy()
        mapping = t["mapping"]
        out[t["column"]] = out[t["column"]].map(lambda v: mapping.get(str(v), v) if pd.notna(v) else v)
        return out
    if k == "transpose":
        idx = t.get("index")
        base = df.set_index(idx) if idx in df.columns else df
        out = base.T.reset_index().rename(columns={"index": t.get("new_index", "Sample")})
        out.columns = [str(c) for c in out.columns]
        return clean_frame(out)
    raise ValueError(f"未知的轉換：{k}")


def apply_transforms(df: pd.DataFrame, transforms: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    for t in transforms:
        try:
            df = apply_transform(df, t)
        except Exception as err:  # 某步驟失敗時跳過並回報
            errors.append(f"{describe_transform(t)}：{err}")
    return df, errors


_P_COL = re.compile(r"(^|[\s_(])p($|[\s_)(])|p 值")


def is_p_col(name: str) -> bool:
    """欄名是否代表 p 值 (p、p_adj、Shapiro p、p (FDR)、p 值 (DeLong)…)。"""
    return bool(_P_COL.search(str(name).lower().strip()))


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """顯示用：p 值轉成易讀字串，其他浮點數取 4 位有效數字。下載時仍使用原始數值。"""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_float_dtype(s):
            if is_p_col(c):
                out[c] = s.map(lambda v: "" if pd.isna(v) else ("< 0.0001" if v < 1e-4 else f"{v:.4f}"))
            else:
                out[c] = s.map(lambda v: v if pd.isna(v) else float(f"{v:.4g}"))
    return out


def to_excel_bytes(sheets: dict[str, pd.DataFrame], index: bool = False) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, frame in sheets.items():
            safe = re.sub(r"[\[\]\*\?/\\:]", "_", str(name))[:31] or "Sheet"
            frame.to_excel(xw, sheet_name=safe, index=index)
    return buf.getvalue()


# ---------------------------------------------------------------- demo data
DEMO_DESCRIPTIONS = {
    "Groups": "四組實驗動物 (長格式)：盒鬚圖、長條圖、ANOVA",
    "Wide": "GraphPad 風格寬格式 (每組一欄)：示範寬轉長",
    "TimeCourse": "重複量測時間序列：折線圖、重複量測 ANOVA",
    "Paired": "治療前後配對資料：配對 t 檢定、Wilcoxon",
    "MethodCompare": "兩種檢測方法比對：Bland-Altman",
    "Volcano": "差異表現基因 (3000 基因)：火山圖",
    "ExprMatrix": "基因表現矩陣：熱圖、階層分群熱圖、PCA",
    "Survival": "存活資料：Kaplan-Meier、log-rank、Cox 迴歸",
    "Diagnostic": "診斷標記：ROC 曲線、AUC",
    "DoseResponse": "藥物劑量反應：4PL 曲線、EC50",
    "Forest": "次族群分析：森林圖",
    "GWAS": "全基因組關聯分析：Manhattan 圖",
    "Clinical": "臨床類別資料：卡方、Fisher、邏輯迴歸",
    "Microbiome": "菌相組成 (長格式)：百分比堆疊長條圖",
    "GeneSets": "三組差異基因清單：文氏圖",
}


def make_demo_data(seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}

    # Groups
    groups = ["Control", "DrugA", "DrugB", "DrugC"]
    n = 15
    g = np.repeat(groups, n)
    tumor_mu = {"Control": 520, "DrugA": 450, "DrugB": 330, "DrugC": 260}
    il6_mu = {"Control": 42, "DrugA": 35, "DrugB": 24, "DrugC": 20}
    tnf_mu = {"Control": 18, "DrugA": 17, "DrugB": 12, "DrugC": 9}
    ki_mu = {"Control": 48, "DrugA": 41, "DrugB": 30, "DrugC": 24}
    out["Groups"] = pd.DataFrame({
        "SubjectID": [f"M{i + 1:02d}" for i in range(len(g))],
        "Group": g,
        "Sex": rng.choice(["Male", "Female"], len(g)),
        "Age_weeks": rng.integers(8, 13, len(g)),
        "Weight_g": np.round(rng.normal(25, 2.5, len(g)), 1),
        "TumorVolume": np.round([rng.lognormal(np.log(tumor_mu[x]), 0.22) for x in g], 1),
        "IL6": np.round([rng.normal(il6_mu[x], 7) for x in g], 2),
        "TNFa": np.round([rng.normal(tnf_mu[x], 3.5) for x in g], 2),
        "Ki67_pct": np.round(np.clip([rng.normal(ki_mu[x], 8) for x in g], 1, 99), 1),
    })

    # Wide (unequal n)
    cols = {"Control": (10, 100, 12), "Drug A": (9, 82, 11), "Drug B": (11, 64, 10), "Drug C": (8, 58, 9)}
    wide = {k: pd.Series(np.round(rng.normal(mu, sd, nn), 1)) for k, (nn, mu, sd) in cols.items()}
    out["Wide"] = pd.DataFrame(wide)

    # TimeCourse
    rows = []
    for grp, slope in (("Vehicle", 1.6), ("Treatment", 0.55)):
        for s in range(10):
            base = rng.normal(100, 8)
            subj_slope = slope + rng.normal(0, 0.18)
            for day in (0, 7, 14, 21, 28):
                rows.append({"Subject": f"{grp[0]}{s + 1:02d}", "Group": grp, "Day": day,
                             "Response": round(base + subj_slope * day + rng.normal(0, 4), 2)})
    out["TimeCourse"] = pd.DataFrame(rows)

    # Paired
    before = rng.normal(145, 14, 24)
    out["Paired"] = pd.DataFrame({
        "PatientID": [f"P{i + 1:02d}" for i in range(24)],
        "Before": np.round(before, 1),
        "After": np.round(before - rng.normal(9, 8, 24), 1),
    })

    # MethodCompare
    true = rng.uniform(2, 12, 60)
    out["MethodCompare"] = pd.DataFrame({
        "Sample": [f"S{i + 1:02d}" for i in range(60)],
        "MethodA": np.round(true + rng.normal(0, 0.35, 60), 2),
        "MethodB": np.round(true * 1.03 + 0.15 + rng.normal(0, 0.4, 60), 2),
    })

    # Volcano
    ng = 3000
    fc = rng.normal(0, 0.45, ng)
    de = rng.random(ng) < 0.06
    fc[de] += rng.choice([-1, 1], de.sum()) * rng.uniform(1.0, 3.5, de.sum())
    z = np.abs(fc) / 0.35
    from scipy import stats as _st

    p = np.clip(2 * _st.norm.sf(z * rng.uniform(0.6, 1.4, ng)), 1e-300, 1)
    p[~de] = np.clip(rng.uniform(0, 1, (~de).sum()) ** 0.9, 1e-6, 1)
    order = np.argsort(p)
    ranked = p[order] * ng / (np.arange(ng) + 1)
    padj = np.empty(ng)
    padj[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    out["Volcano"] = pd.DataFrame({
        "Gene": [f"GENE{i + 1:04d}" for i in range(ng)],
        "log2FoldChange": np.round(fc, 3),
        "pvalue": p,
        "padj": np.clip(padj, 0, 1),
        "baseMean": np.round(rng.lognormal(6, 1.2, ng), 1),
    })

    # ExprMatrix
    genes = [f"Gene{i + 1:02d}" for i in range(30)]
    pathway = np.repeat(["Inflammation", "Cell cycle", "Apoptosis"], 10)
    samples = [f"Ctrl_{i}" for i in range(1, 6)] + [f"Trt_{i}" for i in range(1, 6)]
    base = rng.normal(8, 1.5, (30, 1))
    effect = np.zeros((30, 10))
    effect[:10, 5:] = rng.normal(2.0, 0.4, (10, 5))
    effect[10:20, 5:] = rng.normal(-1.6, 0.4, (10, 5))
    mat = np.round(base + effect + rng.normal(0, 0.45, (30, 10)), 2)
    expr = pd.DataFrame(mat, columns=samples)
    expr.insert(0, "Pathway", pathway)
    expr.insert(0, "Gene", genes)
    out["ExprMatrix"] = expr

    # Survival
    ns = 150
    grp = rng.choice(["Low risk", "Intermediate", "High risk"], ns, p=[0.35, 0.35, 0.3])
    rate = {"Low risk": 0.012, "Intermediate": 0.025, "High risk": 0.05}
    t_event = np.array([rng.exponential(1 / rate[x]) for x in grp])
    t_cens = rng.uniform(12, 72, ns)
    time = np.minimum(t_event, t_cens)
    age = np.round(rng.normal(62, 9, ns))
    out["Survival"] = pd.DataFrame({
        "PatientID": [f"PT{i + 1:03d}" for i in range(ns)],
        "RiskGroup": grp,
        "Treatment": rng.choice(["Standard", "Novel"], ns),
        "Age": age.astype(int),
        "Stage": rng.choice(["I", "II", "III", "IV"], ns, p=[0.2, 0.3, 0.3, 0.2]),
        "Time_months": np.round(time, 1),
        "Event": (t_event <= t_cens).astype(int),
    })

    # Diagnostic
    nd = 220
    disease = (rng.random(nd) < 0.4).astype(int)
    out["Diagnostic"] = pd.DataFrame({
        "SubjectID": [f"D{i + 1:03d}" for i in range(nd)],
        "Disease": disease,
        "MarkerA": np.round(rng.normal(5 + 2.2 * disease, 1.5), 2),
        "MarkerB": np.round(rng.normal(3 + 0.9 * disease, 1.2), 2),
        "MarkerC": np.round(rng.normal(10 + 0.3 * disease, 2.0), 2),
    })

    # DoseResponse
    conc = np.logspace(-3, 2, 9)
    rows = []
    for comp, (ec50, hill, top) in {"Compound X": (0.05, 1.2, 100), "Compound Y": (0.8, 1.0, 95), "Compound Z": (6.0, 1.5, 85)}.items():
        for c in conc:
            for _ in range(3):
                resp = 2 + (top - 2) / (1 + (ec50 / c) ** hill) + rng.normal(0, 3.5)
                rows.append({"Compound": comp, "Concentration_uM": float(f"{c:.4g}"), "Inhibition_pct": round(resp, 2)})
    out["DoseResponse"] = pd.DataFrame(rows)

    # Forest
    out["Forest"] = pd.DataFrame({
        "Subgroup": ["Overall", "Age < 65", "Age ≥ 65", "Male", "Female", "Stage I–II", "Stage III–IV", "Diabetes: Yes", "Diabetes: No"],
        "HR": [0.72, 0.65, 0.81, 0.70, 0.75, 0.58, 0.86, 0.92, 0.66],
        "Lower": [0.60, 0.50, 0.62, 0.55, 0.57, 0.41, 0.66, 0.65, 0.53],
        "Upper": [0.86, 0.85, 1.06, 0.89, 0.99, 0.82, 1.12, 1.30, 0.82],
        "N": [820, 430, 390, 450, 370, 400, 420, 210, 610],
    })

    # GWAS
    rows = []
    for chrom in range(1, 23):
        nsnp = int(260 - chrom * 7)
        bp = np.sort(rng.integers(1, int(250e6 - chrom * 8e6), nsnp))
        pv = rng.uniform(0, 1, nsnp)
        if chrom in (6, 11):
            peak = rng.integers(nsnp // 3, 2 * nsnp // 3)
            for j in range(max(peak - 12, 0), min(peak + 12, nsnp)):
                pv[j] = 10 ** -(rng.uniform(4, 10.5) * np.exp(-abs(j - peak) / 5))
        for b, pp in zip(bp, pv):
            rows.append({"SNP": f"rs{rng.integers(1e5, 9e7)}", "CHR": chrom, "BP": int(b), "P": float(max(pp, 1e-300))})
    out["GWAS"] = pd.DataFrame(rows)

    # Clinical
    nc = 180
    trt = rng.choice(["Drug", "Placebo"], nc)
    prob = np.where(trt == "Drug", 0.62, 0.38)
    age = np.round(rng.normal(58, 11, nc)).astype(int)
    improved = rng.random(nc) < np.clip(prob + (age - 58) * -0.006, 0.05, 0.95)
    outcome = np.where(improved, "Improved", rng.choice(["No change", "Worse"], nc, p=[0.65, 0.35]))
    out["Clinical"] = pd.DataFrame({
        "PatientID": [f"C{i + 1:03d}" for i in range(nc)],
        "Treatment": trt,
        "Sex": rng.choice(["Male", "Female"], nc),
        "Age": age,
        "Smoker": rng.choice(["Yes", "No"], nc, p=[0.3, 0.7]),
        "Outcome": outcome,
        "Responder": improved.astype(int),
    })

    # Microbiome
    taxa = ["Bacteroides", "Prevotella", "Faecalibacterium", "Bifidobacterium", "Lactobacillus", "Akkermansia", "Escherichia", "Others"]
    rows = []
    for grp, shift in (("Healthy", 0.0), ("Disease", 1.0)):
        for i in range(1, 6):
            alpha = np.array([30, 12, 18, 10, 6, 5, 3, 16], dtype=float)
            alpha *= np.array([1 + 0.4 * shift, 1, 1 - 0.5 * shift, 1 - 0.4 * shift, 1, 1 - 0.6 * shift, 1 + 2.5 * shift, 1])
            ab = rng.dirichlet(alpha * 3) * 100
            for tx, a in zip(taxa, ab):
                rows.append({"Sample": f"{grp[0]}{i}", "Group": grp, "Taxon": tx, "Abundance": round(a, 2)})
    out["Microbiome"] = pd.DataFrame(rows)

    # GeneSets
    universe = [f"GENE{i:04d}" for i in range(1, 801)]
    s1 = set(rng.choice(universe, 180, replace=False))
    s2 = set(rng.choice(sorted(s1), 70, replace=False)) | set(rng.choice(universe, 90, replace=False))
    s3 = set(rng.choice(sorted(s1 | s2), 50, replace=False)) | set(rng.choice(universe, 60, replace=False))
    lists = {"RNAseq_DEG": sorted(s1), "Proteomics_DEP": sorted(s2), "ChIP_targets": sorted(s3)}
    m = max(map(len, lists.values()))
    out["GeneSets"] = pd.DataFrame({k: pd.Series(v + [np.nan] * (m - len(v)), dtype=object) for k, v in lists.items()})

    return out


def demo_excel_bytes() -> bytes:
    return to_excel_bytes(make_demo_data())
