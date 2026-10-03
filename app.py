"""Arsenal explorer: MLB pitch labels vs the model, for pitchers the model never trained on.

Run:  streamlit run app.py   (after scripts/run_analysis.py has written data/processed/)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "test_predictions_seed0.parquet"

# Fixed hue per pitch type (identity follows the label, never its rank).
COLORS = {"FF": "#2a78d6", "SI": "#eb6834", "FC": "#1baf7a", "SL": "#eda100", "ST": "#e87ba4",
          "SV": "#008300", "CU": "#4a3aa7", "KC": "#e34948", "CH": "#52514e", "FS": "#8a6d3b"}
NAMES = {"FF": "4-seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider", "ST": "Sweeper",
         "SV": "Slurve", "CU": "Curveball", "KC": "Knuckle curve", "CH": "Changeup",
         "FS": "Splitter"}


@st.cache_data
def load() -> pd.DataFrame:
    return pd.read_parquet(DATA)


def movement_plot(df: pd.DataFrame, col: str, title: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    for lab, g in df.groupby(col):
        ax.scatter(g["hb_in"], g["ivb_in"], s=14, alpha=0.6, color=COLORS.get(lab, "#999"),
                   label=f"{lab} ({len(g)})", edgecolors="none")
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("Horizontal break, in (arm side +, pitcher's view)")
    ax.set_ylabel("Induced vertical break, in")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(color="#e4e3df")
    return fig


st.set_page_config(page_title="Pitch Types Are Relative", layout="wide")
st.title("Pitch types are relative")
st.caption("2025 MLB regular season (public Statcast). Every pitcher shown here was held out of "
           "training. Model: multinomial logistic regression on pitcher-relative features, "
           "the best of the models compared (see the README).")

if not DATA.exists():
    st.error("Run `python scripts/run_analysis.py` first to create the predictions file.")
    st.stop()

df = load()
by_p = (df.assign(agree=df["pred"] == df["label"])
        .groupby(["pitcher", "player_name"])
        .agg(n=("label", "size"), agreement=("agree", "mean")).reset_index())
by_p = by_p.sort_values("player_name")

sort = st.radio("List pitchers by", ["Name", "Most disagreement"], horizontal=True)
if sort == "Most disagreement":
    by_p = by_p.sort_values("agreement")
choice = st.selectbox("Pitcher", by_p.itertuples(index=False),
                      format_func=lambda r: f"{r.player_name}  ({r.n} pitches, "
                                            f"{r.agreement:.0%} agreement)")
p = df[df["pitcher"] == choice.pitcher]

c1, c2 = st.columns(2)
c1.pyplot(movement_plot(p, "label", "MLB label"))
c2.pyplot(movement_plot(p, "pred", "Model label"))

st.subheader("Where the model and MLB disagree")
dis = p[p["pred"] != p["label"]]
if dis.empty:
    st.write("No disagreements for this pitcher.")
else:
    tab = (dis.groupby(["label", "pred"]).agg(pitches=("velo", "size"),
           mean_velo=("velo", "mean"), mean_hb=("hb_in", "mean"), mean_ivb=("ivb_in", "mean"),
           model_confidence=("confidence", "mean")).reset_index()
           .rename(columns={"label": "MLB label", "pred": "Model label"}).round(1))
    st.dataframe(tab, hide_index=True)

with st.expander("Arsenal summary (MLB labels)"):
    summ = (p.groupby("label").agg(pitches=("velo", "size"), velo=("velo", "mean"),
            hb=("hb_in", "mean"), ivb=("ivb_in", "mean"), spin=("spin", "mean"),
            velo_gap=("d_velo", "mean")).round(1).reset_index())
    summ["label"] = summ["label"].map(lambda x: f"{x} · {NAMES.get(x, x)}")
    st.dataframe(summ, hide_index=True)
