"""Streamlit dashboard for the ATP prediction system.

Pages: match predictions (all three targets), backtest performance over time,
calibration, tournament bracket simulation, and current Elo ratings.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis.config import ARTIFACTS  # noqa: E402
from tennis.db.schema import connect  # noqa: E402
from tennis.models.common import calibration_table  # noqa: E402

st.set_page_config(page_title="ATP Prediction Model", layout="wide",
                   page_icon="🎾")

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
LEVELS = ["grand_slam", "masters", "atp500", "atp250", "finals", "davis_cup",
          "olympics", "challenger"]
ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]

# Fixed rather than exposed as sliders. 10k playthroughs put the Monte Carlo
# error on a title probability well inside a tenth of a point, so the knob only
# ever traded runtime for noise. The feed does not publish a draw more than
# about a day before play, so looking further than three days ahead returns the
# same tournaments regardless.
N_SIMS = 10_000
LOOK_AHEAD_DAYS = 3


def show_table(df: pd.DataFrame, container=None, row_classes=None) -> None:
    """Render a small table as static HTML.

    st.dataframe uses a canvas widget that measures its container on first
    render; inside an st.tabs() block the inactive tabs measure zero width and
    the table draws empty. These summary tables are small, so static HTML is
    both reliable and lighter.
    """
    target = container if container is not None else st
    if df is None or df.empty:
        target.info("No rows.")
        return
    def _fmt(v):
        if pd.isna(v):
            return "-"
        # thousands separator for magnitudes, 3dp for rates -- never sci notation
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.3f}".rstrip("0").rstrip(".")

    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(_fmt)
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].map(lambda v: f"{v:,}")
    if row_classes is None:
        target.markdown(
            out.to_html(index=False, escape=False, border=0,
                        classes="tp-table", na_rep="-"),
            unsafe_allow_html=True,
        )
        return

    # Built by hand rather than via to_html so individual rows can carry a
    # class -- used to dim players who are already out.
    import html as _h
    head = "".join(f"<th>{_h.escape(str(c))}</th>" for c in out.columns)
    body = []
    for cls, (_, r) in zip(row_classes, out.iterrows()):
        cells = "".join(f"<td>{'-' if pd.isna(v) else v}</td>" for v in r)
        body.append(f'<tr class="{cls}">{cells}</tr>')
    target.markdown(
        f'<table class="tp-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>',
        unsafe_allow_html=True,
    )


st.markdown(
    """<style>
    .tp-table {width:100%; border-collapse:collapse; font-size:0.88rem;}
    .tp-table th {text-align:left; padding:6px 10px; border-bottom:1px solid #444;
                  font-weight:600; opacity:0.8;}
    .tp-table td {padding:5px 10px; border-bottom:1px solid #2a2a2a;}
    .tp-table tr:hover td {background:rgba(255,255,255,0.04);}
    .tp-table tr.out td {opacity:0.42;}
    .tp-table tr.out td:first-child::after {content:" · out"; opacity:0.6;
                                            font-size:0.8em;}
    </style>""",
    unsafe_allow_html=True,
)



# --------------------------------------------------------------------------
# cached loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_players() -> pd.DataFrame:
    con = connect()
    df = pd.read_sql(
        "SELECT player_id, name, hand, height_cm, ioc, n_matches, last_seen "
        "FROM players ORDER BY n_matches DESC", con)
    con.close()
    return df


@st.cache_data(show_spinner=False)
def load_backtest() -> pd.DataFrame | None:
    p = ARTIFACTS / "backtest.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["season"] = df["tourney_date"] // 10000
    df["tour_level"] = np.where(df["is_challenger"] == 1, "Challenger", "Main tour")
    # ~940 source rows carry no surface; label them rather than showing a blank row
    df["surface"] = df["surface"].replace("", "Unknown").fillna("Unknown")
    return df


@st.cache_data(show_spinner=False)
def load_report() -> dict | None:
    p = ARTIFACTS / "backtest_report.json"
    return json.loads(p.read_text()) if p.exists() else None


ELO_SCOPES = ["overall", "Hard", "Clay", "Grass"]  # Carpet is retired


@st.cache_data(show_spinner=False)
def load_elo() -> pd.DataFrame:
    """One row per player, one column per surface rating.

    Surface columns carry the **blended** rating -- the number the model
    actually consumes -- not the raw value in `elo_state`. The book stores each
    surface rating on its own, and `EloBook.get` blends it toward the player's
    overall rating by surface match count on the way out. Showing the raw figure
    displays a number nothing in the system uses, and understates anyone light
    on a surface: on grass, 37 matches means a ~62% pull toward overall.

    Carpet is excluded: the surface was dropped from the tour after 2009, so its
    ratings are frozen a decade and a half out of date and only add noise to a
    like-for-like comparison.
    """
    con = connect()
    df = pd.read_sql(
        "SELECT e.player_id, p.name, e.scope, e.rating, e.n_matches, p.last_seen "
        "FROM elo_state e JOIN players p USING(player_id) "
        "WHERE e.scope IN ('overall','Hard','Clay','Grass')", con)
    con.close()
    if df.empty:
        return df
    wide = df.pivot_table(index=["player_id", "name", "last_seen"],
                          columns="scope", values="rating").reset_index()
    counts = df.pivot_table(index=["player_id"], columns="scope",
                            values="n_matches").reset_index()
    counts.columns = ["player_id"] + [f"n_{c}" for c in counts.columns[1:]]
    out = wide.merge(counts, on="player_id", how="left")
    for c in ELO_SCOPES:
        if c not in out.columns:
            out[c] = np.nan
        if f"n_{c}" not in out.columns:
            out[f"n_{c}"] = 0

    from tennis.features.elo import SURF_BLEND_N
    for c in ELO_SCOPES:
        if c == "overall":
            continue
        n = out[f"n_{c}"].fillna(0)
        w = n / (n + SURF_BLEND_N)
        # Where a player has never played the surface there is no stored rating
        # and the blend is the overall figure outright.
        out[c] = w * out[c].fillna(out["overall"]) + (1 - w) * out["overall"]
    return out.rename_axis(None, axis=1)


SEASON_CUT = (12, 25)  # month, day


def season_start(date: int) -> int:
    """First day of the ATP season containing `date`, as YYYYMMDD.

    The tour year does not follow the calendar: the season opens in the last
    days of December, so the 2026 season begins in 2025.

    The cut is 25 December, and the date is not arbitrary. December holds
    events at both ends of the tour year -- the Next Gen Finals close a season
    (18 Dec 2024, 22 Dec 2025) while Doha, Adelaide, Chennai, Brisbane, Pune,
    Hong Kong and the United Cup open the next one (26-31 Dec). Across all 26
    seasons in the database, 20-21 and 23-25 December are completely empty:
    the last season-ending event is the 22nd, the earliest opener the 26th.
    A cut of 25 December therefore separates them with room on both sides,
    where 1 December -- tried first -- swept the Next Gen Finals into the
    season it actually closes.
    """
    y, m, d = date // 10000, (date // 100) % 100, date % 100
    m0, d0 = SEASON_CUT
    if (m, d) >= (m0, d0):
        return y * 10000 + m0 * 100 + d0
    return (y - 1) * 10000 + m0 * 100 + d0


def season_label(start: int) -> int:
    """The season a start date belongs to -- December 2025 opens season 2026."""
    return start // 10000 + 1


@st.cache_data(show_spinner="Rebuilding ratings to the season opener…")
def load_elo_season() -> pd.DataFrame:
    """Season-to-date *movement* of the all-time rating, plus season W-L.

    Not a rating restarted at the season opener, which is what this began as
    and what the question usually asks for. A restarted book cannot work over
    one season: with everyone reset to 1500, beating a field is worth the same
    whoever the field is, and Challenger and main-tour players almost never
    play each other, so the two pools never get priced against one another. The
    measured result was a 38-11 Challenger season (median opponent rank 242)
    rating above a 27-7 season against the top 60. That is the book failing to
    tell the pools apart, not a finding about the players.

    Movement of the all-time book has neither problem. It starts from a rating
    that already knows what a Challenger win is worth, so beating rank 242
    *when you are rated to beat rank 242* moves nothing, and the column reads
    as genuine over- or under-performance against expectation.
    """
    from tennis.features.elo import EloBook

    con = connect()
    try:
        end = con.execute("SELECT MAX(tourney_date) FROM matches").fetchone()[0]
        if end is None:
            return pd.DataFrame()
        start = season_start(int(end))
        prior = pd.read_sql(
            """SELECT m.winner_id, m.loser_id, m.tourney_date, t.surface, m.status
               FROM matches m JOIN tournaments t USING(tourney_key)
               WHERE m.tourney_date < ? ORDER BY m.seq""", con, params=(start,))
        season = pd.read_sql(
            """SELECT m.winner_id, m.loser_id, m.status
               FROM matches m WHERE m.tourney_date >= ?""", con, params=(start,))
    finally:
        con.close()
    if season.empty:
        return pd.DataFrame()

    # Replaying only the matches before the opener leaves the book holding
    # exactly the state it had on day one of the season.
    prior = prior[prior["status"] != "walkover"]
    book = EloBook()
    for w, l, sf, d in zip(prior["winner_id"], prior["loser_id"],
                           prior["surface"], prior["tourney_date"]):
        day = _ord(int(d))
        book.update(w, l, "overall", day=day)
        if isinstance(sf, str) and sf in ELO_SCOPES:
            book.update(w, l, sf, day=day)
        book.touch(w, day)
        book.touch(l, day)

    season = season[season["status"] != "walkover"]
    wins = season.groupby("winner_id").size()
    losses = season.groupby("loser_id").size()
    ids = sorted(set(season["winner_id"]) | set(season["loser_id"]))
    day0 = _ord(start)
    return pd.DataFrame({
        "player_id": ids,
        "elo_start": [book.get(p, "overall", day0) for p in ids],
        "season_w": [int(wins.get(p, 0)) for p in ids],
        "season_l": [int(losses.get(p, 0)) for p in ids],
        "season_start": start,
    })


def _ord(d: int) -> int:
    return pd.Timestamp(str(int(d))).toordinal()


@st.cache_resource(show_spinner=False)
def get_predictor():
    from tennis.models.predict import Predictor
    return Predictor()


def models_ready() -> bool:
    return (ARTIFACTS / "models.pkl").exists()


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------
def page_predictions() -> None:
    st.header("Match predictions")
    st.caption(
        "Enter any matchup. The upstream ongoing-tournament files publish only "
        "completed matches, so fixtures are entered by hand rather than scraped."
    )
    if not models_ready():
        st.warning("No trained models found. Run `python -m tennis.models.train` first.")
        return

    players = load_players()
    names = players["name"].dropna().tolist()
    name_to_id = dict(zip(players["name"], players["player_id"]))

    c1, c2 = st.columns(2)
    p1 = c1.selectbox("Player 1", names, index=0)
    p2 = c2.selectbox("Player 2", names, index=min(1, len(names) - 1))

    c3, c4, c5, c6 = st.columns(4)
    surface = c3.selectbox("Surface", SURFACES, index=0)
    level = c4.selectbox("Level", LEVELS, index=3)
    best_of = c5.selectbox("Best of", [3, 5], index=0)
    rnd = c6.selectbox("Round", ROUNDS, index=2)
    indoor = st.checkbox("Indoor", value=False)

    if st.button("Predict", type="primary"):
        if p1 == p2:
            st.error("Pick two different players.")
            return
        from tennis.models.predict import MatchContext
        from tennis.config import ROUND_ORDER
        ctx = MatchContext(
            surface=surface, level=level, best_of=best_of,
            indoor=1.0 if indoor else 0.0, round=rnd,
            round_idx=float(ROUND_ORDER.index(rnd)),
            is_challenger=int(level == "challenger"),
        )
        pred = get_predictor().predict_many([(name_to_id[p1], name_to_id[p2])], ctx)
        r = pred.iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{p1} win", f"{r['p1_win_prob']*100:.1f}%")
        m2.metric(f"{p2} win", f"{(1-r['p1_win_prob'])*100:.1f}%")
        m3.metric("Total games", f"{r['pred_total_games']:.1f}")
        m4.metric("Spread (P1)", f"{r['pred_spread']:+.1f}")
        st.caption(
            f"Projected sets: {r['pred_total_sets']:.2f} · "
            f"Fair decimal odds — {p1}: {1/max(r['p1_win_prob'],1e-6):.2f}, "
            f"{p2}: {1/max(1-r['p1_win_prob'],1e-6):.2f}"
        )

        fig = go.Figure(go.Bar(
            x=[r["p1_win_prob"], 1 - r["p1_win_prob"]], y=[p1, p2],
            orientation="h", marker_color=["#2E86AB", "#A23B72"],
            text=[f"{r['p1_win_prob']*100:.1f}%", f"{(1-r['p1_win_prob'])*100:.1f}%"],
        ))
        fig.update_layout(height=200, xaxis=dict(range=[0, 1], tickformat=".0%"),
                          showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")


def page_backtest() -> None:
    st.header("Backtest performance")
    bt = load_backtest()
    if bt is None:
        st.warning("No backtest found. Run `python -m tennis.models.train`.")
        return
    rep = load_report()

    st.caption(
        f"Walk-forward, split strictly by date: each season is predicted by a "
        f"model trained only on earlier matches. "
        f"{len(bt):,} matches, {bt['season'].min()}–{bt['season'].max()}."
    )

    y, p = bt["y_win"].to_numpy(), bt["p_win"].to_numpy()
    from tennis.models.common import brier, log_loss
    acc = float(((p >= 0.5).astype(int) == y).mean())
    c = st.columns(5)
    c[0].metric("Matches", f"{len(bt):,}")
    c[1].metric("Log loss", f"{log_loss(y, p):.4f}")
    c[2].metric("Brier", f"{brier(y, p):.4f}")
    c[3].metric("Accuracy", f"{acc*100:.2f}%")
    base = bt["elo_prob"].dropna()
    if len(base):
        c[4].metric("Elo baseline log loss",
                    f"{log_loss(bt.loc[base.index,'y_win'].to_numpy(), base.to_numpy()):.4f}")

    # per-season trend
    per = bt.groupby("season").apply(
        lambda g: pd.Series({
            "log_loss": log_loss(g["y_win"].to_numpy(), g["p_win"].to_numpy()),
            "brier": brier(g["y_win"].to_numpy(), g["p_win"].to_numpy()),
            "accuracy": float(((g["p_win"] >= 0.5).astype(int) == g["y_win"]).mean()),
            "n": len(g),
        }), include_groups=False).reset_index()

    tab1, tab2, tab3 = st.tabs(["Over time", "Breakouts", "Market"])
    with tab1:
        fig = px.line(per, x="season", y=["log_loss", "brier"], markers=True,
                      title="Winner model error by season")
        st.plotly_chart(fig, width="stretch")
        fig2 = px.bar(per, x="season", y="accuracy", title="Accuracy by season",
                      range_y=[0.5, 0.8])
        st.plotly_chart(fig2, width="stretch")

        tot = bt.dropna(subset=["y_total"]).groupby("season").apply(
            lambda g: pd.Series({
                "totals_mae": float(np.abs(g["pred_total"] - g["y_total"]).mean()),
                "spread_mae": float(np.abs(g["pred_spread"] - g["y_spread"]).mean()),
            }), include_groups=False).reset_index()
        st.plotly_chart(px.line(tot, x="season", y=["totals_mae", "spread_mae"],
                                markers=True, title="Totals / spread MAE (games)"),
                        width="stretch")

    with tab2:
        for dim, label in [("surface", "Surface"), ("tour_level", "Main tour vs Challenger"),
                           ("level", "Tournament level")]:
            st.subheader(label)
            rows = []
            for k, g in bt.groupby(dim, observed=True):
                if len(g) < 30:
                    continue
                gt = g.dropna(subset=["y_total"])
                rows.append({
                    dim: k, "n": len(g),
                    "log_loss": round(log_loss(g["y_win"].to_numpy(), g["p_win"].to_numpy()), 4),
                    "brier": round(brier(g["y_win"].to_numpy(), g["p_win"].to_numpy()), 4),
                    "accuracy": round(float(((g["p_win"] >= 0.5).astype(int) == g["y_win"]).mean()), 4),
                    "totals_mae": round(float(np.abs(gt["pred_total"] - gt["y_total"]).mean()), 3) if len(gt) > 30 else None,
                    "spread_mae": round(float(np.abs(gt["pred_spread"] - gt["y_spread"]).mean()), 3) if len(gt) > 30 else None,
                })
            show_table(pd.DataFrame(rows).sort_values("n", ascending=False))
        st.info("ATP only for now. The WTA breakout appears here once WTA data is loaded — "
                "the pipeline already carries a `tour` column throughout.")

    with tab3:
        if not (rep and "roi_winner" in rep):
            st.info("Run `python -m tennis.models.evaluate` to generate the ROI report.")
            return

        st.subheader("Winner model — vs real closing prices")
        st.caption("Pinnacle closing odds from tennis-data.co.uk, flat 1u stakes.")
        # Built row-by-row rather than by transposing a dict-of-dicts: the
        # transpose yields object-dtype columns that Streamlit's Arrow
        # serialiser silently drops, leaving a table with only the index.
        win_roi = pd.DataFrame([
            {
                "edge": k.replace("edge_", ""),
                "bets": int(v.get("bets", 0)),
                "profit (u)": f"{v.get('profit', 0):+,.1f}",
                "ROI %": f"{v.get('roi_pct', 0):+.2f}",
                "hit %": f"{v.get('hit_rate', float('nan')) * 100:.2f}",
                "break-even %": f"{v.get('breakeven_hit_rate', float('nan')) * 100:.2f}",
                "avg price": f"{v.get('avg_price', float('nan')):.2f}",
                "% on underdog": f"{v.get('pct_on_market_underdog', float('nan')):.1f}",
            }
            for k, v in rep["roi_winner"].items() if v.get("bets")
        ])
        show_table(win_roi)

        _hit_rate_explainer(rep)

        ms = rep.get("market_subset", {})
        if ms:
            cc = st.columns(3)
            cc[0].metric("Priced matches", f"{ms['n']:,}")
            cc[1].metric("Model log loss", f"{ms['model']['log_loss']:.4f}")
            cc[2].metric("Closing-line log loss", f"{ms['market_closing']['log_loss']:.4f}")
        hl = rep.get("headline", {})
        if hl.get("verdict"):
            st.error(hl["verdict"] if not hl.get("winner_beats_closing_line")
                     else hl["verdict"])

        _roi_by_season(rep)

        _market_lines(rep)


def _hit_rate_explainer(rep: dict) -> None:
    """Why hit rate sits below 50% while accuracy is 66%.

    They answer different questions on different rows, and the reflex to read
    a betting hit rate against 50% is wrong -- the benchmark is the price.
    """
    v = rep["roi_winner"].get("edge_0.03") or {}
    acc = rep.get("winner", {}).get("accuracy")
    if not v.get("bets"):
        return
    hit, be = v["hit_rate"] * 100, v.get("breakeven_hit_rate", 0) * 100
    with st.expander("Why is hit rate under 50% when accuracy is 66%?"):
        st.markdown(
            f"""
They measure different things, on different rows.

**Accuracy ({acc:.1%})** asks: across all {rep['n_matches']:,} matches, did the
player the model *favoured* win? The favourite wins most tennis matches, so a
high number here is expected — the Elo baseline alone gets most of the way.

**Hit rate ({hit:.1f}%)** asks something else entirely: on the
{v['bets']:,} matches where the model disagreed with the price by more than the
edge threshold, did the *value side* win? Those are not the same bet. Value
lives where the market is generous, and the market is generous on underdogs —
**{v.get('pct_on_market_underdog', 0):.0f}% of these bets are on the market
underdog**, at an average price of **{v['avg_price']:.2f}**. Betting longshots,
you are *supposed* to lose most of them. Winning half would be extraordinary.

**So 50% is the wrong benchmark. The benchmark is the price.** At these odds
the break-even hit rate is **{be:.1f}%** — that is what the flat stakes need
just to return the money. The model hits **{hit:.1f}%**.

That gap of **{be - hit:.1f} points** is the entire story, and it is what the
{v['roi_pct']:.2f}% ROI is measuring. The model is not missing wildly; it is
landing just short of the price, consistently. Which is the ordinary result:
the closing line is a strong forecast and beating it is hard.

One caveat on reading the table: compare `hit_rate` to `break-even` on the same
row, never across rows. Raising the edge threshold selects longer prices, so
both columns fall together — a lower hit rate at edge 0.10 is not a worse model.
"""
        )



def _market_lines(rep: dict) -> None:
    """The three models against their own markets, plus closing-line value.

    Precomputed by `models/lines.py` during evaluation and read from the report:
    it rests on 15M+ rows of `odds_quotes`, which the dashboard must never touch
    at render time. A missing block means the tennisexplorer backfill has not
    run, which is a normal state rather than an error.
    """
    ml = rep.get("market_lines")
    if not ml or not ml.get("markets"):
        st.subheader("Totals & spread — synthetic line, not a market")
        st.info(
            "No games-market comparison in this report. Run the tennisexplorer "
            "backfill (`tennis/ingest/odds_te_hist.py`) and re-run evaluation to "
            "replace this with the real thing."
        )
        return

    st.subheader("All three models against their own market")
    ms = rep.get("market_subset", {})
    rows = []
    if ms:
        rows.append({
            "market": "winner (moneyline)", "matches": f"{ms['n']:,}",
            "model": f"{ms['model']['log_loss']:.5f}",
            "market_": f"{ms['market_closing']['log_loss']:.5f}",
            "gap": f"{ms['model']['log_loss'] - ms['market_closing']['log_loss']:+.5f}",
            "vig %": "—", "source": "Pinnacle close",
        })
    for name, m in ml["markets"].items():
        rows.append({
            "market": f"{name} (games)", "matches": f"{m['matches']:,}",
            "model": f"{m['model_ll']:.5f}", "market_": f"{m['market_ll']:.5f}",
            "gap": f"{m['gap']:+.5f}", "vig %": f"{m['median_vig_pct']:.1f}",
            "source": f"books avg, {m['seasons'][0]}–{m['seasons'][1]}",
        })
    show_table(pd.DataFrame(rows).rename(columns={"market_": "market"}))
    st.caption(
        "Log loss; **gap** is model minus market, so positive means the market "
        "is better. The point is that the three gaps are *similar* — no single "
        "model is the weak link. Totals and spread rest on the tennisexplorer "
        "backfill, which covers fewer matches and fewer seasons than the winner "
        "row, so the counts are not comparable."
    )

    tiers = {n: m.get("by_tier", {}) for n, m in ml["markets"].items()}
    if any(tiers.values()):
        st.markdown(
            "**By tier.** The totals gap is five times smaller on Challengers "
            "(0.004 against 0.018) — the smallest gap anywhere in this project. "
            "Spread shows no such split: the two tiers are within 0.00004 of "
            "each other, so whatever makes Challenger *totals* easier to price "
            "relative to the market does not carry over to the handicap.")
        trows = []
        for name, by in tiers.items():
            for tier, v in by.items():
                trows.append({"market": f"{name} (games)", "tier": tier,
                              "rows": f"{v['rows']:,}",
                              "model": f"{v['model_ll']:.5f}",
                              "market ": f"{v['market_ll']:.5f}",
                              "gap": f"{v['gap']:+.5f}"})
        show_table(pd.DataFrame(trows))

    c = ml.get("clv", {})
    if c.get("markets"):
        st.subheader("Closing-line value")
        crows = []
        for name, v in c["markets"].items():
            crows.append({
                "market": f"{name} (games)", "bets": f"{v['bets']:,}",
                "mean CLV (pts)": f"{v['mean_clv_pts']:+.3f}",
                "beat close %": f"{v['beat_close_pct']:.1f}",
                "z": f"{v['z']:+.1f}",
                "vig to overcome (pts)": f"{v['vig_to_overcome_pts']:.1f}",
            })
        show_table(pd.DataFrame(crows))
        st.warning(
            "**Signal, not profit — and the last column is why.** These bets are "
            "placed at "
            f"{c.get('book','Pinnacle')}'s *opening* line and scored against the "
            "close, one bet per match. The value is real and statistically "
            "overwhelming: the market moves toward us more often than not. But "
            "the margin on those same lines is 3–4 points, so the edge covers "
            "roughly a tenth of what totals needs and a third of what spread "
            "needs. Read the CLV column and the vig column together or not at "
            "all. Some of the value is also likely *timing* rather than "
            "modelling — our features are as-of-match while an opening line can "
            "be days old."
        )

def _roi_by_season(rep: dict) -> None:
    blocks = rep.get("roi_winner_by_season")
    if not blocks:
        st.caption("Re-run `python -m tennis.models.evaluate` for the season split.")
        return

    st.subheader("ROI by season")
    edge = st.selectbox("Edge threshold", list(blocks.keys()),
                        index=min(1, len(blocks) - 1),
                        format_func=lambda k: k.replace("edge_", "edge > "))
    rows = blocks.get(edge) or []
    if not rows:
        st.info("No bets at this threshold.")
        return

    s = pd.DataFrame(rows)
    # Thin seasons swing wildly on a handful of bets -- the current one is a few
    # weeks old. Charting them wrecks the scale: a -44% bar off 47 bets flattens
    # sixteen real seasons into a sliver. They stay in the table, dimmed.
    thin = s["bets"] < 200
    full = s[~thin]
    fig = px.bar(full, x="season", y="roi_pct",
                 color=np.where(full["roi_pct"] >= 0, "profit", "loss"),
                 color_discrete_map={"profit": "#7bbf94", "loss": "#d9534f"},
                 title="Flat-stake ROI by season (%)")
    fig.add_hline(y=0, line_width=1, line_color="#888")
    fig.update_layout(showlegend=False, height=380, xaxis_title=None,
                      yaxis_title="ROI %")
    st.plotly_chart(fig, width="stretch")

    if len(full) > 1:
        pos = int((full["roi_pct"] > 0).sum())
        c = st.columns(4)
        best = full.loc[full["roi_pct"].idxmax()]
        worst = full.loc[full["roi_pct"].idxmin()]
        c[0].metric("Seasons profitable", f"{pos} of {len(full)}")
        # Year goes in the label, not the delta slot: a delta renders with an
        # arrow, and "↑ 2023" under the worst ROI on the board reads as a gain.
        c[1].metric(f"Best · {int(best['season'])}", f"{best['roi_pct']:+.1f}%")
        c[2].metric(f"Worst · {int(worst['season'])}", f"{worst['roi_pct']:+.1f}%")
        c[3].metric("Season spread (sd)", f"{full['roi_pct'].std():.1f} pts")

    # Formatted as strings: show_table gives numeric columns a thousands
    # separator, which turns a year into "2,010", and trims trailing zeros,
    # which leaves a column reading -4 next to -12.47.
    disp = pd.DataFrame({
        "Season": s["season"].astype(int).astype(str),
        "Bets": s["bets"].astype(int),
        "Profit (u)": s["profit"].map(lambda v: f"{v:+,.1f}"),
        "ROI %": s["roi_pct"].map(lambda v: f"{v:+.2f}"),
        "Hit %": (s["hit_rate"] * 100).map(lambda v: f"{v:.1f}"),
        "Break-even %": (s["breakeven_hit_rate"] * 100).map(lambda v: f"{v:.1f}"),
        "Avg price": s["avg_price"].map(lambda v: f"{v:.2f}"),
    })
    show_table(disp, row_classes=["out" if t else "" for t in thin])
    st.caption(
        "Dimmed seasons carry under 200 bets and are left out of the chart and "
        "the summary — the current season is only a few weeks old, and ROI on a "
        "few dozen bets is noise. Each season is priced by its own market, so "
        "the spread across seasons is the honest read on how much weight the "
        "single headline figure carries."
    )


def page_calibration() -> None:
    st.header("Calibration")
    bt = load_backtest()
    if bt is None:
        st.warning("No backtest found.")
        return
    seasons = sorted(bt["season"].unique())
    sel = st.multiselect("Seasons", seasons, default=seasons)
    sub = bt[bt["season"].isin(sel)] if sel else bt
    if sub.empty:
        st.info("No matches selected.")
        return

    y, p = sub["y_win"].to_numpy(), sub["p_win"].to_numpy()
    tbl = calibration_table(y, p, bins=10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="grey"), name="Perfect"))
    fig.add_trace(go.Scatter(x=tbl["pred_mean"], y=tbl["actual"], mode="markers+lines",
                             marker=dict(size=8), name="Model",
                             text=tbl["n"], hovertemplate="pred %{x:.3f}<br>actual %{y:.3f}<br>n=%{text}"))
    fig.update_layout(xaxis_title="Predicted probability", yaxis_title="Observed win rate",
                      height=520, xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(tbl, width="stretch", hide_index=True)

    st.subheader("Prediction distribution")
    st.plotly_chart(px.histogram(sub, x="p_win", nbins=20), width="stretch")


BRACKET_CSS = """<style>
.bkt {display:flex; gap:22px; overflow-x:auto; padding:6px 2px 14px;
      --bline:#4a4a4a; --qline:#4d4d4d; --hline:#7a7a7a; --tgap:6px;}
.bkt-col {min-width:230px; display:flex; flex-direction:column;}
.bkt-head {font-weight:600; font-size:0.82rem; opacity:0.75; letter-spacing:.04em;
           text-transform:uppercase; margin-bottom:8px; position:sticky; top:0;}
.bkt-ties {display:flex; flex-direction:column; justify-content:space-around;
           flex:1; gap:var(--tgap);}
/* Quarter rules. With ties distributed evenly down the column, the boundary
   between quarters lands at exactly 25/50/75% of the column in every round --
   for 64 ties the gap between the 16th and 17th sits at 25%, for 4 ties the
   gap between the 1st and 2nd sits there too -- so three background lines
   line up across the whole bracket without measuring anything. The middle
   rule is brighter: that one splits the draw into the two halves that can
   only meet in the final. */
.bkt-ties.q4 {background-image:
      linear-gradient(var(--qline),var(--qline)),
      linear-gradient(var(--hline),var(--hline)),
      linear-gradient(var(--qline),var(--qline));
   background-size:100% 1px; background-repeat:no-repeat;
   background-position:0 25%, 0 50%, 0 75%;}
.bkt-ties.q2 {background-image:linear-gradient(var(--hline),var(--hline));
   background-size:100% 1px; background-repeat:no-repeat; background-position:0 50%;}
/* A pair of ties feeding one slot in the next round. Both ties sit at 25% and
   75% of the pair, so the elbow joining them spans exactly that. */
.pair {flex:1; display:flex; flex-direction:column; justify-content:space-around;
       gap:var(--tgap); position:relative;}
.bkt.link .bkt-col:not(:last-child) .pair::after {content:""; position:absolute;
       right:-11px; border-right:1px solid var(--bline);
       /* space-around splits the free space as F/4 either side of the gap, so a
          tie's centre sits gap/4 above the flat 25% mark. Correcting by that
          lands the elbow exactly on the stubs instead of 1.5px adrift. */
       top:calc(25% - var(--tgap) / 4); bottom:calc(25% - var(--tgap) / 4);}
.tw {position:relative; display:flex; flex-direction:column;}
.bkt.link .bkt-col:not(:last-child) .tw::after {content:""; position:absolute;
       right:-11px; top:50%; width:11px; border-top:1px solid var(--bline);}
.bkt.link .bkt-col:not(:first-child) .tw::before {content:""; position:absolute;
       left:-11px; top:50%; width:11px; border-top:1px solid var(--bline);}
.tie {border:1px solid #333; border-radius:6px; overflow:hidden;
      background:rgba(255,255,255,0.02);}
.tie .p {display:flex; justify-content:space-between; gap:8px; padding:4px 8px;
         font-size:0.82rem; border-bottom:1px solid #262626;}
.tie .p:last-child {border-bottom:none;}
.tie .p .nm {white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.tie .p .pc {opacity:0.85; font-variant-numeric:tabular-nums;}
.tie .p .sets {margin-left:auto; display:flex; gap:5px;
               font-variant-numeric:tabular-nums; opacity:0.9;}
.tie .p .sets i {font-style:normal; min-width:0.75rem; text-align:right;}
.tie .p .sets sup {font-size:0.62em; opacity:0.75; margin-left:1px;}
.tie.played .win .sets {font-weight:600; opacity:1;}
.tie.played {border-color:#3d5a45;}
.tie.played .win {background:rgba(76,159,112,0.18); font-weight:600;}
.tie.played .lose {opacity:0.45;}
.tie.live {border-color:#2E86AB;}
.tie.live .fav {font-weight:600;}
.tie.pending {border-style:dashed; border-color:#3a3a3a; opacity:0.55;}
.tie .meta {padding:2px 8px; font-size:0.7rem; opacity:0.65;
            border-top:1px solid #262626;}
.tie .meta.ok    {color:#7bbf94;}
.tie .meta.near  {color:#c9b458;}   /* wrong, but near a coin flip */
.tie .meta.miss  {color:#d08a5a;}
.tie .meta.upset {color:#d9534f; font-weight:600;}
.tie .meta.muted {opacity:0.4; font-style:italic;}
/* Holds the row's height without showing anything, so ties stay a uniform
   height and the connecting elbows land on their stubs. */
.tie .meta.empty {visibility:hidden;}
.tie .meta.empty::before {content:"\00a0";}
</style>"""


def _render_bracket(draw, rounds, show_rounds):
    """Bracket as columns of ties: results on the left, TBD on the right."""
    import html as _html

    from tennis.ingest.parse import score_sets

    def nm(p):
        from tennis.sim.bracket import BYE
        if p == BYE:
            return "bye"
        if p is None:
            return "TBD"
        return _html.escape(str(draw.player_names.get(p, p)))

    cols = []
    for rd in rounds:
        if rd["round"] not in show_rounds:
            continue
        ties_html = []
        for tie in rd["ties"]:
            from tennis.sim.bracket import BYE
            if tie["p1"] == BYE and tie["p2"] == BYE:
                continue
            state = tie["state"]
            prob = tie.get("p1_win_prob")
            if state == "played":
                w = tie["winner"]
                wp = tie.get("winner_prob")
                pcts = [prob, 1 - prob] if prob is not None else [None, None]
                # score_sets returns winner-first, so flip when p1 lost
                sets = score_sets(tie.get("score"))
                cells = {tie["p1"]: [], tie["p2"]: []}
                loser = tie["p2"] if w == tie["p1"] else tie["p1"]
                for wg, lg, tb in sets:
                    # The tiebreak figure belongs to whoever lost that set,
                    # shown as a superscript beside their games.
                    hi, lo = (w, loser) if wg >= lg else (loser, w)
                    cells[hi].append(f"{max(wg, lg)}")
                    cells[lo].append(
                        f"{min(wg, lg)}<sup>{tb}</sup>" if tb is not None
                        else f"{min(wg, lg)}")
                rows = "".join(
                    f'<div class="p {"win" if p == w else "lose"}">'
                    f'<span class="nm">{nm(p)}</span>'
                    + (f'<span class="sets">{"".join(f"<i>{c}</i>" for c in cells[p])}</span>'
                       if cells.get(p) else "")
                    + (f'<span class="pc">{pc*100:.0f}%</span>' if pc is not None else "")
                    + '</div>'
                    for p, pc in zip((tie["p1"], tie["p2"]), pcts))
                if wp is None:
                    # No prediction to grade -- a bye, or a walkover with no
                    # pre-match price. Say so rather than leaving the row
                    # blank: every tie then has the same height, which is what
                    # keeps the connecting elbows meeting their ties exactly.
                    meta = '<div class="meta muted">no match played</div>'
                else:
                    ok = tie.get("correct")
                    # Shade by how wrong, not merely whether wrong: a 48% miss
                    # is a coin flip, a 12% miss is a real upset.
                    cls = ("ok" if ok else
                           "near" if wp >= 0.40 else
                           "miss" if wp >= 0.25 else "upset")
                    mark = "\u2713" if ok else "\u2717"
                    meta = (f'<div class="meta {cls}">{mark} model gave winner '
                            f'{wp*100:.0f}%</div>')
            elif state == "live" and prob is not None:
                pcts = [prob, 1 - prob]
                rows = "".join(
                    f'<div class="p {"fav" if pc >= 0.5 else ""}">'
                    f'<span class="nm">{nm(p)}</span>'
                    f'<span class="pc">{pc*100:.0f}%</span></div>'
                    for p, pc in zip((tie["p1"], tie["p2"]), pcts))
                tg, sp = tie.get("total_games"), tie.get("spread")
                bits = []
                if tg is not None:
                    bits.append(f"total {tg:.1f} games")
                if sp is not None:
                    # Quoted the way a handicap is: the favoured player carries
                    # the negative number. `spread` is p1 games minus p2 games.
                    fav = tie["p1"] if sp >= 0 else tie["p2"]
                    bits.append(f"{nm(fav)} −{abs(sp):.1f}")
                meta = (f'<div class="meta">{" · ".join(bits)}</div>'
                        if bits else '<div class="meta empty"></div>')
            else:
                rows = "".join(
                    f'<div class="p"><span class="nm">{nm(p)}</span></div>'
                    for p in (tie["p1"], tie["p2"]))
                meta = '<div class="meta empty"></div>' 
            ties_html.append(
                f'<div class="tw"><div class="tie {state}">{rows}{meta}</div></div>')
        # Ties are wrapped two at a time: the pair that feeds one slot in the
        # next round. That grouping is what the connecting elbow is drawn from,
        # and it leaves the vertical spacing identical to the flat layout.
        pairs = "".join(f'<div class="pair">{"".join(ties_html[i:i + 2])}</div>'
                        for i in range(0, len(ties_html), 2))
        # Quarter rules need four ties to divide; a two-tie semi-final gets the
        # halfway rule alone, and the final gets neither.
        q = "q4" if len(ties_html) >= 4 else "q2" if len(ties_html) == 2 else ""
        cols.append(f'<div class="bkt-col"><div class="bkt-head">{rd["round"]}'
                    f'</div><div class="bkt-ties {q}">{pairs}</div></div>')

    # Connectors are only drawn when the visible rounds are consecutive. With
    # a round filtered out of the middle, a line from R32 to the semi-finals
    # would claim a progression that skips a round.
    order = [rd["round"] for rd in rounds]
    shown = [i for i, r in enumerate(order) if r in show_rounds]
    linked = len(shown) < 2 or shown == list(range(shown[0], shown[-1] + 1))
    st.markdown(
        BRACKET_CSS
        + f'<div class="bkt{" link" if linked else ""}">{"".join(cols)}</div>',
        unsafe_allow_html=True)


def _bracket_section(draw, rounds, key_prefix="bkt"):
    """Render a precomputed bracket. Does no work, so widgets are free.

    `rounds` comes from `bracket_state` and is computed once by the caller and
    kept in session state. Computing it here would mean every change of the
    round filter re-ran the whole fetch-and-simulate, which in practice wiped
    the view entirely.
    """
    counts = {"played": 0, "live": 0, "pending": 0}
    for rd in rounds:
        for tie in rd["ties"]:
            counts[tie["state"]] += 1

    st.subheader("Bracket")
    st.caption(
        "Every tie with both players decided carries the model's pre-match "
        "probability \u2014 including matches already played, so you can see what "
        "it said and whether it was right. Undecided slots stay **TBD**: naming "
        "a likely opponent and pricing that matchup is the deterministic-chain "
        "error the Monte Carlo exists to avoid.\n\n"
        "Elbows join the two ties feeding each slot in the next round. The "
        "horizontal rules mark the **quarters** of the draw \u2014 two players can "
        "only meet before the semi-finals if they share a quarter \u2014 and the "
        "brighter middle rule splits the two halves, which can only meet in the "
        "final. Drop a round from the middle of the filter and the elbows "
        "disappear, since a line spanning a hidden round would claim a "
        "progression that skips it."
    )

    names = [rd["round"] for rd in rounds]
    # Everything up to and including the round being played. Future rounds are
    # all TBD, so they add width without information until results arrive.
    live_at = next((i for i, rd in enumerate(rounds)
                    if any(t["state"] == "live" for t in rd["ties"])), len(rounds) - 1)
    default = names[:live_at + 1]
    show = st.multiselect("Rounds to show", names, default=default,
                          key=f"{key_prefix}_rounds")
    if not show:
        st.info("Pick at least one round.")
        return
    _render_bracket(draw, rounds, set(show))

    st.caption(
        f"{counts['played']} decided \u00b7 {counts['live']} priced now \u00b7 "
        f"{counts['pending']} awaiting an opponent \u2003|\u2003 "
        "\u2713 the model's pick won \u00b7 \u2717 it did not "
        "\u2014 the % shown is what it gave the actual winner, so 48% is a "
        "coin flip and 12% is a genuine upset."
    )


def _row_controls(res, alive, key_prefix):
    """How many rows of the advancement table to show.

    `res` is already sorted with survivors first, so a survivors-only view is
    just a prefix of it -- no second filter pass, and the row classes used for
    dimming stay aligned with the frame.
    """
    n = len(res)
    n_alive = int(res["player_id"].isin(alive).sum()) if alive is not None else n
    c1, c2 = st.columns([2, 1])
    scope = "Still in"
    if n_alive < n:
        scope = c1.radio(
            "Show", ["Still in", "Everyone"], horizontal=True,
            key=f"{key_prefix}_scope",
            help="Eliminated players keep their pre-tournament forecast, dimmed.")
    else:
        c1.caption("No one is out yet.")
    # The cap is keyed per scope so each remembers its own value and, more
    # importantly, gets its own default. Sharing one key makes the toggle look
    # broken: with 32 still in and a cap of 32, switching to "Everyone" adds
    # nothing visible because the cap is already the binding constraint.
    pool = n if scope == "Everyone" else max(1, n_alive)
    default = min(32, pool) if scope == "Still in" else min(n_alive + 16, n)
    cap = int(c2.number_input(
        "Max rows", min_value=1, max_value=pool, value=default, step=8,
        key=f"{key_prefix}_rows_{scope.replace(' ', '_')}"))
    return min(pool, cap)


def _render_sim(draw, predictor, n_sims, actual=None, key_prefix="sim",
                resolved=None, compare=None, compare_label="Pre-tournament",
                precomputed=None, alive=None):
    """Shared rendering for a simulated draw.


    `resolved` pins matches already played (see bracket.simulate). `compare` is
    an optional pre-tournament `simulate` frame, shown alongside the live run.
    `alive` is the set of players still in, used only to dim the rest.
    """
    from tennis.sim.bracket import simulate

    if precomputed is not None:
        res = precomputed["res"]
    else:
        with st.spinner(f"Running {n_sims:,} playthroughs\u2026"):
            res = simulate(draw, predictor, n_sims=n_sims, resolved=resolved)

    order = [c for c in ["R128", "R64", "R32", "R16", "QF", "SF", "Final", "Champion"]
             if c in res.columns]
    if alive is not None:
        # Sort survivors above everyone else. Sorting on title probability
        # alone mixes them: a long-shot who is still in and a player already
        # knocked out both sit at 0.0%, and the tie breaks arbitrarily.
        res = res.assign(_alive=res["player_id"].isin(alive)) \
                 .sort_values(["_alive", "Champion"], ascending=[False, False]) \
                 .drop(columns="_alive")
    else:
        res = res.sort_values("Champion", ascending=False)

    if actual is not None:
        # aliased: a bare `compare` here would shadow this function's own
        # `compare` parameter, and .map() would silently receive the function
        from tennis.sim.draws import compare as compare_with_actual
        cmp = compare_with_actual(res, actual)
        champ = actual.loc[actual["Champion"] == 1, "player_id"]
        if len(champ):
            cid = champ.iloc[0]
            row = res[res["player_id"] == cid]
            if not row.empty:
                r = row.iloc[0]
                c = st.columns(4)
                c[0].metric("Actual champion", str(r["player_name"]))
                c[1].metric("Simulated title prob", f"{r['Champion']*100:.1f}%")
                c[2].metric("Model's favourite", str(res.iloc[0]["player_name"]))
                c[3].metric("Favourite's prob", f"{res.iloc[0]['Champion']*100:.1f}%")

        st.subheader("Simulated vs actual")
        st.caption(
            "Feature state and rankings are rebuilt to the day before this event "
            "began, so nothing from the tournament itself informs the simulation."
        )
        show = res[["player_name"] + order].copy()
        for col in order:
            act = actual.set_index("player_id")[col] if col in actual.columns else None
            simcol = (show[col] * 100).round(1).astype(str) + "%"
            if act is not None:
                got = res["player_id"].map(act).fillna(0).to_numpy()
                simcol = [f"{v} {'\u2713' if g else ''}".strip()
                          for v, g in zip(simcol, got)]
            show[col] = simcol
        show = show.head(24)
        show.columns = ["Player"] + order
        if compare is not None:
            pre_champ = compare.set_index("player_id")["Champion"]
            show.insert(1, compare_label,
                        (res["player_id"].map(pre_champ).head(24) * 100)
                        .round(1).astype(str) + "%")
        show_table(show)
        st.caption("\u2713 marks the rounds the player actually reached.")

        big = cmp[(cmp["actual"] == 1) & (cmp["sim_prob"] < 0.10)]
        if not big.empty:
            st.markdown("**Biggest surprises** \u2014 reached a round the model gave under 10%")
            b = (big.sort_values("sim_prob")
                    .head(10)[["player_name", "round", "sim_prob"]].copy())
            b["sim_prob"] = (b["sim_prob"] * 100).round(1)
            b.columns = ["Player", "Reached", "Model gave (%)"]
            show_table(b)
    else:
        st.subheader("Advancement probabilities (%)")
        n_rows = _row_controls(res, alive, key_prefix)
        if compare is None:
            disp = res[["player_name"] + order].copy()
            for col in order:
                disp[col] = (disp[col] * 100).round(1)
            show_table(disp.head(n_rows))
        else:
            # Round columns are the PRE-TOURNAMENT forecast; only the title has
            # a live figure. Current per-round numbers degenerate as the event
            # runs -- once two rounds are played every survivor sits at 100% for
            # them, and by the semi-finals five of seven columns say nothing.
            # Pre-tournament columns never degenerate, and for rounds already
            # decided they become the forecast being graded: "we gave him 82%
            # to clear R32, he went out".
            pre_df = compare.set_index("player_id")
            pre_champ = res["player_id"].map(pre_df["Champion"]) * 100
            now_champ = res["Champion"] * 100
            delta = now_champ - pre_champ

            disp = pd.DataFrame({"Player": res["player_name"]})
            for col in order:
                if col == "Champion":
                    continue
                disp[f"{col} (pre)"] = (
                    res["player_id"].map(pre_df[col]) * 100).round(1)
            disp["Champ (pre)"] = pre_champ.round(1)
            disp["Champ (now)"] = now_champ.round(1)
            disp["Move"] = [
                "\u2013" if pd.isna(d) or abs(d) < 0.05 else
                f"{'\u25b2' if d > 0 else '\u25bc'} {abs(d):.1f}"
                for d in delta
            ]
            # Players already out still show their pre-tournament numbers --
            # those remain true as a forecast -- but dimmed, so a 12% chance of
            # a semi-final is not misread as still live.
            classes = ["" if (alive is None or pid in alive) else "out"
                       for pid in res["player_id"]]
            show_table(disp.head(n_rows), row_classes=classes[:n_rows])
            st.caption(
                "Round columns are the forecast made **before the event began** "
                "and do not update \u2014 for rounds already played they show what "
                "the model expected, against what happened. Only **Champ (now)** "
                "is live. Rows are sorted by it."
            )

    st.plotly_chart(
        px.bar(res.head(16), x="Champion", y="player_name", orientation="h",
               title="Title probability").update_layout(
                   yaxis=dict(autorange="reversed"), xaxis_tickformat=".1%",
                   height=460),
        width="stretch")

    return res


@st.cache_data(show_spinner="Scanning past tournaments\u2026")
def replayable_index(min_season: int) -> pd.DataFrame:
    from tennis.sim.draws import list_replayable
    con = connect()
    try:
        return list_replayable(con, min_season=min_season)
    finally:
        con.close()


@st.cache_resource(show_spinner="Rebuilding pre-tournament state\u2026")
def predictor_as_of(seq: int, date: int):
    """Predictor whose knowledge stops before a given match sequence."""
    from tennis.models.predict import Predictor
    from tennis.sim.draws import engine_as_of
    return Predictor(engine=engine_as_of(seq), as_of=date)


def page_simulation() -> None:
    st.header("Tournament simulation")
    st.caption(
        f"Monte Carlo over the whole bracket: **{N_SIMS:,} playthroughs**, each "
        "resolving every match by a random draw weighted by the model's "
        "probability. Features are frozen at pre-tournament values for the run."
    )
    if not models_ready():
        st.warning("No trained models found.")
        return

    mode = st.radio(
        "Draw source",
        ["Current Tournament (live)", "Replay a past tournament"],
        horizontal=True,
    )
    if mode == "Current Tournament (live)":
        _mode_live(N_SIMS)
    else:
        _mode_replay(N_SIMS)


def _mode_replay(n_sims: int) -> None:
    st.caption(
        "Draws are reconstructed from our own match history \u2014 the first round of a "
        "completed event is the draw, and who-beat-whom recovers the bracket. "
        "No manual entry, no external source."
    )
    min_season = st.select_slider("From season", options=list(range(2010, 2027)),
                                  value=2022)
    idx = replayable_index(min_season)
    if idx.empty:
        st.warning("No reconstructable tournaments in that range.")
        return

    levels = st.multiselect("Level", sorted(idx["level"].unique()),
                            default=[l for l in ("grand_slam", "masters")
                                     if l in set(idx["level"])])
    sub = idx[idx["level"].isin(levels)] if levels else idx
    if sub.empty:
        st.info("No tournaments match that filter.")
        return

    pick = st.selectbox(f"Tournament ({len(sub):,} available)", sub["label"].tolist())
    row = sub[sub["label"] == pick].iloc[0]
    n_rounds = int(np.log2(int(row["draw_size"])))

    c1, c2 = st.columns(2)
    from_round = c1.selectbox(
        "Simulate from", list(range(n_rounds)),
        format_func=lambda r: "Pre-tournament" if r == 0 else f"After round {r}",
        help="Rounds before this point are pinned to what actually happened; "
             "everything after is simulated.")
    feature_mode = c2.radio(
        "Feature state", ["Updated to that round", "Frozen pre-tournament"],
        horizontal=True, disabled=(from_round == 0),
        help="Whether each player's form and Elo include the rounds already "
             "played in this event.")

    if from_round > 0:
        st.info(
            "**These two settings do different jobs.** *Simulate from* changes "
            "how much the model is told (rounds pinned to reality) \u2014 a later "
            "start always scores better simply because less is unknown, so it is "
            "not a fairer model, just a better-informed one. *Feature state* is "
            "the real modelling question: with the same rounds pinned, does "
            "knowing a player's form from this week help? Recent form is the "
            "second-largest permutation-importance family, so it is worth a look."
        )

    if st.button("Run replay", type="primary"):
        from tennis.sim.draws import (
            actual_progression, build_draw, engine_as_of, replay_from_round,
        )
        con = connect()
        try:
            rep = replay_from_round(con, row["tourney_key"], from_round)
        finally:
            con.close()
        if rep is None:
            st.error("Could not reconstruct this bracket.")
            return
        draw, g = rep["draw"], rep["matches"]
        actual = actual_progression(g, draw.slots)
        pre_seq = int(row["min_seq"])
        date = int(row["tourney_date"])

        if rep["completed_rounds"] < from_round:
            st.warning(
                f"Only {rep['completed_rounds']} round(s) could be pinned \u2014 the "
                "bracket has an unplayed or unresolved match before that point.")

        # Feature state: pre-tournament, or rebuilt to the start of this round.
        use_seq = pre_seq
        if from_round > 0 and feature_mode.startswith("Updated") and rep["state_seq"]:
            use_seq = int(rep["state_seq"])
        pred = predictor_as_of(use_seq, date)

        # Unconditional pre-tournament run, shown as a reference column.
        compare = None
        if from_round > 0:
            from tennis.sim.bracket import simulate
            with st.spinner("Pre-tournament reference run\u2026"):
                compare = simulate(draw, predictor_as_of(pre_seq, date),
                                   n_sims=n_sims)

        from tennis.sim.bracket import bracket_state, simulate as _simulate
        with st.spinner("Building the bracket\u2026"):
            from tennis.sim.draws import score_lookup
            rounds = bracket_state(draw, pred, rep["resolved"],
                                   scores=score_lookup(g))
        with st.spinner(f"Running {n_sims:,} playthroughs\u2026"):
            res = _simulate(draw, pred, n_sims=n_sims, resolved=rep["resolved"])

        # Same reason as live mode: the round filter reruns the script, and
        # results held only inside this button block would disappear.
        st.session_state["replay_payload"] = {
            "draw": draw, "resolved": rep["resolved"], "rounds": rounds,
            "res": res, "compare": compare, "actual": actual,
            "n_sims": n_sims, "seq": use_seq, "date": date, "pre_seq": pre_seq,
            "pinned": sum(len(v) for v in rep["resolved"].values()),
            "from_round": from_round,
        }

    payload = st.session_state.get("replay_payload")
    if payload:
        st.caption(
            f"{payload['pinned']} match(es) pinned to actual results \u00b7 feature "
            f"state {'as of round ' + str(payload['from_round'] + 1) if payload['seq'] != payload['pre_seq'] else 'pre-tournament'}"
        )
        _bracket_section(payload["draw"], payload["rounds"], key_prefix="replay")
        _render_sim(payload["draw"], predictor_as_of(payload["seq"], payload["date"]),
                    payload["n_sims"], actual=payload["actual"],
                    key_prefix="replay", resolved=payload["resolved"],
                    compare=payload["compare"], compare_label="Pre-tourn %",
                    precomputed={"res": payload["res"]})


@st.cache_data(ttl=1800, show_spinner="Finding live tournaments\u2026")
def _live_tournaments(days: int) -> pd.DataFrame:
    from tennis.ingest.draws_api import discover_tournaments
    return discover_tournaments(days=days)


# Which tier a level belongs to. History is only ever consulted to resolve an
# ambiguity *inside* a tier, never to move an event between them.
LEVEL_BAND = {
    "challenger": "challenger", "atp250": "tour", "atp500": "tour",
    "masters": "masters", "finals": "finals", "grand_slam": "slam",
    "davis_cup": "team", "olympics": "team",
}
# The feed's rankId, mapped to the same tiers. It is coarse but reliable at this
# resolution -- the one thing it cannot do is split ATP 250 from ATP 500.
RANK_BAND = {1: "challenger", 2: "tour", 3: "masters", 7: "finals"}

# Display order for the live tournament list: biggest event first. LEVELS is a
# vocabulary, not a ranking, so the order is spelled out rather than reused.
LEVEL_RANK = {
    "grand_slam": 0, "finals": 1, "masters": 2, "atp500": 3, "atp250": 4,
    "olympics": 5, "davis_cup": 6, "challenger": 7,
}


@st.cache_data(show_spinner=False)
def _history_context() -> pd.DataFrame:
    """Surface and level we already hold for each tournament city."""
    con = connect()
    try:
        return pd.read_sql(
            "SELECT name, surface, level, indoor, COUNT(*) n, "
            "MAX(tourney_date) last_held FROM tournaments "
            "WHERE season >= 2015 AND surface IS NOT NULL "
            "GROUP BY name, surface, level, indoor", con)
    finally:
        con.close()


def _refine_from_history(feed_name: str, band: str | None = None) -> dict:
    """Look a feed tournament up in our own history, within one tier.

    The feed's rankId is a coarse band -- it cannot tell ATP 250 from ATP 500,
    which would label the Citi Open (a 500) as a 250 and feed the model the
    wrong tournament level. Our own tournaments table knows the difference, so
    it wins when the city matches.

    `band` is what keeps that from going too far. A city can host events at
    several tiers -- Hamburg has an ATP 500, a retired Masters *and* a
    Challenger; Buenos Aires and Barcelona each carry 12-21 Challengers
    alongside a main-tour week -- and matching on the city alone returns
    whichever has the longer history. That relabelled the Hamburg Challenger as
    an ATP 500. Restricting the lookup to the tier the feed already reports
    leaves history doing only the job it is needed for.

    Note the level in our history is trustworthy here precisely because it is
    not inferred from the name either: `ingest/load.py` sets it from the source
    file, so every row out of `*_challenger.csv` is marked challenger outright.
    """
    hist = _history_context()
    if hist.empty or not feed_name:
        return {}
    if band:
        hist = hist[hist["level"].map(LEVEL_BAND) == band]
        if hist.empty:
            return {}
    # Feed names look like "Citi Open - Washington" or "Hagen Challenger";
    # ours are bare cities. Stripping the suffix is only ever used to match the
    # *city* -- the tier comes from `band`, never from the words in the name.
    tail = feed_name.split(" - ")[-1]
    for junk in (" Challenger", " Open", " 2", " 1"):
        tail = tail.replace(junk, "")
    tail = tail.strip()
    if len(tail) < 4:
        return {}
    m = hist[hist["name"].str.lower() == tail.lower()]
    if m.empty:
        m = hist[hist["name"].str.contains(tail, case=False, regex=False, na=False)]
    if m.empty:
        return {}
    # Most recent first, then longest-running. Frequency alone picks up surfaces
    # an event has since abandoned -- the Hamburg Challenger's four carpet
    # editions tie its four hard ones, and carpet left the tour in 2009.
    top = m.sort_values(["last_held", "n"], ascending=False).iloc[0]
    out = {"surface": top["surface"], "level": top["level"]}
    if pd.notna(top["indoor"]):
        out["indoor"] = float(top["indoor"])
    return out


def _mode_live(n_sims: int) -> None:
    import os

    st.caption(
        "Upcoming draws come from the RapidAPI fixtures feed. Past draws never "
        "use it \u2014 those are reconstructed locally."
    )
    st.info(
        "**What this feed can do:** once an opening round is scheduled \u2014 in "
        "practice about a day before play \u2014 the full first round is available. "
        "It does **not** publish draws at ceremony time: an event eight weeks out "
        "returns nothing. Simulate the evening before, not a week ahead."
    )
    if not os.getenv("RAPIDAPI_KEY"):
        st.warning(
            "`RAPIDAPI_KEY` is not set. Put it in a `.env` file at the repo root "
            "(already gitignored) or export it before starting Streamlit."
        )
        return

    from tennis.ingest.draws_api import (
        COURT_SURFACE, RANK_ITF, RANK_LEVEL, DrawFeedError,
        build_draw_from_fixtures, build_draw_from_sheet, first_round,
        event_matches, parse_draw_sheet, parse_fixtures, played_results,
        resolve_tourney_key, tournament_draw, tournament_fixtures,
        tournament_slug, verify_winner_convention,
    )
    from tennis.sim.draws import walk_bracket

    try:
        live = _live_tournaments(LOOK_AHEAD_DAYS)
    except DrawFeedError as exc:
        st.error(str(exc))
        return
    if live.empty:
        st.warning("No scheduled fixtures found in that window.")
        return

    # ITF events sit below this project's Challenger floor, so hide them by
    # default rather than letting someone simulate a field we have no data for.
    atp = live[live["rank_id"].fillna(-1) != RANK_ITF]
    show_itf = st.checkbox(
        f"Include ITF events ({len(live) - len(atp)} hidden)", value=False,
        help="M15/M25 futures are below Challenger level and are largely absent "
             "from our player database.")
    listed = live if show_itf else atp
    if listed.empty:
        st.warning("No ATP-level tournaments scheduled in that window.")
        return

    def _ctx_for(row) -> dict:
        """Feed codes, refined by our own history within the same tier.

        The feed reports this event; history reports a different one that
        happened to share a city, so the feed wins wherever it has an answer.
        History fills the gaps and splits ATP 250 from ATP 500, which is the
        one distinction the feed cannot make.
        """
        out = {}
        band = None
        if pd.notna(row["court_id"]) and int(row["court_id"]) in COURT_SURFACE:
            out["surface"], out["indoor"] = COURT_SURFACE[int(row["court_id"])]
        if pd.notna(row["rank_id"]):
            band = RANK_BAND.get(int(row["rank_id"]))
            if int(row["rank_id"]) in RANK_LEVEL:
                out["level"] = RANK_LEVEL[int(row["rank_id"])]
        hist = _refine_from_history(str(row["name"]), band)
        for k, v in hist.items():
            # Level is the one field history is allowed to overwrite; it is
            # already confined to the feed's own tier by `band`.
            if k == "level" or k not in out:
                out[k] = v
        return out

    ctx_by_id = {r.tournament_id: _ctx_for(r._asdict())
                 for r in listed.itertuples()}
    # Biggest event first. The feed orders by fixture count, which buries a
    # Masters under whichever Challenger happens to have more matches left.
    # Unknown levels sort last rather than jumping the queue.
    listed = listed.assign(
        _lvl=[LEVEL_RANK.get(ctx_by_id[i].get("level"), 99)
              for i in listed["tournament_id"]]
    ).sort_values(["_lvl", "fixtures"], ascending=[True, False]).drop(columns="_lvl")

    disp = listed.copy()
    disp["surface"] = disp["tournament_id"].map(
        lambda i: ctx_by_id[i].get("surface", "?"))
    disp["level"] = disp["tournament_id"].map(
        lambda i: ctx_by_id[i].get("level", "?"))
    st.markdown("**Tournaments with scheduled fixtures**")
    show_table(disp[["name", "fixtures", "surface", "level",
                     "first_date", "last_date"]].rename(columns={
        "name": "Tournament", "fixtures": "Fixtures", "surface": "Surface",
        "level": "Level", "first_date": "From", "last_date": "To"}))

    # Select by name; the id travels along but is never what the user reads.
    id_by_label = {}
    for r in listed.itertuples():
        label = f"{r.name}  \u2014  {r.fixtures} fixtures"
        id_by_label[label] = r.tournament_id
    choice = st.selectbox("Tournament", list(id_by_label))
    tid = id_by_label[choice]
    row = listed[listed["tournament_id"] == tid].iloc[0]

    ctx = ctx_by_id[tid]
    hint_surface, hint_level = ctx.get("surface"), ctx.get("level")

    c1, c2, c3 = st.columns(3)
    surface = c1.selectbox(
        "Surface", SURFACES,
        index=SURFACES.index(hint_surface) if hint_surface in SURFACES else 0,
        key="live_surf")
    level = c2.selectbox(
        "Level", LEVELS,
        index=LEVELS.index(hint_level) if hint_level in LEVELS else 3,
        key="live_lvl")
    best_of = c3.selectbox("Best of", [3, 5], key="live_bo")
    st.caption(
        "Surface and level are pre-filled from the feed's court and tier codes, "
        "refined against our own history for the same city and **the same "
        "tier** \u2014 the feed cannot tell ATP 250 from ATP 500, which is all the "
        "lookup is for. It is tier-scoped because a city can host several: "
        "Hamburg runs an ATP 500, a retired Masters and a Challenger, and an "
        "unscoped match relabelled the Challenger a 500. Both are editable. "
        "Best-of is not carried by the feed \u2014 set it yourself (5 only for "
        "Grand Slams)."
    )

    slug_default = tournament_slug(str(row["name"]))
    slug = st.text_input(
        "Draw-sheet slug", slug_default,
        help="The draw endpoint keys on a name slug rather than the numeric id. "
             "Derived from the tournament name; override if the lookup fails.")

    if st.button("Fetch draw & simulate", type="primary"):
        year = int(str(row["first_date"])[:4]) if row["first_date"] else None
        live_resolved = None
        done_rounds = 0
        stats_row = None
        live_scores = {}
        standing = None

        # Preferred path: the full draw sheet. It is the only source that
        # includes players holding a first-round bye -- the fixtures feed omits
        # them entirely, which silently removed all 32 seeds from a 96-draw.
        sheet = pd.DataFrame()
        full_sheet = pd.DataFrame()
        try:
            payload = tournament_draw(slug, year)
            sheet = parse_draw_sheet(payload)
            full_sheet = parse_draw_sheet(payload, first_round_only=False)
        except DrawFeedError as exc:
            st.caption(f"Draw sheet unavailable ({exc}); falling back to fixtures.")

        if not sheet.empty:
            n_byes = int(sheet["is_bye"].sum())
            n_ties = len(sheet)
            # A real bracket has a power-of-two number of first-round ties.
            # Anything else means the sheet is malformed, and padding it up is
            # how one stray row once doubled a 128-slot draw to 256 and handed
            # that player a free run to the final. Show the sheet only here,
            # where you need it to see what is wrong.
            if n_ties & (n_ties - 1):
                st.error(
                    f"Draw sheet has {n_ties} first-round ties, which is not a "
                    "power of two. The bracket would be padded with phantom "
                    "byes and the simulation would be meaningless.")
                show_table(sheet)
                return
            st.success(
                f"{row['name']} \u2014 {n_ties} first-round ties, {n_byes} byes. "
                "The bracket below is the draw.")
            draw, unresolved = build_draw_from_sheet(
                sheet, name=str(row["name"]), surface=surface, level=level,
                best_of=best_of, indoor=ctx.get("indoor", 0.0))
            st.caption(
                "Bracket positions come from the draw sheet, so later-round "
                "matchups are the real ones rather than an approximation.")

            # Pin matches already played, so a mid-event simulation does not
            # keep giving knocked-out players a chance of going deep.
            res_df = played_results(full_sheet)
            if not res_df.empty:
                live_resolved, standing, done_rounds = walk_bracket(draw, res_df)
                live_scores = {(r.winner_id, r.loser_id): r.score
                               for r in res_df.itertuples()
                               if isinstance(getattr(r, "score", None), str)}
                pinned = sum(len(v) for v in live_resolved.values())
                st.success(
                    f"{pinned} completed match(es) pinned to their real result; "
                    f"{done_rounds} round(s) fully decided. Everything after is "
                    "simulated from current form.")
                # The sheet has no explicit winner field -- player1 is assumed to
                # be the winner. Check that against results we already hold and
                # say so if the feed ever changes convention.
                # Link the feed event to ours once, then read our rows by key.
                # The two sources share no tournament id, and their names
                # disagree, so the link comes from player overlap inside a date
                # window and is cached in tourney_xref.
                tkey = resolve_tourney_key(
                    tid, str(row["name"]),
                    [x for x in draw.slots if x != "__BYE__"],
                    year or 0, row.get("first_date"))
                chk = verify_winner_convention(
                    full_sheet, event_matches(tkey) if tkey else pd.DataFrame())
                if chk["contradicted"]:
                    st.error(
                        f"{chk['contradicted']} of {chk['checked']} completed "
                        "matches disagree with our own records about who won. "
                        "The feed may have changed how it marks winners \u2014 treat "
                        "the pinned results with suspicion.")
                elif chk["confirmed"]:
                    st.caption(
                        f"Winner convention cross-checked against our own data: "
                        f"{chk['confirmed']} agree, 0 disagree "
                        f"({chk['unknown']} not yet ingested).")
        else:
            try:
                payload = tournament_fixtures(tid)
            except DrawFeedError as exc:
                st.error(str(exc))
                return
            fx = parse_fixtures(payload)
            if fx.empty:
                st.warning(
                    "No draw sheet and no unplayed singles fixtures. Either the "
                    "event has finished, or its draw is not published yet.")
                return
            r1 = first_round(fx)
            st.warning(
                f"{row['name']} \u2014 no draw sheet available, using {len(r1)} "
                "scheduled fixtures. Players holding a first-round bye are **not** "
                "in this feed, so if the event has byes the field is incomplete "
                "and every listed player's chances are overstated.")
            show_table(r1.rename(columns={
                "p1_name": "Player 1", "p2_name": "Player 2", "seed1": "Seed 1",
                "seed2": "Seed 2", "date": "Date"})[
                ["Player 1", "Seed 1", "Player 2", "Seed 2", "Date"]])
            draw, unresolved = build_draw_from_fixtures(
                r1, name=str(row["name"]), surface=surface, level=level,
                best_of=best_of, indoor=ctx.get("indoor", 0.0))
            st.caption(
                "Bracket order follows the feed's order: correct for round-1 "
                "pairings, but the arrangement of halves may differ from the "
                "official sheet, so later-round matchups are indicative.")

        if unresolved:
            st.warning(
                "Not matched to a player in our database, so their slot is "
                "treated as a bye: " + ", ".join(unresolved[:12]))
        if draw.size < 2:
            st.error("Too few resolved players to simulate.")
            return
        # Same model, same features, no results pinned -- so the difference
        # between the two runs is caused by the completed matches alone.
        compare = None
        if live_resolved and any(live_resolved.values()):
            from tennis.sim.bracket import simulate as _sim
            with st.spinner("Pre-tournament reference run\u2026"):
                # Same model, same features, nothing pinned -- so every
                # difference from the live run is caused by results alone.
                compare = _sim(draw, get_predictor(), n_sims=n_sims)

            # `standing` is who occupies each slot after the last fully
            # decided round -- i.e. actually still in. The union of every
            # round's winners would wrongly include players since knocked out.
            alive = {x for x in standing if x not in (None, "__BYE__")}
            stats_row = {
                "played": sum(len(v) for v in live_resolved.values()),
                "done": done_rounds, "n_rounds": draw.n_rounds,
                "field": len([x for x in draw.slots if x != "__BYE__"]),
                "alive": len(alive),
            }

        from tennis.sim.bracket import bracket_state, simulate as _simulate
        with st.spinner("Building the bracket\u2026"):
            rounds = bracket_state(draw, get_predictor(), live_resolved,
                                   scores=live_scores)
        with st.spinner(f"Running {n_sims:,} playthroughs\u2026"):
            res = _simulate(draw, get_predictor(), n_sims=n_sims,
                            resolved=live_resolved)

        # Everything the view needs, kept in session state. Widgets below
        # (the round filter especially) trigger a rerun, and if the results
        # only existed inside this button block they would vanish on the
        # first interaction -- which is exactly what happened.
        st.session_state["live_payload"] = {
            "draw": draw, "resolved": live_resolved, "rounds": rounds,
            "res": res, "compare": compare,
            "n_sims": n_sims, "stats": stats_row,
            "alive": {x for x in (standing or []) if x not in (None, "__BYE__")},
        }

    payload = st.session_state.get("live_payload")
    if payload:
        s_ = payload.get("stats") or {}
        if s_:
            c = st.columns(4)
            c[0].metric("Matches played", f"{s_['played']}")
            c[1].metric("Rounds complete", f"{s_['done']} of {s_['n_rounds']}")
            c[2].metric("Field", f"{s_['field']} entered")
            c[3].metric("Still alive", f"{s_['alive']}")
        _bracket_section(payload["draw"], payload["rounds"], key_prefix="live")
        _render_sim(payload["draw"], get_predictor(), payload["n_sims"],
                    key_prefix="live", resolved=payload["resolved"],
                    compare=payload["compare"], compare_label="Pre-tourn",
                    alive=payload.get("alive"),
                    precomputed={"res": payload["res"]})


def page_ratings() -> None:
    st.header("Elo ratings")

    basis = st.radio(
        "Rating basis", ["All-time", "Season to date"], horizontal=True,
        help="All-time is the career rating. Season to date shows how far that "
             "rating has moved since the season opened, plus this season's W-L.")
    season = basis != "All-time"
    elo = load_elo()
    if elo.empty:
        st.warning("No Elo state. Run the feature pipeline.")
        return

    sea = load_elo_season() if season else None
    if season and (sea is None or sea.empty):
        st.info("No matches yet this season.")
        return
    if season:
        elo = elo.merge(sea, on="player_id", how="inner")
        opened = int(elo["season_start"].iloc[0])
        st.caption(
            f"**{season_label(opened)} season**, opened "
            f"**{pd.to_datetime(str(opened)).date()}** — the tour year runs from "
            "late December, so the season began in the previous calendar year. "
            "The cut falls after the Next Gen Finals, which close a season, and "
            "before the United Cup and Brisbane, which open the next. **Change** "
            "is how far a player's rating has moved since that day.\n\n"
            "This is deliberately *not* a rating restarted at the opener. With "
            "everyone reset to 1500, beating a field is worth the same whoever "
            "the field is, and Challenger and main-tour players barely play each "
            "other, so one season never prices the two pools against each other: "
            "a 38-11 Challenger run against median rank 242 came out rated above "
            "a 27-7 run against the top 60. Moving the calibrated rating has "
            "neither problem — beating rank 242 when you are rated to beat rank "
            "242 correctly moves nothing."
        )
    else:
        st.caption(
            "Overall and per-surface Elo side by side, as the model sees them. A "
            "surface rating is seeded from the player's overall rating and blends "
            "toward the surface-specific figure as matches on it accumulate, so a "
            "player new to a surface reads near their overall level rather than at "
            "a meaningless 1500. Carpet is excluded \u2014 the tour stopped using it "
            "after 2009."
        )

    c1, c2, c3 = st.columns(3)
    if season:
        min_n = c1.slider("Minimum matches this season", 0, 60, 10)
        sort_label = c2.selectbox(
            "Sort by", ["Change this season", "Overall", "Hard", "Clay", "Grass"],
            index=0)
    else:
        min_n = c1.slider("Minimum matches (overall)", 0, 200, 30)
        sort_label = c2.selectbox("Sort by", ["Overall", "Hard", "Clay", "Grass"],
                                  index=0)
    top_n = c3.slider("Show top", 10, 200, 60, step=10)
    active_only = st.checkbox(
        "Active players only (played in the last year)", value=True,
        help="Season mode already excludes anyone who has not played."
             if season else None)

    if season:
        elo["season_n"] = elo["season_w"] + elo["season_l"]
        elo["change"] = elo["overall"] - elo["elo_start"]
        sub = elo[elo["season_n"] >= min_n].copy()
    else:
        sub = elo[elo["n_overall"].fillna(0) >= min_n].copy()
    if active_only:
        cutoff = (pd.Timestamp.today() - pd.DateOffset(years=1)).strftime("%Y%m%d")
        sub = sub[sub["last_seen"].astype("Int64").astype(str) >= cutoff]
    if sub.empty:
        st.info("No players match those filters.")
        return

    sort_by = {"Overall": "overall", "Change this season": "change"}.get(
        sort_label, sort_label)
    sub = sub.sort_values(sort_by, ascending=False).head(top_n)

    disp = sub[["name"] + ELO_SCOPES + ["n_overall"]].copy()
    disp.columns = ["Player", "Overall", "Hard", "Clay", "Grass", "Matches"]
    for c in ("Overall", "Hard", "Clay", "Grass"):
        disp[c] = disp[c].round(0)
    disp["Matches"] = disp["Matches"].fillna(0).astype(int)

    if season:
        chg = sub["change"].to_numpy()
        disp.insert(2, "Change", [
            "\u2013" if pd.isna(d) or abs(d) < 0.5 else
            f"{'\u25b2' if d > 0 else '\u25bc'} {abs(d):.0f}" for d in chg])
        disp.insert(3, "Season W-L",
                    [f"{w}-{l}" for w, l in zip(sub["season_w"], sub["season_l"])])
        disp = disp.drop(columns="Matches")
    show_table(disp)

    st.subheader("Surface specialists")
    st.caption(
        "Each rating is shown against the **median for that surface** among the "
        "players listed above, not against the player's own overall rating. "
        "Surfaces are not on a common scale \u2014 clay and hard fields differ in "
        "depth \u2014 so comparing a player to the same surface's median is the only "
        "like-for-like read. The blend removes the old sample-size artefact, "
        "but not this one."
    )
    min_surface_n = st.slider(
        "Minimum matches on a surface to plot it", 5, 60, 15,
        help="Below this the blend is dominated by the overall rating, so the "
             "bar would mostly restate a player's general level rather than "
             "saying anything about the surface.")

    rel = sub.head(20).copy()
    for c in ("Hard", "Clay", "Grass"):
        thin = rel[f"n_{c}"].fillna(0) < min_surface_n
        qualified = sub.loc[sub[f"n_{c}"].fillna(0) >= min_surface_n, c]
        median = qualified.median() if len(qualified) else np.nan
        rel[c] = (rel[c] - median).mask(thin)

    long = rel.melt(id_vars=["name"], value_vars=["Hard", "Clay", "Grass"],
                    var_name="Surface", value_name="Elo vs surface median").dropna()
    if long.empty:
        st.info("No players meet that surface-match threshold.")
        return
    st.plotly_chart(
        px.bar(long, x="Elo vs surface median", y="name", color="Surface",
               barmode="group", height=620,
               color_discrete_map={"Hard": "#2E86AB", "Clay": "#C1666B",
                                   "Grass": "#4C9F70"}
               ).update_layout(yaxis=dict(autorange="reversed", title=None)),
        width="stretch")


def page_status() -> None:
    st.header("Data & pipeline status")
    con = connect()
    counts = {}
    for t in ["matches", "players", "tournaments", "odds", "rankings",
              "elo_state", "sim_results", "backtest"]:
        try:
            counts[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception:
            counts[t] = 0
    rng = con.execute("SELECT MIN(tourney_date), MAX(tourney_date) FROM matches").fetchone()
    # odds holds one row per (match, bookmaker), so count distinct matches.
    priced = con.execute("SELECT COUNT(DISTINCT match_id) FROM odds").fetchone()[0]
    state = dict(con.execute("SELECT key, value FROM pipeline_state").fetchall())
    con.close()

    c = st.columns(4)
    c[0].metric("Matches", f"{counts['matches']:,}")
    c[1].metric("Players", f"{counts['players']:,}")
    c[2].metric("Tournaments", f"{counts['tournaments']:,}")
    c[3].metric("Matches with odds", f"{priced:,}",
                help=f"{counts['odds']:,} price rows across all bookmakers")
    st.write(f"**Coverage:** {rng[0]} → {rng[1]}")
    st.json(state)

    p = ARTIFACTS / "last_daily_run.json"
    if p.exists():
        st.subheader("Last daily run")
        st.json(json.loads(p.read_text()))


def dedupe_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """Latest priced capture per *pairing*, not per (pairing, date).

    A postponed or re-listed match carries a new play_date, so keying on the
    date let one fixture through three times over -- and on the betting page a
    duplicated fixture is a tripled stake on a single result. Names are sorted
    into the key because the site's p1/p2 order is not stable between captures;
    the surviving row keeps whatever orientation it was captured with.
    """
    if df.empty:
        return df
    pair = pd.DataFrame({"a": df.p1_name, "b": df.p2_name})
    return (df.assign(_pair=pair.min(axis=1) + "|" + pair.max(axis=1))
              .sort_values("captured_at").drop_duplicates("_pair", keep="last")
              .drop(columns="_pair"))


# Highest tour level first. Anything unrecognised sorts last rather than
# jumping the queue on a name we failed to resolve.
LEVEL_ORDER = {"grand_slam": 0, "finals": 1, "masters": 2, "olympics": 3,
               "atp500": 4, "atp250": 5, "davis_cup": 6, "challenger": 7}
LEVEL_LABEL = {"grand_slam": "Grand Slam", "finals": "Finals",
               "masters": "Masters", "olympics": "Olympics",
               "atp500": "ATP 500", "atp250": "ATP 250",
               "davis_cup": "Davis Cup", "challenger": "Challenger"}


def _tourney_meta(names: pd.Series, con) -> pd.DataFrame:
    """Resolve each fixture's tour level and surface from its event name.

    Two sources, in this order, because neither alone is enough. A name
    carrying "challenger" states its own level and is trusted outright -- the
    events currently in play (Kingston, Quebec City, Roehampton, Sion) are too
    new to appear in `tournaments` at all, since that table is built from
    results. Everything else is matched by normalised name against the whole
    history, levels being stable year to year.

    Surface comes from the same lookup, and matters more than the ordering it
    was added for: the predictions on this page were being made with surface
    hardcoded to Hard, which silently mispriced every clay and grass event.
    """
    import re

    def norm(s: str) -> str:
        s = re.sub(r"\b(challenger|wta|atp|masters|open|itf)\b", "",
                   (s or "").lower())
        return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())

    hist = pd.read_sql("SELECT name, level, surface, season FROM tournaments", con)
    hist["k"] = hist.name.map(norm)
    hist = (hist.sort_values("season", ascending=False)
                .drop_duplicates("k").set_index("k"))

    rows = []
    for n in names:
        lo = (n or "").lower()
        hit = hist.loc[norm(n)] if norm(n) in hist.index else None
        if "challenger" in lo:
            level = "challenger"          # the name is authoritative
        else:
            level = hit.level if hit is not None else None
        rows.append({"tournament": n, "level": level,
                     "surface": hit.surface if hit is not None else "Hard"})
    return pd.DataFrame(rows).drop_duplicates("tournament")


# Short TTL, not the default "cache forever". This page shows live prices and
# the capture agent writes every three hours; without a TTL the dashboard
# served the same frame for a week and the staleness banner reported the age of
# the *cached* capture, so a freshly-run agent changed nothing on screen.
@st.cache_data(ttl=300, show_spinner=False)
def _upcoming(max_age_days: int = 3) -> pd.DataFrame:
    """Fixtures from the most recent live capture, with their moneyline.

    `odds_snapshots` holds one row per capture, so the latest capture per
    fixture is the freshest price we hold. Stale captures are surfaced rather
    than hidden -- a price from three days ago is not a price you can bet.

    ATP only. The capture agent deliberately pulls both tours, but every model
    here is trained on ATP matches, so a WTA fixture would be priced by a model
    that has never seen those players -- and `surname_key` can quietly match a
    WTA name onto an unrelated ATP player, which turns a meaningless number
    into a confident-looking one.
    """
    con = connect()
    try:
        last = con.execute("SELECT MAX(play_date) FROM odds_snapshots").fetchone()[0]
        if last is None:
            return pd.DataFrame()
        df = pd.read_sql(
            "SELECT * FROM odds_snapshots "
            "WHERE play_date >= ? AND LOWER(COALESCE(tour,'atp')) = 'atp'",
            con, params=(int(last) - max_age_days,))
        players = pd.read_sql(
            "SELECT player_id, name, last_seen FROM players", con)
        meta = _tourney_meta(df.tournament.dropna().unique(), con) \
            if not df.empty else pd.DataFrame()
    finally:
        con.close()
    if not df.empty and not meta.empty:
        df = df.merge(meta, on="tournament", how="left")
    if df.empty:
        return df
    df = dedupe_fixtures(df[df.p1_odds.notna() & df.p2_odds.notna()])

    from tennis.ingest.odds_live import surname_key
    # Most-recently-active player wins a key collision; two players can share a
    # surname and initial, and the active one is overwhelmingly the fixture.
    players["key"] = players.name.map(surname_key)
    players = (players.sort_values("last_seen", ascending=False)
                      .drop_duplicates("key"))
    lookup = dict(zip(players.key, players.player_id))
    df["p1_id"] = df.p1_name.map(surname_key).map(lookup)
    df["p2_id"] = df.p2_name.map(surname_key).map(lookup)
    return df


def _kelly(p: np.ndarray, price: np.ndarray, cap: float) -> np.ndarray:
    """Fractional Kelly, capped. Negative edges stake nothing."""
    b = price - 1
    f = (p * b - (1 - p)) / np.where(b > 0, b, np.nan)
    return np.clip(np.nan_to_num(f), 0, cap)


def page_betting() -> None:
    st.header("Betting")
    st.caption(
        "Model probability against the live moneyline for upcoming fixtures. "
        "Expected value is shown beside **what that expected value has "
        "historically actually returned**, because the two are not the same "
        "number and only one of them is measured."
    )
    if not models_ready():
        st.warning("No trained models found.")
        return

    df = _upcoming()
    if df.empty:
        st.info("No captured fixtures. Run `scripts/capture_odds.sh`.")
        return

    last_cap = pd.to_datetime(df.captured_at.max())
    age_h = (pd.Timestamp.now(tz="UTC") - last_cap.tz_localize("UTC")
             if last_cap.tzinfo is None else
             pd.Timestamp.now(tz="UTC") - last_cap).total_seconds() / 3600
    if age_h > 6:
        st.error(
            f"Latest capture is **{age_h:.0f} hours old** ({last_cap:%Y-%m-%d %H:%M} "
            "UTC). These are not currently bettable prices — the capture agent "
            "(`com.tennis.odds`) is not running, or the backfill is holding the "
            "database. Shown for inspection only."
        )
    else:
        st.caption(f"Latest capture {last_cap:%Y-%m-%d %H:%M} UTC "
                   f"({age_h:.1f}h ago).")

    unresolved = int(df.p1_id.isna().sum() + df.p2_id.isna().sum())
    d = df.dropna(subset=["p1_id", "p2_id"]).copy()
    if d.empty:
        st.warning("No fixtures resolved to known players.")
        return

    c1, c2, c3 = st.columns(3)
    edge = c1.slider("Minimum edge (EV)", 0.0, 0.25, 0.05, step=0.01)
    cap = c2.slider("Kelly cap (fraction of bank)", 0.005, 0.05, 0.02, step=0.005)
    bank = c3.number_input("Bankroll", min_value=0.0, value=1000.0, step=100.0)

    from tennis.models.devig import shin
    from tennis.models.predict import MatchContext

    pred = get_predictor()
    rows = []
    for r in d.itertuples():
        # Surface and level come from the event, not from a constant. They were
        # pinned to Hard/atp250 here, which mispriced every clay and grass
        # fixture on the page while looking entirely normal.
        level = getattr(r, "level", None) or "atp250"
        ctx = MatchContext(surface=getattr(r, "surface", None) or "Hard",
                           level=level,
                           best_of=5 if level == "grand_slam" else 3,
                           is_challenger=int(level == "challenger"))
        try:
            pm = float(pred.predict_many([(r.p1_id, r.p2_id)], ctx)
                       .iloc[0]["p1_win_prob"])
        except Exception:
            continue
        mk = shin(np.array([[r.p1_odds, r.p2_odds]], dtype=float))[0][0, 0]
        ev1, ev2 = pm * r.p1_odds - 1, (1 - pm) * r.p2_odds - 1
        side1 = ev1 >= ev2
        rows.append({
            "Tournament": r.tournament, "Time": r.start_time or "",
            "_lvl": LEVEL_ORDER.get(level, 99),
            "Level": LEVEL_LABEL.get(level, "—"),
            "Pick": r.p1_name if side1 else r.p2_name,
            "Against": r.p2_name if side1 else r.p1_name,
            "Price": r.p1_odds if side1 else r.p2_odds,
            "Model %": round((pm if side1 else 1 - pm) * 100, 1),
            "Market %": round((mk if side1 else 1 - mk) * 100, 1),
            "EV %": round(max(ev1, ev2) * 100, 1),
            "_p": pm if side1 else 1 - pm,
            "_price": r.p1_odds if side1 else r.p2_odds,
        })
    if not rows:
        st.warning("No fixtures could be priced.")
        return

    t = pd.DataFrame(rows)
    t["Stake"] = (_kelly(t._p.to_numpy(), t._price.to_numpy(), cap) * bank).round(2)
    # Order by tour level (highest first), then the event, then start time --
    # a schedule you can read down, rather than by EV, which scattered the same
    # tournament across the table and put the least trustworthy rows on top.
    shown = (t[t["EV %"] >= edge * 100]
             .sort_values(["_lvl", "Tournament", "Time"])
             .drop(columns="_lvl"))

    st.subheader(f"{len(shown)} of {len(t)} fixtures clear a {edge:.0%} edge")
    if unresolved:
        st.caption(f"{unresolved} player name(s) did not resolve and were skipped.")
    if shown.empty:
        st.info("Nothing clears that threshold.")
    else:
        show_table(shown.drop(columns=["_p", "_price"]))
        big = shown[shown["EV %"] >= 50]
        if len(big):
            st.warning(
                f"**The {len(big)} row(s) above 50% EV are the least trustworthy "
                "on this page, not the most.** They are cases where the model "
                "disagrees violently with the price, and disagreement is exactly "
                "where it has been measured to be worst: split by how far model "
                "and market diverge, the market wins *every* bucket, and the gap "
                "widens with the disagreement — where the model is 10pts above "
                "the price it scores 0.653 log loss against the market's 0.601. "
                "A 160% EV is not a 160% edge; it is the model being wrong about "
                "a longshot."
            )

    _betting_reality(edge)


def _betting_reality(edge: float) -> None:
    """What this strategy actually returned, measured, next to the EV above.

    This is the part that makes the screen honest rather than merely accurate.
    Expected value is what the model *thinks*; these are the realised numbers
    from the walk-forward backtest against real closing prices, and they are the
    better estimate of what the rows above will do.
    """
    rep = load_report() or {}
    roi = rep.get("roi_winner", {})
    if not roi:
        return
    st.subheader("What this has actually returned")
    rows = []
    for k, v in roi.items():
        if not v.get("bets"):
            continue
        rows.append({
            "edge": k.replace("edge_", ""),
            "bets": f"{v['bets']:,}",
            "ROI %": f"{v.get('roi_pct', float('nan')):+.2f}",
            "hit %": f"{v.get('hit_rate', float('nan')) * 100:.1f}",
            "break-even %": f"{v.get('breakeven_hit_rate', float('nan')) * 100:.1f}",
        })
    show_table(pd.DataFrame(rows))
    hl = rep.get("headline", {})
    best = hl.get("best_winner_roi_pct")
    st.error(
        "**Every threshold above loses money.** These are walk-forward results "
        "against real Pinnacle closing prices"
        + (f"; the best of them is {best:+.2f}%. " if best is not None else ". ")
        + "The EV column in the table above is the model's own opinion of its "
        "edge; this table is what that opinion was worth when it was tested. "
        "Where they disagree, this one is the evidence. The model's measured "
        "value is as a calibrated probability source, not a staking signal."
    )


PAGES = {
    "Tournament simulation": page_simulation,
    "Match predictions": page_predictions,
    "Betting": page_betting,
    "Backtest performance": page_backtest,
    "Calibration": page_calibration,
    "Elo ratings": page_ratings,
    "Status": page_status,
}

st.sidebar.title("🎾 ATP model")
choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption("ATP main tour + Challenger, 2000–present. WTA deferred.")
PAGES[choice]()
