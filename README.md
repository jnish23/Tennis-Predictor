# ATP Tennis Prediction System

Win-probability, totals and spread models for ATP main tour and Challenger
matches, 2000–present, with walk-forward backtesting, Monte Carlo bracket
simulation, a nightly ingestion job and a Streamlit dashboard.

Architecture follows the NBA system described in CLAUDE.md: LightGBM,
walk-forward validation, SQLite + parquet, Streamlit. Note that the shared NBA
cloud VM that document refers to does not exist — see `deploy/` for what this
actually needs to run (short version: 563 MB to serve, and it runs fine on a
laptop).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m tennis.ingest.download && .venv/bin/python -m tennis.ingest.load && .venv/bin/python -m tennis.ingest.odds && .venv/bin/python -m tennis.features.pipeline && .venv/bin/python -m tennis.models.train && .venv/bin/python -m tennis.models.evaluate
```

```bash
./scripts/run_dashboard.sh
```

## What the numbers say

Walk-forward over 2010–2026, 125,557 matches, each season predicted by a model
trained only on earlier matches.

| | value |
|---|---|
| Log loss | **0.6053** (Elo-only baseline 0.6268) |
| Brier | **0.2097** |
| Accuracy | **66.5%** |
| Totals MAE | **5.17 games** |
| Spread MAE | **4.10 games** |
| Total-sets MAE | 0.47 sets |

Calibration is close to the diagonal across the full probability range.

**The model does not beat the closing line.** On the 38,666 matches with
Pinnacle prices, the closing line's log loss is 0.5736 against the model's
0.5872, and flat-staking every positive-EV bet loses money at every edge
threshold tested (best −2.80%, worst −4.25%; fractional Kelly −2.84%). Split by
season, 2 of 16 full seasons are profitable (best 2017, +8.3%; worst 2023,
−14.0%). This is the expected result — tennis closing lines are sharp — and it
is the honest headline: treat this as a calibrated probability source, not a
betting edge.

**Hit rate is not accuracy, and its benchmark is not 50%.** Accuracy (66.5%)
asks whether the model's favourite won, across every match. Hit rate (41.0% at
edge 0.03) asks whether the *value side* won, on the subset where the model
disagreed with the price — and 69% of those bets are on the market underdog at
an average price of 3.75. Most are meant to lose. The benchmark is the price:
break-even is mean(1/price) = 41.7%, so the model lands **0.64 points short**,
which is precisely the −2.91% ROI. The report and dashboard both carry
`breakeven_hit_rate` beside `hit_rate` so the two are never read apart. Note
that 1/mean(price) is *not* the break-even and understates it badly on a skewed
book — `tests/test_pipeline.py` pins the correct formula.

Accuracy is meaningfully higher on main tour (68.3%) than Challenger (64.8%),
and highest on Grand Slams (74.5%), where best-of-five and large skill gaps make
matches more predictable.

## Layout

```
tennis/
  config.py            paths, category vocabularies
  ingest/
    download.py        fetch + cache every source file
    parse.py           score parsing, retirement/walkover classification
    load.py            clean, standardise, dedupe, load to SQLite
    odds.py            tennis-data.co.uk join (by name; also fixes match dates)
    odds_cbo.py        checkbestodds.com scrape -- historical Challenger prices
    odds_live.py       tennisexplorer capture (own scraper; keeps every snapshot)
    draws_api.py       upcoming-draw fetch (RapidAPI fixtures; needs RAPIDAPI_KEY)
    daily.py           nightly job
  features/
    elo.py             overall + per-surface Elo
    build.py           pre-match feature engine (leakage-safe by construction)
    pipeline.py        full or incremental feature build
  models/
    common.py          feature allow-list, walk-forward splitter, metrics
    train.py           three targets + walk-forward backtest
    evaluate.py        ROI, calibration, breakouts
    predict.py         live scoring for arbitrary matchups
  sim/
    bracket.py         Monte Carlo tournament simulation
    draws.py           bracket reconstruction + past-event replay
  db/schema.py         SQLite schema
dashboard/app.py       Streamlit UI
notebooks/
  model_analysis.py    feature importance + hyperparameter search (paste into Jupyter)
                       selection uses multi-fold time-series CV, holdout untouched
deploy/                systemd unit, crontab, deployment notes
tests/                 leakage guards, parser, simulator invariants
```

## Data

| source | what | coverage |
|---|---|---|
| TennisMyLife | matches, players, stats, tournament metadata | 200,470 matches, 2000-01-03 → 2026-08-02 |
| checkbestodds.com | match-winner odds, **incl. Challengers** | 36,888 matches joined (22,250 Challenger, 14,638 main tour), 2011–2022 |
| tennisexplorer.com | live capture, 4x daily, ATP+WTA+Challenger | own scraper; every snapshot kept, so the last before play is a closing-line proxy |
| tennis-data.co.uk | match-winner odds (12 books) | 64,642 matches priced (82% of main tour; the rest is Davis Cup, Olympics and pre-2001, none of which the source covers) |

Every file pulled is cached under `data/raw/` and never re-fetched unless it
changed upstream. CLAUDE.md flags single-maintainer risk on TennisMyLife (the
risk class that took the Sackmann repos offline); the local cache, not the
website, is the working copy.

### Things the data does not do

Three findings that shape what the system can claim:

1. **Totals and handicap lines on games exist only in the tennisexplorer
   backfill**, and nowhere in tennis-data.co.uk or checkbestodds. Until that
   scrape landed, two of the three models had never been scored against a
   market at all, and the totals/spread "ROI" in the report was a *synthetic*
   test against a naive reference line at −110 — a line so nearly constant
   (std 1.8 games against an actual 7.2) that its large positive ROI reflects
   the baseline's weakness rather than profitability. That synthetic figure is
   still labelled as such everywhere it appears; the real comparison is now in
   `scripts/totals_vs_market.py`.
2. **`ongoing_tourneys.csv` contains only completed matches** — zero rows without
   a score or winner. It cannot supply a bracket before it is played, which
   settles the open question in CLAUDE.md. Past draws are reconstructed from our
   own data instead; upcoming ones need an external feed. See
   [Where draws come from](#where-draws-come-from).
3. **The two file generations disagree about `tourney_date`.** Archived seasons
   stamp every match with the week the event began; the current season stamps
   each match with the day it was played. Keying events on the date therefore
   shattered live events — Roland Garros 2026 arrived as 13 separate
   "tournaments", one per playing day. Events are keyed on
   `(tourney_id, name, level)` with a 30-day gap split instead, since a few
   malformed ids are reused across years (one literal `Rome`, and Davis Cup's
   `D001` spans 25 venues). Chronological order comes from
   `(tourney_date, round_idx, match_num)`; exact per-match dates are kept in
   `match_date`.

### Cleaning decisions

- **Retirements, walkovers and defaults are labelled, not dropped.** A
  retirement is a real win, so it counts for the winner model and for Elo; but
  its scoreline is truncated, so it is flagged `totals_usable=0` and never
  becomes a totals or spread label. Walkovers update nothing — no play happened.
- **Categorical encodings differ from the CLAUDE.md description.** The live
  files use `250/500/M/G/D/A/O/F/C` for `tourney_level` (not `A/G/D/F`) and
  `I/O` for indoor (not Yes/No). Both are mapped to a stable internal vocabulary
  in `config.py`.
- **Player IDs were verified, not assumed.** Of 5,640 ATP IDs, 2,098 appear in
  both main-tour and Challenger files with only 4 name conflicts, all of which
  are spelling variants or a name change for the same player. IDs are safe to
  join on.
- **Four Challenger tournaments carry a wrong year in `tourney_id`** (2008
  events filed as `2024-…`). `tourney_date` is correct in those rows, so season
  is always derived from the date, never the ID.

## What was tried, and what it bought

Feature work is measured on an untouched holdout with a paired bootstrap, not
adopted because it sounds sensible — and then confirmed on the full walk-forward
before it is kept. That second step is not ceremony: one addition below cleared
the holdout at P = 0.99 and reversed sign under walk-forward.

**Handedness is already handled.** Residuals by matchup are flat (L vs R
z = −0.09, R vs L z = +0.42), so the tree learns the `p1_lefty × p2_lefty`
interaction on its own. A *player-specific* lefty effect does exist — 581
players with ≥20 matches vs a lefty show z-score sd 1.056 against 1.000 expected,
χ² p = 0.023 — but it is small, and with 20–50 such matches per player a
"win rate vs lefties" feature would be mostly noise. Not added.

**Four feature groups were A/B'd** on a 2023+ holdout. Individually none cleared
noise; together they did:

| addition | Δ holdout log loss | verdict |
|---|---|---|
| EWMA form (5- and 20-match half-lives) | −0.00043 | noise on its own |
| opponent strength faced | +0.00004 | nothing |
| clutch (deciding sets, tiebreaks) | −0.00007 | nothing |
| surface streak | −0.00012 | nothing |
| **all four** | **−0.00090** | CI [−0.00164, −0.00015] — real |

Kept, and the full walk-forward agrees: log loss 0.60696 → **0.60519**, accuracy
66.18% → **66.40%**. EWMA earns the most gain share of the four (7.9%), which
supports the intuition that a flat 25-match mean discards recency — but the
honest size of the win is about a third of what hyperparameter tuning bought.

**A short-memory Elo passed the holdout and failed the walk-forward.** The
all-time book has long memory by design, so a constant-K companion rating plus
`d_elo_trend` (the gap between the two) looked like a genuine hole in the
feature set — and the diagnostics agreed. `d_elo_trend` regressed on the 44
existing recency, opponent-strength and rank features came back at R² = 0.35, so
two thirds of it was signal nothing else carried, and `d_elo_fast` landed 14th
of 157 by gain.

On a 2023+ holdout it won cleanly: **−0.00041 log loss, 95% CI [−0.00077,
−0.00009], P(better) = 0.992**, and a K sweep from 16 to 64 came back negative
at every value, so it was not a lucky constant.

Under walk-forward it reversed: **+0.00023, CI [−0.00005, +0.00052],
P(better) = 0.054, 6 of 17 seasons improved.** Reverted.

The explanation is also the lesson. A single holdout trains one model on
pre-2023 and asks it to predict four years forward; a recency feature is worth
real money to a model that stale. Walk-forward refits every season, so the
all-time Elo is already current and the trend has nothing left to add. **For
recency features specifically, a fixed holdout flatters — it measures a staleness
the production system does not have.** The four-feature bundle above survives
because it was confirmed on the full walk-forward, not because it cleared the
holdout. `scripts/experiment_recent_elo.py` reproduces both halves.

**A better Elo bought nothing.** Three changes were measured standalone on
124,899 matches (`scripts/experiment_elo.py`, `experiment_elo2.py`): K 250 → 200,
surface ratings seeded from the overall rating and blended by surface match
count, and decay after a layoff. Standalone the surface fix is large — surface
Elo was *worse than ignoring surface entirely* (0.6366 against 0.6297) and
became the best of the three at 0.6243.

It did not transfer. Rebuilt end to end, the Elo-only baseline improved exactly
as predicted (0.63057 → **0.62678**) and the model did not move: log loss
0.60519 → 0.60525, paired bootstrap CI [−0.00040, +0.00038], P(better) = 0.54,
9 of 17 seasons improved. LightGBM already sees `elo_surf` *and* `elo_surf_n`,
so it had learned to discount a thin surface rating on its own — the blend just
moved that work upstream. Kept anyway, because the ratings are displayed and a
young player's clay rating reading 1500 was simply wrong, but it is not an
accuracy gain and is not claimed as one.

**Deeper history helps, but not enough to be worth fetching.** The source has
files back to 1967; the system ingests from 2000. Truncating the training window
at each fold (`scripts/experiment_history.py`, 72 model fits) shows log loss
falling monotonically with window length — 0.61536 at 4 seasons through 0.61047
at all of them, with "all" or 14 winning in all 12 test seasons. So tennis has
not drifted enough for old matches to mislead. But the marginal gain per extra
4 years decays from −0.0018 to −0.0006, and 2025 is already better on a 14-year
window than on everything, so five more years extrapolates below fold noise.
More to the point it would change nothing about today's predictions: every
active player's ratings and rolling features are long since converged from
2000+ data. Not pursued.

**Live odds are collected by our own scraper, not a third party's.** The
obvious shortcut was the `Mriganka-codes/tennis_data` repo, which wraps the same
site on a 6-hourly GitHub Action. Three reasons it is not usable as code: it
publishes **no licence**, so it is all rights reserved and cannot be copied or
vendored; its `matches.json` is **overwritten every run**, so it accumulates no
history, which is the entire point here; and its documented behaviour is not its
actual behaviour — `main_only` claims to drop Challengers but the filter list
omits "challenger", so they pass through by accident. It also disables TLS
verification with no need. Its *published data* is fine to read, and
`backfill_from_github` mines its commit history — 610 commits since March — to
seed roughly five months we would otherwise have had to wait for.

`tennis/ingest/odds_live.py` scrapes tennisexplorer directly (robots.txt allows
the match pages) into `odds_snapshots`, one row per capture rather than one per
fixture. That is the design decision that matters: repeated captures make the
last price before play a genuine closing-line proxy, where checkbestodds carried
no timestamp at all. Measured on the first backfill — 14,208 snapshots over
5,149 fixtures, 2.8 captures each, 50% Challenger; 1,250 resolved to results at
a median overround of **1.071**, a real single-book margin rather than a
cross-book maximum, with the favourite winning 66.4%.

One parsing trap is worth recording because it fails silently and backwards: the
winner's price cell is relabelled `coursew` **only once a result exists**, so
reading `coursew` as "player one" captures finished matches and skips every
upcoming one. The first version returned prices for two completed main-tour
matches and none of the 39 upcoming Challengers on the same page. Prices are now
read positionally and a test pins all three states.

**The Challenger softness did not replicate on 2026 prices — but the test had
no power to confirm it.** `scripts/backtest_te.py` runs the betting backtest on
the tennisexplorer captures (4,358 matches, Mar–Aug 2026), the first prices here
that are single-book and timestamped. On Challengers every strategy lost
**−13% to −15%**, worse than backing every underdog (−11.5%) and worse than
random (−9.2%); the model steers into underdogs at an average price near 3,
which is exactly where the favourite-longshot bias loads the vig. Main tour's
best was **+1.70% on 702 bets**, CI [−9.46, +13.06] — and under the null the
best of ten strategies clears that 97% of the time. Zero of ten had a CI
excluding zero.

The ensemble barely fires at all: with weights taken from the 2011–22 era and
applied forward, only **57 of 3,237** Challenger matches show positive EV. That
is the cleanest statement of the problem — a correctly weighted ensemble is
close to the price, and the model's information is not worth 6.7% of vig.

The replication itself is inconclusive rather than negative. Challenger
`w_model` went +0.155 (z = 4.2, n = 22,250) to −0.062 (z = −0.6, n = 3,237), but
the same effect at the smaller n would only give **z ≈ 1.60** — the sample was
never large enough to confirm it. Two confounds also separate the datasets: era,
and price type. checkbestodds was a cross-book best-odds composite at 2.3% vig;
these are one book at 6.7%, and proportional devigging leaves that book's
favourite-longshot bias in place (measured here: favourites −5.4%, underdogs
−11.5% against a 6.7% floor). That bias flows straight into the ensemble
regression, which is a concrete argument for implementing Shin devigging before
treating the non-replication as settled.

**The models show real closing-line value — an order of magnitude short of
profitable.** Placing one bet per match at Pinnacle's *opening* games line and
scoring it against the close (`scripts/backtest_lines.py` and the CLV analysis
alongside it):

| market | bets | mean CLV | 95% CI | beat the close | z |
|---|---|---|---|---|---|
| totals | 12,917 | **+0.33 pts** | [+0.29, +0.37] | 52.9% | +6.6 |
| spread | 14,979 | **+1.11 pts** | [+1.05, +1.17] | 60.6% | +25.9 |

CLV is the right success metric because it scores every bet rather than only
the ones that won, and it is what the market itself later agrees with. Both are
overwhelmingly significant, both CIs exclude zero by a wide margin, and CLV
*rises* with the edge threshold (totals: +0.298 at edge 0, +0.331 at 2%, +0.359
at 5%), which is what signal looks like and noise does not.

It is still not enough. The vig on those same lines is **3.5-3.8 points**, so
totals covers about a tenth of what it needs and spread about a third. The ROI
figures agree: against Pinnacle's tighter 4.0% margin, totals runs -1.79% to
-0.24% and spread -0.55% to +0.63%, with every confidence interval spanning
zero. Against the cross-book average at 7.3% vig both are clearly negative
(-2.8% and -4.3%), though still well ahead of backing every over (-8.9%) or
every under (-6.4%), so the model is extracting real value from the ladder --
just less than the house charges to play.

Two practical notes. Betting **every** positive-EV rung rather than the single
best one costs about two points (totals -4.74% against -2.77%), because a ladder
is a menu, not a dozen independent bets; for the same reason every interval here
is bootstrapped **by match**, since rungs on one match settle on one scoreline.
And some of the CLV may be timing rather than modelling: our features are
as-of-match while an opening line can be days old, so part of what looks like
foresight is simply more recent information.

**Totals and spread, measured against a market for the first time.** On
29,158 matches (2023–2026) with residual distributions fitted only on earlier
seasons, the market wins both — but by less than the framing "MAE 5.17 games"
suggests, and by almost exactly the margin the winner model concedes:

| market | n (match-lines) | model | market | gap |
|---|---|---|---|---|
| totals (games) | 370,452 | 0.66050 | 0.65000 | **+0.0105** |
| — Challenger | 199,113 | 0.66498 | 0.66124 | **+0.0037** |
| — main tour | 171,339 | 0.65530 | 0.63693 | +0.0184 |
| spread (games) | 345,843 | 0.64847 | 0.63297 | **+0.0155** |
| — Challenger | 197,943 | 0.66911 | 0.65363 | +0.0155 |
| — main tour | 147,900 | 0.62085 | 0.60533 | +0.0155 |

For scale the winner model concedes 0.0135 to Pinnacle, so all three models sit
a comparable distance behind their own market. The Challenger totals gap of
0.0037 is the smallest of any comparison in this project, and it is the third
independent market on which the Challenger-versus-main-tour pattern has shown
up.

Getting there needed two orientation traps fixed, both worth knowing about
before touching `odds_quotes`. The k1/k2 columns are **winner-first** (99.8% of
resolved matches), while the handicap `line` is quoted against a *pre-match*
ordering, so the two agree only 51% of the time — pooled, that cancelled the
signal to a correlation of +0.09 and a market that scored worse than a coin
flip. The ladder's own direction identifies whose frame the line is in, which
needs no re-parse and lifted the correlation to +0.43; moving from the
winner's frame into the backtest's hash-chosen p1 removed the remaining
12-point level bias (market 0.500, actual 0.501). Totals need neither fix,
which is why that half worked immediately: over/under has no player
orientation.

**Shin devigging is now the default, and it did not change any conclusion.**
Proportional devigging assumes the margin is split evenly between the two
sides; books load it onto the longshot, so proportional over-states underdogs.
`tennis/models/devig.py` implements Shin and power alongside it, and the choice
was made on outcomes rather than literature: scored against realised results on
**523,879 quotes across 18 books**, Shin beat proportional in **16 of 17**, and
its advantage tracks how much margin a book charges (**r = 0.72, p = 0.001**).
The single exception is Betfair — an exchange, with no bookmaker margin to
misallocate, which is the control the theory would ask for.

The effect is real but small: Pinnacle's market log loss moves 0.57364 →
0.57349. It was adopted because it is the *benchmark* every model-vs-market
comparison is measured against, so a bias there biases all of them.

It was also expected to matter for one specific open question — whether the
favourite-longshot bias was what made the Challenger softness fail to replicate
on 2026 prices. It was not. Re-running that comparison under both methods moves
the z-scores by **at most 0.2** (checkbestodds Challenger 4.2 → 4.0;
tennisexplorer Challenger 0.5 → 0.5). The non-replication stands on its own and
the devigging confound is closed rather than resolved in our favour.

**Challenger lines are measurably softer — and still not beatable.** The
biggest gap in the data was that tennis-data.co.uk prices zero Challenger
matches, so half the match volume had no price and the "is the thin market
soft?" question was untestable. `tennis/ingest/odds_cbo.py` scrapes
checkbestodds.com (robots.txt permits `/tennis-odds/`; pages cached, requests
spaced) for 234 Challenger and 89 ATP tournaments, 2011–2022 — **22,250
Challenger matches priced, from none.**

Softness is confirmed three independent ways. Median overround **2.4% against
0.7%** on main tour from the same source; market calibration error **0.0220
against 0.0103**, with the textbook favourite-longshot shape (longshots under
10% implied win 5.8pp less often than priced, favourites over 90% win 4.5pp
more); and our probability earns a real ensemble weight beside the price,
**0.155 at z = 4.2**, where the same fit on main tour gives 0.062 at z = 1.3.
The Challenger line genuinely does not know things our model knows.

None of that converts. Flat-staking the model loses **−1.74% to −2.27%** at
every edge threshold. Betting a walk-forward model+market ensemble gets to
**+2.48% on 2,310 bets at edge 0.05, 95% CI [−1.05%, +6.13%]** — indistinguishable
from zero. The softness is real and the margin eats it: the market is twice as
wrong and charges three times as much for being wrong.

Two caveats bound how hard this can be read, and both matter. The price is
**best odds** — the maximum per side across books, an upper bound no single
account could take — so these ROIs are ceilings. And best odds is contaminated:
the max is taken per side independently, so one bookmaker with its sides
transposed poisons a column (Cecchinato–Sergeyev 2013 listed 3.47/26.00 while
eleven of twelve books said ~3.20/~1.29). The overround floor that removes it is
**0.96, calibrated against Pinnacle** rather than assumed — on 19,525 matches
priced by both, agreement by band runs 0.486 at [0.5,0.7) up to 0.998 at
[1.0,1.05), and contamination stops dominating around 0.96. A 1.0 floor looks
safer and is worse: best odds sits below 1.0 legitimately whenever a genuine
cross-book arbitrage exists, so cutting there discards 11,555 good rows —
exactly the best-priced ones an ROI depends on.
`scripts/experiment_challenger_market.py` reproduces it.

**Blending with the closing line adds nothing.** The model and the Pinnacle
close agree closely — Pearson 0.943 on the probabilities, 0.935 on the logits,
which is the scale a blend works on — but they are not interchangeable: the
median gap is 4.7pp, only 23% of matches land within 2pp, and 17% differ by more
than 10pp. So there is plenty of disagreement to arbitrate.

The market wins all of it. Split by how far the two diverge, the market has the
lower log loss in **every** bucket, and the gap *widens* with the disagreement —
where the model is 10pp above the price it scores 0.653 against the market's
0.601, and 10pp below, 0.661 against 0.613. The model is most wrong exactly
where it is most confident the market is wrong.

Fitting `y ~ logit(model) + logit(market)` gives the model a coefficient of
**0.054** (z = 1.8) against the market's **0.988**, a likelihood-ratio p of 0.07
— not significant even in-sample, with the answers visible. The best fixed blend
puts 5% on the model and buys 0.00002 log loss. Refit walk-forward the ensemble
scores 0.57642 against the market's 0.57645: Δ −0.00003, 95% CI [−0.00032,
+0.00023], P(better) = 0.62. The weight also decays across the backtest, 0.118 in
2012 to 0.055 in 2025.

No segment rescues it — no level clears z = 2, and the largest nominal weight is
at Grand Slams (0.108), the *most* heavily traded events, which is the opposite
of a soft-market story and reads as noise. The obvious test cannot be run at all:
tennis-data.co.uk prices **zero** Challenger matches, so the thinnest markets in
scope are invisible here. `scripts/experiment_market_ensemble.py` reproduces it.

**The ceiling.** The closing line is the practical bound, and the gap is
measurable: 0.5873 against 0.5736 log loss, 1.6pp of accuracy, on the priced
subset. These features closed roughly a tenth of it. The headline 66% is also
held down by Challengers (64.8%) versus main tour (68.5%), which are inherently
less predictable rather than badly modelled.

The remaining structural gap is that this predicts matches directly and ignores
tennis's scoring hierarchy. Estimating serve/return point probabilities and
propagating them through game → set → match is the approach that tends to close
on market pricing, and is a rebuild rather than a feature.

## Leakage control

CLAUDE.md treats leakage as a standing risk, so it is enforced structurally
rather than checked once. The feature engine walks matches in order and, for
each, reads player state to emit a row *before* folding that match's result in.
Nothing a model sees can contain information from its own match or any later
one.

Three tests in `tests/test_leakage.py` back this up, the strongest being
**prefix invariance**: features built over the first N matches must be
identical to the first N rows of features built over 2N matches. If anything
peeked forward, the two runs would diverge.

`FEATURE_COLS` in `models/common.py` is the single allow-list of model inputs,
and a test asserts no same-match stat column appears in it.

## Features

Elo (overall + per surface, K decaying with match count, surface ratings blended
toward overall by surface match count, ratings decaying after a layoff) ·
rolling win rate over
10/25/50 matches · surface-specific form · serve and return rates (1st-in,
1st-won, 2nd-won, ace and double-fault rate, break points saved, hold%, break%)
· head-to-head overall and by surface · rest days, matches and minutes in the
last 14/30 days · rank, rank points, seed, age, height, handedness · and
pairwise differences of all of the above. 126 columns.

## Models

| target | type | note |
|---|---|---|
| winner | LightGBM binary + isotonic calibration | calibrator fitted on a held-out *earlier* slice, never on test data |
| totals | LightGBM regression on **total games** | |
| spread | LightGBM regression on signed game margin | p1-relative |

**Why games, not sets, for totals.** In a best-of-three match "total sets" takes
only the values 2 and 3, which makes it very nearly a restatement of the winner
model. Total games spans roughly 12–40, carries real distributional information,
and is what totals markets actually price. Total sets is still produced as a
secondary output for the dashboard.

The totals and spread models additionally receive absolute values of the
mismatch features (`abs_d_elo` and friends): a 200-point Elo gap shortens a
match whichever player holds it.

Predictions are **symmetrised** — scored in both orientations and averaged — so
that P(A beats B) + P(B beats A) is exactly 1. Without this the raw ensemble
sums to about 0.95 and an answer would depend on which player was entered first.

## Tournament simulation

Monte Carlo, per CLAUDE.md — never a deterministic round-by-round chain, which
would collapse the uncertainty into one fictional bracket by round 3 or 4. Each
of several thousand playthroughs resolves every match by a random draw weighted
by the model's probability. Features stay frozen at pre-tournament values for
the whole run, so the win-probability matrix is computed once and reused.

### Where draws come from

| source | used for | needs |
|---|---|---|
| **Reconstructed from our own data** | any completed event | nothing |
| **RapidAPI fixtures feed** | events drawn but not yet played | `RAPIDAPI_KEY` |

Between them these cover every event that exists, so the page opens on the live
feed and has no hand-entry mode. Simulation count (10,000) and the feed's
look-ahead window (3 days) are constants in `dashboard/app.py`, not controls —
10k playthroughs hold Monte Carlo error on a title probability inside a tenth of
a point, and since no draw publishes more than about a day ahead, a longer
window returns the same tournaments.

**Past draws need no external source and no typing.** The first round of a
completed tournament *is* the draw, and the result graph — who beat whom —
recovers the full bracket tree. Validated at **99.3% across 5,158 events**; the
remainder are malformed upstream (one "32-draw" carries 55 players) or team
events with no bracket. 1,170 tournaments are replayable from 2022 alone.

`match_num` is deliberately **not** used to recover bracket order. It encodes
the bracket in some files and not others — the naive rule held for only 712 of
5,662 events — so the result graph is the reliable signal.

Replaying a past event rebuilds feature state *and rankings* to the day before
it began. Today's Elo already contains that tournament's results, and simulating
with it would be the same leakage class CLAUDE.md flags as a standing risk. The
rebuild takes ~8 seconds and is cached per event.

The replay view marks the rounds each player actually reached and surfaces the
biggest misses — players who reached a round the model gave under 10%.

### The bracket view

A bracket that fills in as results arrive. Ties whose players are both decided
carry a win probability, a projected total and a handicap quoted on the favoured
player; everything downstream stays **TBD**. It deliberately never names a likely future opponent and prices that
matchup — doing so is the deterministic-chain error the Monte Carlo exists to
avoid, and CLAUDE.md rules it out explicitly.

So before play, round 1 is priced and nothing else is. Once round 1 finishes,
round 2's pairings are real and get priced. A half-finished round leaves its
dependent slots TBD rather than guessing.

Connecting elbows join the two ties feeding each slot in the next round, and
horizontal rules mark the quarters of the draw — the brighter middle one splits
the halves that can only meet in the final. Both are pure CSS with no measuring:
ties are wrapped in pairs so the elbow spans a known fraction of the pair, and
because ties are distributed evenly down a column the quarter boundary lands at
exactly 25/50/75% in every round, whether it holds 64 ties or 4. Two details
make it exact rather than approximate — byes now carry a "no match played" note
so every tie is the same height (without it the R64 elbows sat 6px off), and the
elbow is offset by `gap/4`, which is precisely how `space-around` distributes
free space either side of the gap. Connectors are dropped when the round filter
hides a round in the middle, since a line spanning it would assert a
progression that skips a round.

Matches already played keep their pre-match probability and are graded against
what happened. The number shown is what the model gave the **actual winner**,
which separates a coin flip from a real miss on one axis: 59% is a clean call,
45% is barely wrong, 12% is a genuine upset. They are shaded accordingly, so a
wall of red does not appear for matches the model essentially called even.

Everything the view needs is held in session state. Earlier the results lived
inside the fetch button's block, so changing the round filter re-ran the script,
found the button `False`, and wiped the whole view — the filter was unusable.

Alongside it, the advancement table holds the **pre-tournament forecast** for
every round, plus a live title probability and the move since.

The round columns deliberately do not update. Current per-round probabilities
degenerate as an event runs: measured two rounds into a Masters, R64 and R32
had zero spread with every survivor pinned at 100%, and by the semi-finals five
of seven columns say nothing at all. Pre-tournament columns never degenerate,
and for rounds already decided they become the forecast being graded — *we gave
him 82% to clear R32, he went out*.

The cost is that "chance to reach the final **from here**" is no longer shown;
only the title probability is live. Headers are labelled `(pre)` / `(now)`
rather than relying on a caption, since mixing time bases in one row is the
obvious way to mislead. Players already out keep their pre-tournament numbers —
still true as a forecast — but their rows are dimmed and sorted below the
survivors, so a 12% semi-final chance is not read as live.

By default the table shows survivors only. A **Show** toggle adds the eliminated
players back, and **Max rows** caps the length. The cap is stored per scope so
each keeps its own default — sharing one made the toggle look broken, since with
32 still in and a cap of 32 switching to *Everyone* changed nothing visible.

Both simulation runs use the same model and features and differ only in which
results are pinned, so the move is caused by results alone.

The bracket carries every number a separate first-round lines table used to, for
every live tie rather than round 1 only, so that table is gone. A "biggest
movers" chart went with it: it plotted the Move column, and for an eliminated
player the move is mechanically minus their pre-tournament probability, so half
the chart restated *they lost*.

**Upcoming draws are the one thing no source we hold provides.**
TennisMyLife's ongoing files contain only completed matches (zero rows without a
score or winner), and the ATP site returns 403 behind bot protection.
`tennis/ingest/draws_api.py` covers that gap via the RapidAPI tennis feed.

What that feed actually does, measured rather than assumed:

- The **draw sheet** endpoint (`/ms-api/.../draws`, in the *Advanced* section of
  the docs) returns the full bracket — including players with a first-round bye,
  and with true bracket positions.
- `/fixtures/tournament/{id}` is a fallback only. It publishes a match once both
  players are known, so **everyone holding a bye is absent**: in a 96-draw
  Masters that dropped all 32 seeds and left the model handing the title to a
  qualifier. The plain "Results" endpoint returns completed matches only.
- **It does not publish draws at ceremony time.** A tournament eight weeks out
  returns zero fixtures, and a 17-day date-range query returned matches for only
  the next two days. In practice the draw lands about a day before play.

So this supports simulating an event **the evening before it starts**, not a
week ahead. Tournament ids are discovered from the fixtures feed rather than the
calendar endpoint, which proved incomplete for the near term (queried in August,
its earliest entry was late September).

Responses paginate at 10 rows but honour `pageSize`, so one large draw is a
single request. Budget is ~1 request per tournament plus one discovery call;
a local guard caps daily spend and caches responses so repeated dashboard clicks
cost nothing.

Feed names resolve to our ATP ids at 100% on the draw tested (60/60), handling
both `"Jannik Sinner"` and `"Sinner J."` forms plus compound surnames
(`"Daniel Merida Aguilar"` → `Daniel Merida`). Unresolved names are surfaced and
excluded rather than silently guessed.

Bracket positions come from the draw sheet, so later-round matchups are the real
ones. Only the fixtures fallback approximates the order, and the dashboard says
so when it is used.

> **This overrides a CLAUDE.md instruction.** That file says not to use
> tennis-api.com, because its pricing page showed three different numbers for
> one tier and it resells rather than originates data. This is that vendor,
> reached directly through RapidAPI, where pricing is unambiguous ($0/$29/$59/$99)
> and the free tier is a hard-limited 50 requests/day with no overage billing.
> The "not a primary source" objection stands, so every fetched draw is validated
> against our own match data once play begins, and historical work never touches
> the feed. Put `RAPIDAPI_KEY` in a gitignored `.env` at the repo root; it is
> loaded automatically and never logged.

## Database

SQLite (`data/tennis.db`) — `matches`, `players`, `tournaments`, `rankings`,
`odds`, `elo_state`, `draws`, `sim_results`, `predictions`, `backtest`. The wide
feature frame lives in `artifacts/features.parquet`; SQLite holds the state
needed to resume.

Every table that could differ between tours carries a `tour` column even though
only `atp` is populated, so adding WTA is a load-time filter change rather than
a migration. WTA files exist upstream (1990–2026) but are deferred per CLAUDE.md
because the maintainer describes them as less reliable than the ATP data.

**Feature updates are incremental.** The engine pickles its running state, and
the nightly job replays only new matches against it. A test asserts the
incremental path reproduces a full recompute bit-for-bit.

## Nightly job

```bash
./scripts/daily.sh              # fetch, reload, extend features (~40s)
./scripts/daily.sh --retrain    # also refit models and rerun the backtest
```

The match table is fully reloaded each run rather than upserted. It takes ~25s
for 200k rows and it is the only way upstream corrections — fixed scores,
backfilled stats — actually reach us. Feature computation is the part that is
incremental. Retraining is weekly, not daily: one day of matches cannot move a
model fit on 200k rows.

## Dashboard

Six pages: match predictions (all three targets), tournament simulation over a
live or replayed draw, backtest performance over time with breakouts,
calibration plots, Elo ratings, and pipeline status.

The ratings page offers **All-time** and **Season to date**. Season to date
shows how far the career rating has moved since the season opened.

The season boundary is **25 December**, and December is trickier than it looks:
it holds events at *both* ends of the tour year. The Next Gen Finals close a
season (18 Dec 2024, 22 Dec 2025) while Doha, Adelaide, Chennai, Brisbane, Pune,
Hong Kong and the United Cup open the next (26-31 Dec). Across all 26 seasons in
the database, 20-21 and 23-25 December are completely empty — last closer on the
22nd, earliest opener on the 26th — so a 25 December cut separates them with
room either side. An earlier 1 December cut swept both Next Gen editions into
the season they actually end.

It is deliberately not a rating *restarted* at the opener, which is the obvious
implementation and does not work. Reset everyone to 1500 and beating a field is
worth the same whoever the field is; Challenger and main-tour players almost
never meet, so a single season never prices the two pools against each other.
Built that way, a 38-11 Challenger season against median opponent rank 242 rated
*above* a 27-7 season against the top 60 — the book failing to tell the pools
apart, not a fact about the players. Moving the calibrated rating instead means
beating rank 242 when you are rated to beat rank 242 correctly moves nothing,
and the column reads as real over- or under-performance.

Hosting is undecided — there is no existing VM to deploy onto. `deploy/` has
measured resource numbers and the three realistic options (run locally, one
small VM, or managed hosting).

## Does in-tournament form matter? (measured)

`scripts/experiment_feature_recency.py` answers a question CLAUDE.md flags:
when simulating from round N, should a player's features include the rounds they
have already played this week, or stay frozen at pre-tournament values?

It is a *controlled* test. Both arms pin exactly the same completed matches, so
the only difference is feature recency. (Comparing pre-tournament against
conditional would be meaningless — simulating from a later round always scores
better simply because less is unknown.)

Across 39 Grand Slams and Masters, 2023–25, 146 comparisons:

| | frozen | updated | delta |
|---|---|---|---|
| Log loss | 0.40468 | **0.39989** | **−0.00478** |
| Brier | 0.13177 | **0.13024** | −0.00154 |

Updated wins 84/146 (Wilcoxon p = 0.047). The headline is marginal, but the
gradient is the real signal — the benefit grows monotonically the deeper you
start:

| simulate from | delta (log loss) |
|---|---|
| round 2 | −0.00070 |
| round 3 | −0.00281 |
| round 4 | −0.00738 |
| round 5 | **−0.00944** |

That is exactly the pattern the mechanism predicts: the further into an event,
the more a player's pre-tournament form has gone stale. Noise does not usually
produce a clean four-step monotonic trend. The dashboard therefore defaults to
updated features.

Two caveats kept in view: the 146 comparisons are not independent (one event
contributes several rounds), so the p-value is optimistic; and this tests
feature state *at the conditioning point*, with features still frozen **during**
each simulated run. CLAUDE.md's standing default is about the latter and remains
untouched — though this result is reason to test it too.

Rerun with `python scripts/experiment_feature_recency.py` (~20 min; `--events 5`
for a quick look).

## Model analysis notebook

`notebooks/model_analysis.py` (paste into Jupyter — `# %%` cells) covers feature
importance and hyperparameter tuning. Two things it does deliberately:

**Importance is reported three ways**, because gain alone misleads. Grouped by
feature family, Elo takes ~46% of gain and by far the largest permutation hit;
serve/return takes 18% of gain but shuffling it barely moves holdout loss (31
correlated columns splitting credit for information Elo already carries); and
head-to-head is worth essentially nothing on either view — meetings between any
two players are too rare for it to be more than mostly zeros.

**Tuning selects on several validation seasons averaged, never one block.** A
random CV would select using matches that happen after the ones it scores, and
even a single time-ordered validation block proved too noisy to rank
configurations reliably. Measured across 14 random configurations:

| selection criterion | rank correlation with holdout |
|---|---|
| single 2021–22 block | ρ = +0.65 (p = 0.011) |
| **mean of 4 expanding-window folds** | **ρ = +0.83 (p = 0.0002)** |

Selection therefore averages four folds. The notebook computes both criteria and
prints their rank correlations, so it verifies this rather than asserting it.
In that particular run both criteria happened to pick the same config — with a
flat response surface and few candidates the better ranking did not change the
winner. The reliability gain is the durable result; it matters more as trials
grow. The whole prize from picking well among reasonable configs is ~0.001 log
loss, so Bayesian optimisation (Optuna, included as an option) buys wall-clock
rather than accuracy.

A paired bootstrap on the holdout decides whether any improvement is real —
on ~34k matches, log-loss differences under ~0.002 are noise, and the notebook
says so rather than letting a lucky draw get adopted.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

40 tests: leakage guards, score-parser edge cases (retirements, walkovers,
defaults, tiebreaks, malformed scores), simulator invariants, prediction
symmetry, incremental-vs-full equivalence, and draw reconstruction (including a
regression guard on the 99.3% recovery rate).
