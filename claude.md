# Tennis Prediction Model

Production tennis match prediction system: ATP win probability, totals, and spread models, backtested and served via a dashboard. Mirrors an existing production NBA prediction system (LightGBM, walk-forward backtesting, Streamlit dashboard on a cloud VM). Default to that architecture unless tennis-specific requirements argue otherwise, and flag any place you deviate from it.

## Scope

ATP tour only for now: main draw and Challenger level, 2000 onward. WTA is deferred, not dropped — keep the schema and pipeline tour-agnostic where that costs nothing extra, since WTA is still the eventual target.

## Data sourcing

Decided; don't re-evaluate vendors from scratch.

- **Primary source: TennisMyLife** (stats.tennismylife.org/tennis-match-database). Yearly CSVs for ATP main tour, Challenger, and qualifying, 2000 onward, discoverable via `https://stats.tennismylife.org/api/data-files` (JSON listing) or `/api/download-all` (bulk ZIP). WTA files exist on the same site but the maintainer describes them as not yet as reliable as the ATP data — that's the reason WTA is deferred.
- **Ongoing updates:** `ongoing_tourneys.csv` and `challenger_ongoing_tourneys.csv` on the same site, updated close to real time. Use these for the daily ingestion job.
- **Schema:** closely matches the standard Sackmann-style column layout, but player IDs are official ATP IDs directly, which simplifies identity resolution.

    Database Columns Include:
    tourney_id: Tournament ID based on ATP database
    tourney_name: City where the tournament was played
    surface: Hard, clay, grass, carpet
    draw_size: Tournament draw (128, 64, 32, 16, 8, 4)
    tourney_level: G (Grand Slam), A (ATP Tour), D (Davis Cup), F (Masters/ATP Finals)
    indoor: Yes/No
    tourney_date: Week of the tournament (YYYYMMDD)
    match_num: Match number in the tournament
    winner_id: ATP player ID of the winner
    winner_seed: Seed of the winner
    winner_entry: How the winner entered the tournament (e.g., Q, WC)
    winner_name: Full name of the winner
    winner_hand: Playing hand of the winner (R/L)
    winner_ht: Height of the winner in cm
    winner_ioc: Country code of the winner
    winner_age: Age of the winner at match time
    winner_rank: ATP ranking of the winner at match time
    winner_rank_points: ATP ranking points of the winner at match time
    loser_id: ATP player ID of the loser
    loser_seed: Seed of the loser
    loser_entry: How the loser entered the tournament
    loser_name: Full name of the loser
    loser_hand: Playing hand of the loser (R/L)
    loser_ht: Height of the loser in cm
    loser_ioc: Country code of the loser
    loser_age: Age of the loser at match time
    loser_rank: ATP ranking of the loser at match time
    loser_rank_points: ATP ranking points of the loser at match time
    score: Final match score (set by set)
    best_of: Number of sets (3 or 5)
    round: R128, R64, R32, R16, QF, SF, F
    minutes: Match duration in minutes
    w_ace: Aces by winner
    w_df: Double faults by winner
    w_svpt: Total serve points by winner
    w_1stIn: First serves in by winner
    w_1stWon: First serve points won by winner
    w_2ndWon: Second serve points won by winner
    w_SvGms: Service games played by winner
    w_bpSaved: Break points saved by winner
    w_bpFaced: Break points faced by winner
    l_ace: Aces by loser
    l_df: Double faults by loser
    l_svpt: Total serve points by loser
    l_1stIn: First serves in by loser
    l_1stWon: First serve points won by loser
    l_2ndWon: Second serve points won by loser
    l_SvGms: Service games played by loser
    l_bpSaved: Break points saved by loser
    l_bpFaced: Break points faced by loser

- **Single-maintainer risk:** same risk class that took Jeff Sackmann's GitHub repos offline (they're gone — don't build against them). Cache every CSV pulled, historical and ongoing, into your own storage as it's ingested rather than treating the source as the only copy.
- **Backtesting/odds data:** tennis-data.co.uk (historical results and odds across multiple bookmakers, ATP back to roughly 2001), unaffected by the above.
- **Paid-feed trigger:** only add a paid API if walk-forward backtesting shows a measurable accuracy cost from gaps or lag in the free sources. api-tennis.com's Starter plan ($40/month, 8,000 req/day) is the most coherent option found so far if that trigger fires — verify pricing directly first. Don't add a paid feed pre-emptively.
- Neither TennisMyLife nor tennis-data.co.uk is confirmed to expose a tournament's draw before it's played. Check whether `ongoing_tourneys.csv` populates the bracket early enough to use; don't assume it does.

## Modeling

Three targets, sharing a feature pipeline: match winner (calibrated probability), total games/sets, spread/handicap. LightGBM by default, consistent with the NBA system.

**Data leakage is a standing risk here, not a one-time check.** It broke the NBA model's early results and cost real time to trace. Any feature computed using information not available before the match starts (same-match stats, post-hoc rankings) is a leak — watch for this whenever a model is touched, not just at initial build.

## Tournament simulation

Bracket forecasting uses Monte Carlo simulation (thousands of simulated playthroughs, each match resolved by a random draw weighted by the model's probability), never a deterministic round-by-round chain — that collapses uncertainty into a fictional bracket by round 3 or 4.

Two standing implementation defaults:
- Features stay frozen at pre-tournament values within a single simulated run. Only revisit if backtesting shows this is visibly miscalibrated in round 3+.
- Draw ingestion defaults to manual entry per tournament unless `ongoing_tourneys.csv` turns out to populate brackets early enough to use instead.

## Working conventions

- Make routine implementation calls yourself (library choices, schema details, feature windows) without checking in. Check in only where a decision would materially change the system's shape — the paid-feed trigger above is the clearest example.
- One-line update before starting each phase; a brief note when something changes the plan; skip routine progress narration otherwise. Lead with what's working and what isn't when reporting back.
- Delegate to subagents only for genuinely independent, substantial tracks (e.g. data pipeline and modeling running in parallel once the schema is settled). Don't delegate small or sequential steps, and don't use a subagent purely to double-check another's output.
- Keep READMEs and written docs proportional to what's needed to run and understand the system. No padding.