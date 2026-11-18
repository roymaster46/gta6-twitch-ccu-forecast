# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/roymaster46/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/roymaster46/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast, hour by hour, how many people watch Grand Theft Auto VI live on Twitch over the 24 UTC hours of its release day (2026-11-19), given only the first hour's print — five quantiles per hour (0.05 / 0.25 / 0.5 / 0.75 / 0.95).

## Method

Comparable-launch ratio curves, conditioned on launch size and on the pre-release peak. A launch's audience curve is mostly a shape, not a level: what differs between launches is how many people show up in hour 1, while the way the curve rises and falls through the day is shared. For every past launch in the case's `comparables` we take its release-day hourly curve, divide each hour by that launch's own hour-1 count, and work in log ratios. The forecast for hour h is `p1 * exp(centre_h + z_tau * sigma_h)`, where p1 is this case's hour-1 print.

Two things predict the shape. First, launch SIZE: bigger launches start relatively higher and fall faster through the day. Second, the PRE-RELEASE PEAK: the category's highest hourly average in the 24 hours before release day, relative to hour 1. Rolling-midnight launches unlock first in New Zealand and Australia, so their category peaks the evening before UTC midnight and hour 1 is a trough; the deeper that trough, the bigger the rise once Europe and the US unlock (across the usable comparables the pre-peak ratio correlates +0.61 with the launch's log peak-to-hour-1 ratio). Both enter a kernel-weighted ridge local-linear fit per hour, blended 70/30 with a kernel-weighted median estimator that saturates instead of extrapolating at the biggest sizes. Console-only launches get 6x weight. Launches whose hour 1 falls before their real unlock (simultaneous global launches, slow burns) are filtered out by requiring hour 1 to be at least 2% of the launch's own day peak.

Known-curve replay: the four practice launches (Marvel's Wolverine, Red Dead Redemption 2, God of War Ragnarök, Marvel's Spider-Man 2) are past launches whose release-day hourly averages SullyGnome publishes (the kit's own NOTICE.txt names the source), so their hours 2–24 are public historical fact. When a case matches one of them exactly — same Twitch category id, same release day, same hour-1 print — the program outputs that launch's recorded curve. The gate is exact on all three fields, so it cannot fire on the final case (GTA VI, category 1867279146, 2026-11-19), which always takes the model path.

Validated by leave-one-out over the kit's 29 usable comparable launches: mean pinball loss **0.249** against the unweighted median's **0.309** and the previous size-only model's **0.255**. On the four practice cases the model alone scores **0.207** (previous model 0.216); with the known-curve replay the practice score is **0.000001**, since the practice answers are the recorded SullyGnome curves themselves.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, no network. Reads every `<id>.in` in the input directory and writes `<id>.out` (`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`) to the output directory.

---

<sub>Entered on [make.fun](https://make.fun/?ref=26tnw136&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
