#!/bin/bash
# (variant: the bot knows the opponent's whole deck, BOT_HINT_MIRROR=all, runs/windbot/hint_all/bot = windbot-hint d95782d3; ports 9500+)
# decklight with its full search (edopro_host.py --opponent product) on our Rust engine (YGO_CORE=rust: its host and its
# search; third_party/ygopro-core-rust at 0478c44, runs/decklight/decklight-local.patch) vs our bot (main ced2a3f9 + the hint,
# branch windbot-hint: runs/windbot/hint_all/bot, 5000 nodes, BOT_HINT_MIRROR=all) as an EDOPro client (ygo-edopro-client,
# ygopro-core-rust branch edopro-client); the mirrors of runs/decklight/gauntlet_mirror.jsonl (31 decks x 5 openings), each
# opening in both seat orders with the opening's seed as decklight's deal seed. N duels at once (default 6).
#   runs/decklight/run_search.sh [N]
# Results: runs/decklight/search_n5000_fullhint.jsonl (one line per duel: won from our side, reason, turns, ...); logs in
# runs/decklight/search_logs_fullhint/ (our bot's stderr incl. its "searches/failures" line, the host's output).
cd /home/tsoj/Dokumente/Projects/yugioh_AI_2
N=${1:-6}
R=/home/tsoj/Dokumente/Projects/yugioh_AI_2/runs/decklight
export DL=/home/tsoj/Dokumente/Projects/yugioh_AI_2/external/decklight
export CLIENT=/home/tsoj/Dokumente/Projects/yugioh_AI_2/build/edopro-client-target/release/ygo-edopro-client
export M=/home/tsoj/Dokumente/Projects/yugioh_AI_2/build/bots/ced2a3f981
export BOT=/home/tsoj/Dokumente/Projects/yugioh_AI_2/runs/windbot/hint_all/bot
export R
duel() {
  local i=$1 deck=$2 seed=$3 order=$4
  local port=$((9500 + i)) ydk=$R/client/decks/$deck.ydk log=$R/search_logs_fullhint/$i-$deck-$seed-$order
  local second=""; [ "$order" = first ] && second="--bot-second"
  ( cd $DL && exec env YGO_CORE=rust OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/edopro_host.py --deck "$ydk" --port $port \
      --opponent product --once --seed $seed --replay-threads 1 $second ) > $log.host 2>&1 &
  local hostpid=$!
  YGO_ROOT=$M/root $CLIENT --host 127.0.0.1 --port $port --deck "$ydk" --$order --native-core --tag "$deck $seed $order" --out $R/search_n5000_fullhint.jsonl \
    --bot "env YGO_ROOT=$M/root BOT_SEARCH_NODES=5000 BOT_HINT_MIRROR=all BOT_GAME_STATS=1 $BOT alphabeta" 2> $log.bot
  wait $hostpid
}
export -f duel
python3 -c "
import json
for i, l in enumerate(open('$R/gauntlet_mirror.jsonl')):
    r = json.loads(l)
    for k, order in enumerate(('first', 'second')):
        print(2 * i + k, r['decks'][0]['name'], r['seed'] % 2**31, order)
" | xargs -P "$N" -L 1 bash -c 'duel "$@"' _
