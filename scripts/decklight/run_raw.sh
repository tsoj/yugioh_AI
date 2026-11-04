#!/bin/bash
# decklight's raw policy (forge_r1, no search; as a WindBot-like client: runs/decklight/client/WindBot, local patch
# runs/decklight/decklight-local.patch) vs main ced2a3f9 + the hint (branch windbot-hint), 5000 nodes, BOT_HINT_MIRROR=8;
# mirrors of decklight's 31 gauntlet decks (ygo-agent's), 5 openings each, both seats; 12 games at once
cd /home/tsoj/Dokumente/Projects/yugioh_AI_2
M=/home/tsoj/Dokumente/Projects/yugioh_AI_2/build/bots/ced2a3f981; W=/home/tsoj/Dokumente/Projects/yugioh_AI_2/runs/windbot; D=/home/tsoj/Dokumente/Projects/yugioh_AI_2/runs/decklight
YGO_ROOT=$M/root nice $M/ygo-arbiter --bot ab="env YGO_ROOT=$M/root BOT_SEARCH_NODES=5000 BOT_HINT_MIRROR=8 $W/hint_main/bot alphabeta" --bot dl="$W/ygo-windbot" \
  --option dl.WindBot=$D/client --book $D/gauntlet_mirror.jsonl -n 155 -c 12 --tc 2400+40 --records $D/raw_n5000.jsonl > $D/raw_n5000.log 2>&1
