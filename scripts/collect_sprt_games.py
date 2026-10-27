"""SPRT match records with search labels from runs started at or after CUTOFF into one games file,
plus a sidecar (one line per game): run, deck pair key, seat order (1 if seat 0 holds the pair's
first deck by name), winner deck index. The first step of the eval's data from SPRT games (then
bot-refeature GAMES ROWS --features mover2, then scripts/sprt_games_to_npz.py).

  scripts/collect_sprt_games.py "2026-10-17 23:22" GAMES.jsonl META.jsonl   (run in the repo root)"""
import json, glob, os, time, sys
cutoff = time.mktime(time.strptime(sys.argv[1], '%Y-%m-%d %H:%M'))
out, meta = sys.argv[2], sys.argv[3]
n = 0
with open(out, 'w') as fo, open(meta, 'w') as fm:
    for f in sorted(glob.glob('runs/sprt/*.jsonl')):
        base = os.path.basename(f)
        if base == 'results.jsonl' or base.startswith('vsold'): continue
        try: start = time.mktime(time.strptime(base.rsplit('_', 1)[1][:15], '%Y%m%d-%H%M%S'))
        except Exception: continue
        if start < cutoff: continue
        for line in open(f):
            try: d = json.loads(line)
            except Exception: continue
            if 'decks' not in d or not d.get('labels'): continue
            names = [x['name'] for x in d['decks']]
            key = '\t'.join(sorted(names))
            order = int(names[0] == sorted(names)[0])
            w = d['result'].get('winner')
            fo.write(line if line.endswith('\n') else line + '\n')
            fm.write(json.dumps({'run': base, 'pair': key, 'order': order, 'winner': w}) + '\n')
            n += 1
print(n, 'games')
