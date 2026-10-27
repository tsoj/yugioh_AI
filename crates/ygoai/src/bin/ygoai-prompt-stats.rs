//! How many choices each kind of decision has in the real game (not as the search caps them):
//! replays recorded games (ygoai-datagen's OUT.games.jsonl or ygo-arbiter's records) and, per
//! prompt type, prints the count, mean, min, percentiles and max of the number of legal responses.
//!
//!   ygoai-prompt-stats GAMES.jsonl [JOBS] [MAX_GAMES]
//!
//! What counts as one choice: a whole response. A card selection of MIN..MAX of N cards has
//! sum C(N, k) choices; a tribute or a sum selection the sets of its cards that the rules accept
//! (from up to 20 cards); a zone choice C(free zones, count); a sort N!; a counter removal the ways
//! to take COUNT counters from the cards; a card-by-card selection (SELECT_UNSELECT_CARD) one
//! step: the cards to select or unselect, and finishing. A chain prompt with nothing to chain
//! (the hidden pass) is counted apart.

use std::collections::BTreeMap;
use std::io::BufRead;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ygo_arbiter::actions::{available_places, battle_options, check_sum, encode, idle_options};
use ygoai::agents::declarable_codes;
use ygoai::datarow::GameData;
use ygo_arbiter::game::{GameRecord, init, run_until_prompt};
use ygo_arbiter::messages::Msg;
use ygo_arbiter::observe::{is_hidden_prompt, to_core_response};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// (subsets of more cards are not enumerated)
const MAX_ENUM_CARDS: usize = 20;

fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// The subsets of `n` cards (non-empty) that `ok` accepts; None with more than MAX_ENUM_CARDS.
fn count_subsets(n: usize, ok: impl Fn(&[usize]) -> bool) -> Option<f64> {
    if n > MAX_ENUM_CARDS {
        return None;
    }
    let mut count = 0.0;
    let mut set = Vec::with_capacity(n);
    for mask in 1u32..(1 << n) {
        set.clear();
        set.extend((0..n).filter(|&i| mask & (1 << i) != 0));
        if ok(&set) {
            count += 1.0;
        }
    }
    Some(count)
}

/// (type, number of choices); None for a prompt whose choices are not counted
fn choices(m: &Msg) -> Option<(String, f64)> {
    let name = m.name().to_string();
    let n = match m {
        Msg::SelectIdlecmd { .. } => idle_options(m).len() as f64,
        Msg::SelectBattlecmd { .. } => battle_options(m).len() as f64,
        Msg::SelectChain { chains, forced, .. } => {
            if is_hidden_prompt(m) {
                return Some(("SELECT_CHAIN (nothing to chain)".into(), 1.0));
            }
            chains.len() as f64 + if *forced != 0 { 0.0 } else { 1.0 }
        }
        Msg::SelectEffectyn { .. } | Msg::SelectYesno { .. } => 2.0,
        Msg::SelectOption { options, .. } => options.len() as f64,
        Msg::SelectCard { min, max, cards, .. } => {
            let n: f64 = (*min..=*max).map(|k| binom(cards.len(), k as usize)).sum();
            let kind = if *max == 1 { "SELECT_CARD (1 card)" } else { "SELECT_CARD (several)" };
            return Some((kind.into(), n));
        }
        Msg::SelectUnselectCard { finishable, cancelable, select_cards, unselect_cards, .. } => {
            (select_cards.len() + unselect_cards.len()) as f64 + if *finishable != 0 || *cancelable != 0 { 1.0 } else { 0.0 }
        }
        Msg::SelectTribute { min, max, cards, .. } => {
            let rel = |i: usize| cards[i].release_param.unwrap_or(1).max(1);
            count_subsets(cards.len(), |s| {
                let total: u32 = s.iter().map(|&i| rel(i)).sum();
                s.len() <= (*max).max(1) as usize && total >= *min
            })?
        }
        Msg::SelectSum { cards, .. } => count_subsets(cards.len(), |s| check_sum(m, s))?,
        Msg::SelectPosition { positions, .. } => positions.count_ones() as f64,
        Msg::SelectPlace { player, count, flag } | Msg::SelectDisfield { player, count, flag } => {
            binom(available_places(*player, *flag).len(), *count as usize)
        }
        Msg::SortCard { cards, .. } | Msg::SortChain { cards, .. } => (1..=cards.len()).map(|i| i as f64).product(),
        Msg::SelectCounter { count, cards, .. } => {
            // (ways to take `count` counters, at most each card's own from it)
            let mut ways = vec![0.0f64; *count as usize + 1];
            ways[0] = 1.0;
            for c in cards {
                let have = c.counters.unwrap_or(0) as usize;
                let mut next = vec![0.0; ways.len()];
                for (t, w) in ways.iter().enumerate() {
                    for take in 0..=have.min(ways.len() - 1 - t) {
                        next[t + take] += w;
                    }
                }
                ways = next;
            }
            ways[*count as usize]
        }
        Msg::AnnounceRace { count, available, .. } => binom(available.count_ones() as usize, *count as usize),
        Msg::AnnounceAttrib { count, available, .. } => binom(available.count_ones() as usize, *count as usize),
        Msg::AnnounceCard { values, .. } => declarable_codes(values).len() as f64,
        Msg::AnnounceNumber { values, .. } => values.len() as f64,
        Msg::RockPaperScissors { .. } => 3.0,
        _ => return None,
    };
    Some((name, n))
}

fn replay(g: &GameData, out: &mut BTreeMap<String, Vec<f64>>, uncounted: &mut BTreeMap<String, usize>) -> Result<(), String> {
    let mut duel = g.record.setup(None).map_err(|e| e.0)?;
    duel.ocg_start_duel();
    let mut prompt: Option<Msg> = None;
    for (i, entry) in g.record.responses.iter().enumerate() {
        let step = run_until_prompt(&mut duel, prompt.as_ref()).map_err(|e| e.0)?;
        let p = step.prompt.ok_or_else(|| format!("game {}: the duel ended before response {i}", g.game))?;
        match choices(&p) {
            Some((kind, n)) => out.entry(kind).or_default().push(n),
            None => *uncounted.entry(p.name().to_string()).or_default() += 1,
        }
        let core = if g.viewer_responses { to_core_response(&p, entry.player, &entry.response).map_err(|e| format!("{e:?}"))? } else { entry.response.clone() };
        duel.ocg_set_response(&encode(&p, &core).map_err(|e| format!("{e:?}"))?);
        prompt = Some(p);
    }
    Ok(())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: ygoai-prompt-stats GAMES.jsonl [JOBS] [MAX_GAMES]");
        std::process::exit(2);
    }
    let jobs: usize = a.get(2).map_or(8, |s| s.parse().unwrap());
    let max_games: usize = a.get(3).map_or(usize::MAX, |s| s.parse().unwrap());
    let games: Vec<GameData> = std::io::BufReader::new(std::fs::File::open(&a[1]).expect("games"))
        .lines()
        .take(max_games)
        .enumerate()
        .map(|(i, l)| {
            let l = l.unwrap();
            serde_json::from_str::<GameData>(&l).unwrap_or_else(|_| {
                let record: GameRecord = serde_json::from_str(&l).expect("a GameData or GameRecord line");
                GameData { game: i, labels: vec![], record, viewer_responses: true }
            })
        })
        .collect();
    init(None).unwrap();
    let games = Arc::new(games);
    let all = Arc::new(Mutex::new((BTreeMap::<String, Vec<f64>>::new(), BTreeMap::<String, usize>::new())));
    let (next, failed) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let threads: Vec<_> = (0..jobs)
        .map(|_| {
            let (games, all, next, failed) = (games.clone(), all.clone(), next.clone(), failed.clone());
            std::thread::spawn(move || {
                let (mut out, mut unc) = (BTreeMap::new(), BTreeMap::new());
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(g) = games.get(i) else { break };
                    if !matches!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| replay(g, &mut out, &mut unc))), Ok(Ok(()))) {
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
                let mut all = all.lock().unwrap();
                for (k, v) in out {
                    all.0.entry(k).or_default().extend(v);
                }
                for (k, v) in unc {
                    *all.1.entry(k).or_default() += v;
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    let (mut all, unc) = Arc::try_unwrap(all).ok().unwrap().into_inner().unwrap();
    println!("{} games ({} failed to replay)", games.len(), failed.load(Ordering::Relaxed));
    println!("{:<34} {:>9} {:>9} {:>7} {:>6} {:>6} {:>7} {:>7} {:>8} {:>10}", "type", "prompts", "mean", "min", "p1", "p10", "median", "p90", "p99", "max");
    let mut rows: Vec<_> = all.iter_mut().collect();
    rows.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));
    for (k, v) in rows {
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        println!("{k:<34} {:>9} {mean:>9.1} {:>7} {:>6} {:>6} {:>7} {:>7} {:>8} {:>10}", v.len(), q(0.0), q(0.01), q(0.1), q(0.5), q(0.9), q(0.99), q(1.0));
    }
    if !unc.is_empty() {
        println!("not counted: {unc:?}");
    }
}
