//! How much the guess of the opponent's hidden cards changes the search: plays book openings with
//! the heuristic player and, at player 0's searched decisions, searches the position (fixed
//! nodes, fresh tables) once with the true cards and once per guess of the cards hidden from
//! player 0 (the likeliest deck as `naive_hidden_cards` guesses it, then draws of
//! `belief::guesses::resample`), with the search's trace (search::Trace). Then compares each
//! guess's tree with the truth's and the guesses' with each other, per iteration both completed:
//!
//! - shared: the edges (lines from the root, moves as `move_key` and response) in both trees, of
//!   those in either (Jaccard)
//! - same view: of the shared edges, those after which player 0 sees the same (the unknown cards
//!   did not show on that line)
//! - same value: of the shared edges, those valued within SAME_VALUE LP
//!
//! and per tree: the edges after which an unknown card was revealed, and those answering a prompt
//! that lists a hidden unknown card (the opponent could use it); at the root: whether the best
//! move is the truth's, and the value difference.
//!
//!   ygoai-tree-compare BOOK [OPENINGS] [NODES] [GUESSES] [EVERY]
//!
//! (EVERY: every that many searched decisions; $YGOAI_SKIP: openings to skip; $YGOAI_TREES: a
//! file for the per-position results as JSON lines)

use std::collections::{HashMap, HashSet};
use std::io::Write;

use ygoai::belief::guesses::{Guesses, resample};
use ygoai::belief::tracker::Guess;
use ygoai::belief_agents::{completion, naive_hidden_cards};
use ygoai::game::{DuelConfig, init, new_record, run_until_prompt};
use ygoai::heuristic::Heuristic;
use ygoai::matchrun::load_book_head;
use ygoai::observe::{auto_response, is_hidden_prompt};
use ygoai::pyrandom::PyRandom;
use ygoai::rduel::RDuel;
use ygoai::search::{AlphaBeta, SearchParams, Trace, TraceEdge, candidates, is_hidden, is_searched};
use ygo_core::duel::Duel;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Values this close (LP) count as the same
const SAME_VALUE: f64 = 50.0;

struct Tree {
    best: usize,
    value: f64,
    depth: i32,
    edges: Vec<TraceEdge>,
}

/// Sums of the comparison of two trees, over their iterations: [shared, either, same view, same
/// value]
#[derive(Default, Clone, Copy)]
struct Cmp {
    shared: u64,
    either: u64,
    same_view: u64,
    same_value: u64,
    /// by height (the last: that height or more): [shared, either]
    by_height: [[u64; 2]; MAX_HEIGHT],
}

/// Heights in the table by height
const MAX_HEIGHT: usize = 12;

impl Cmp {
    fn add(&mut self, o: Cmp) {
        self.shared += o.shared;
        self.either += o.either;
        self.same_view += o.same_view;
        self.same_value += o.same_value;
        for h in 0..MAX_HEIGHT {
            self.by_height[h][0] += o.by_height[h][0];
            self.by_height[h][1] += o.by_height[h][1];
        }
    }
    fn heights(&self) -> String {
        (0..MAX_HEIGHT).map(|h| format!("{:>5.1}", self.by_height[h][0] as f64 / self.by_height[h][1].max(1) as f64 * 100.0)).collect::<Vec<_>>().join(" ")
    }
    fn line(&self) -> String {
        let pct = |a: u64, b: u64| a as f64 / b.max(1) as f64 * 100.0;
        format!(
            "shared {:5.1}% of {:>8} edges; of the shared: same view {:5.1}%, same value {:5.1}%",
            pct(self.shared, self.either),
            self.either,
            pct(self.same_view, self.shared),
            pct(self.same_value, self.shared)
        )
    }
}

/// The edges of an iteration by path (an edge searched more than once: its last visit).
fn iteration_edges(t: &Tree, it: i32) -> HashMap<u64, &TraceEdge> {
    t.edges.iter().filter(|e| e.iteration == it).map(|e| (e.path, e)).collect()
}

fn compare(a: &Tree, b: &Tree) -> Cmp {
    let mut out = Cmp::default();
    for it in 1..=a.depth.min(b.depth) {
        let (ea, eb) = (iteration_edges(a, it), iteration_edges(b, it));
        let union: HashSet<u64> = ea.keys().chain(eb.keys()).copied().collect();
        out.either += union.len() as u64;
        let slot = |e: &TraceEdge| (e.height as usize).clamp(1, MAX_HEIGHT) - 1;
        for p in &union {
            let e = ea.get(p).or_else(|| eb.get(p)).unwrap();
            out.by_height[slot(e)][1] += 1;
        }
        for (p, x) in &ea {
            let Some(y) = eb.get(p) else { continue };
            out.shared += 1;
            out.by_height[slot(x)][0] += 1;
            out.same_view += (x.visible == y.visible) as u64;
            out.same_value += matches!((x.value, y.value), (Some(u), Some(v)) if (u - v).abs() <= SAME_VALUE) as u64;
        }
    }
    out
}

fn search(duel: &Duel, prompt: &ygoai::messages::Msg, cands: &[ygoai::actions::Response], cards: &[u32], nodes: u64) -> Tree {
    let mut ab = AlphaBeta::default();
    ab.trace = Some(Trace::new(0, cards.to_vec()));
    ab.prepare();
    let params = SearchParams { budget: None, elapsed: 0.0, clock_remaining: 1000.0, hint: None, seeds: [1, 2], max_nodes: Some(nodes) };
    let (best, depth) = ab.search(duel, prompt, cands, &params);
    Tree { best, value: ab.last_value, depth, edges: ab.trace.take().unwrap().edges }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(2).map_or(5, |s| s.parse().unwrap());
    let nodes: u64 = args.get(3).map_or(3000, |s| s.parse().unwrap());
    let n_guesses: usize = args.get(4).map_or(4, |s| s.parse().unwrap());
    let every: usize = args.get(5).map_or(3, |s| s.parse().unwrap());
    let skip: usize = std::env::var("YGOAI_SKIP").map_or(0, |s| s.parse().unwrap());
    let book = load_book_head(args[1].as_ref(), 0, n).expect("book");
    init(None).unwrap();
    let comp = completion("neural").unwrap();
    let mut out = std::env::var("YGOAI_TREES").ok().map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("trees file")));
    let mut rng = PyRandom::new(7);
    // totals: vs the truth (likeliest, drawn), drawn vs drawn
    let (mut vs_likeliest, mut vs_drawn, mut drawn_drawn) = (Cmp::default(), Cmp::default(), Cmp::default());
    // per tree kind (truth, likeliest, drawn): [edges, revealed, listed]; at the root: [searches, best = truth's, sum |value - truth's|]
    let mut per_tree = [[0u64; 3]; 3];
    let mut root = [[0.0f64; 3]; 3];
    let mut positions = 0;
    // (the truth searched again: the same tree, if the search is deterministic)
    let mut control = Cmp::default();
    for (oi, opening) in book.iter().enumerate().take(n).skip(skip) {
        let record = new_record(&opening.decks, opening.seed, &DuelConfig::default());
        let mut duel = record.setup(None).unwrap();
        duel.ocg_start_duel();
        let mut players = [Heuristic::new(1, 0), Heuristic::new(2, 1)];
        let (mut prompt, mut si) = (None, 0usize);
        for _ in 0..400 {
            let step = run_until_prompt(&mut duel, prompt.as_ref()).unwrap();
            if step.messages.iter().any(|m| matches!(m, ygoai::messages::Msg::Win { .. })) || step.ended {
                break;
            }
            for m in &step.messages {
                for p in &mut players {
                    p.observe_msg(m);
                }
            }
            let p = step.prompt.unwrap();
            let player = p.player().unwrap();
            let cands = candidates(&p);
            if player == 0 && is_searched(&p) && cands.len() > 1 {
                si += 1;
                if si % every == 0 {
                    positions += 1;
                    // the opponent's cards hidden from player 0, and its cards player 0 sees
                    let all = duel.ext_get_all_cards();
                    let cards: Vec<u32> = all.iter().filter(|e| e[0] == 1 && is_hidden(&duel, e[5])).map(|e| e[5]).collect();
                    let known: Vec<(u32, u32)> = all.iter().filter(|e| e[0] == 1 && e[5] != 0 && !is_hidden(&duel, e[5])).map(|e| (e[4], e[1])).collect();
                    let is_extra = |c: u32| ygoai::belief::completion::card(c).is_some_and(ygoai::belief::completion::is_extra_deck);
                    let guesses = Guesses {
                        viewer: 0,
                        cards: cards
                            .iter()
                            .map(|&cid| {
                                let extra = duel.card_by_id(cid).is_some_and(|c| is_extra(duel.card(c).data.code));
                                (cid, Guess { player: 1, extra, code: 0, rank: 0, need: None, excluded: vec![] })
                            })
                            .collect(),
                        known: (known.iter().filter(|k| !is_extra(k.0)).map(|k| k.0).collect(), known.iter().filter(|k| is_extra(k.0)).map(|k| k.0).collect()),
                    };
                    let truth = search(&duel, &p, &cands, &cards, nodes);
                    control.add(compare(&truth, &search(&duel, &p, &cands, &cards, nodes)));
                    let mut likeliest = RDuel::from_duel(duel.clone());
                    naive_hidden_cards(&mut likeliest, 0, comp, &mut rng);
                    let mut trees = vec![search(&likeliest.d, &p, &cands, &cards, nodes)];
                    for _ in 0..n_guesses {
                        let mut d = duel.clone();
                        resample(&mut d, &guesses, comp.unwrap(), &mut rng);
                        trees.push(search(&d, &p, &cands, &cards, nodes));
                    }
                    let mut pos = (Cmp::default(), Cmp::default(), Cmp::default());
                    for (k, t) in trees.iter().enumerate() {
                        let c = compare(&truth, t);
                        if k == 0 { pos.0.add(c) } else { pos.1.add(c) }
                        for u in trees.iter().skip(k + 1).filter(|_| k > 0) {
                            pos.2.add(compare(t, u));
                        }
                    }
                    for (kind, t) in std::iter::once((0, &truth)).chain(trees.iter().enumerate().map(|(k, t)| ((k > 0) as usize + 1, t))) {
                        let e = &mut per_tree[kind];
                        e[0] += t.edges.len() as u64;
                        e[1] += t.edges.iter().filter(|x| x.revealed > 0).count() as u64;
                        e[2] += t.edges.iter().filter(|x| x.listed).count() as u64;
                        let r = &mut root[kind];
                        r[0] += 1.0;
                        r[1] += (t.best == truth.best) as u64 as f64;
                        r[2] += if t.value.is_finite() && truth.value.is_finite() { (t.value - truth.value).abs().min(10000.0) } else { 0.0 };
                    }
                    vs_likeliest.add(pos.0);
                    vs_drawn.add(pos.1);
                    drawn_drawn.add(pos.2);
                    let frac = |c: Cmp| c.shared as f64 / c.either.max(1) as f64;
                    let depths: Vec<i32> = std::iter::once(truth.depth).chain(trees.iter().map(|t| t.depth)).collect();
                    let bests: Vec<usize> = std::iter::once(truth.best).chain(trees.iter().map(|t| t.best)).collect();
                    let values: Vec<f64> = std::iter::once(truth.value).chain(trees.iter().map(|t| t.value)).collect();
                    println!(
                        "opening {oi} search {si} {} ({} moves, {} hidden): shared with the truth {:.2} (likeliest) {:.2} (drawn), drawn-drawn {:.2}; depths {depths:?}; best {bests:?}; values {:?}",
                        p.name(),
                        cands.len(),
                        cards.len(),
                        frac(pos.0),
                        frac(pos.1),
                        frac(pos.2),
                        values.iter().map(|v| v.round() as i64).collect::<Vec<_>>()
                    );
                    if let Some(f) = &mut out {
                        let j = serde_json::json!({
                            "opening": oi, "search": si, "prompt": p.name(), "moves": cands.len(), "hidden": cards.len(),
                            "depths": depths, "best": bests, "values": values,
                            "vs_likeliest": [pos.0.shared, pos.0.either, pos.0.same_view, pos.0.same_value],
                            "vs_drawn": [pos.1.shared, pos.1.either, pos.1.same_view, pos.1.same_value],
                            "drawn_drawn": [pos.2.shared, pos.2.either, pos.2.same_view, pos.2.same_value],
                            "revealed": std::iter::once(&truth).chain(&trees).map(|t| t.edges.iter().filter(|x| x.revealed > 0).count()).collect::<Vec<_>>(),
                            "listed": std::iter::once(&truth).chain(&trees).map(|t| t.edges.iter().filter(|x| x.listed).count()).collect::<Vec<_>>(),
                            "edges": std::iter::once(&truth).chain(&trees).map(|t| t.edges.len()).collect::<Vec<_>>(),
                        });
                        writeln!(f, "{j}").unwrap();
                    }
                }
            }
            let resp = if is_hidden_prompt(&p) { auto_response() } else { players[player as usize].respond_to(&p, None).unwrap() };
            duel.ocg_set_response(&ygoai::actions::encode(&p, &resp).unwrap());
            prompt = Some(p);
        }
    }
    println!("\n{positions} positions, {nodes} nodes per search, {n_guesses} drawn guesses each");
    println!("trees vs the truth's, per iteration both completed:");
    println!("  likeliest deck: {}", vs_likeliest.line());
    println!("  drawn decks:    {}", vs_drawn.line());
    println!("drawn vs drawn:   {}", drawn_drawn.line());
    println!("(control, the truth again: {})", control.line());
    println!("\nshared edges by height (%): {}", (1..=MAX_HEIGHT).map(|h| format!("{h:>5}")).collect::<Vec<_>>().join(" "));
    println!("  likeliest vs truth:        {}", vs_likeliest.heights());
    println!("  drawn vs truth:            {}", vs_drawn.heights());
    println!("  drawn vs drawn:            {}", drawn_drawn.heights());
    println!("\nper tree         edges   revealed an unknown card   answered a prompt listing one   best = truth's   mean |value - truth's| (LP)");
    for (k, name) in ["truth", "likeliest", "drawn"].iter().enumerate() {
        let (e, r) = (per_tree[k], root[k]);
        let pct = |a: u64| a as f64 / e[0].max(1) as f64 * 100.0;
        println!(
            "  {name:<10} {:>10}   {:>22.1}%   {:>28.1}%   {:>13.1}%   {:>10.0}",
            e[0],
            pct(e[1]),
            pct(e[2]),
            r[1] / r[0].max(1.0) * 100.0,
            r[2] / r[0].max(1.0)
        );
    }
}
