//! Deck completion by the learned deck models (belief/neural_completion.py and the inference of
//! download_human_decks/complete_deck.py), on burn code generated from their ONNX export
//! (crates/deckmodels).
//!
//! An existence model predicts which cards a deck holds given the ones seen so far, a count model
//! how many copies of each. The replay's own constraints apply on top, as for the corpus
//! completion: no win-condition pieces, no card that adds a duel-wide effect no known card brings
//! (script_index::global_keys), only cards the card database knows.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Mutex, OnceLock};


use crate::belief::completion::{Completion, CorpusCompletion, VanillaCompletion, canonical_code, card, fits_need};
use crate::belief::script_index::{GlobalKey, global_keys, win_condition_codes};
use crate::belief::tracker::Need;

/// Rounds of the search for the number of distinct cards (complete_deck's max_iter)
const MAX_ITER: usize = 60;
/// The deck size the ranking for stand-ins assumes
const RANK_SLOTS: f32 = 55.0;

#[derive(serde::Deserialize)]
struct Meta {
    codes: Vec<u32>,
    is_extra: Vec<bool>,
    avg_copies: f64,
}

/// The models and their vocabulary.
pub struct DeckModels {
    models: deckmodels::Models,
    codes: Vec<u32>,
    index: HashMap<u32, usize>,
    is_extra: Vec<bool>,
    avg_copies: f64,
}

impl DeckModels {
    /// The models and their vocabulary, as built into crates/deckmodels.
    pub fn load() -> Result<DeckModels, String> {
        let models = deckmodels::Models::load()?;
        let meta: Meta = serde_json::from_str(deckmodels::meta_json().ok_or("no deck models")?).map_err(|e| e.to_string())?;
        let index = meta.codes.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Ok(DeckModels { models, codes: meta.codes, index, is_extra: meta.is_extra, avg_copies: meta.avg_copies })
    }

    pub fn vocab_len(&self) -> usize {
        self.codes.len()
    }

    /// P(presence) per vocabulary index, given the visible counts per index.
    pub fn presence(&self, c_vis: &[i64], slots: f32) -> Vec<f32> {
        let cards: Vec<(usize, i64)> = (0..c_vis.len()).filter(|&j| c_vis[j] > 0).map(|j| (j, c_vis[j])).collect();
        self.models.existence_logits(&cards, slots).into_iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
    }

    /// (expected count in [1, 3], argmax count) per card of the set (vocabulary indices).
    pub fn counts(&self, set: &[usize]) -> (Vec<f64>, Vec<i64>) {
        if set.is_empty() {
            return (vec![], vec![]);
        }
        let l = set.len();
        let v = self.models.count_logits(set);
        let (mut raw, mut discrete) = (Vec::with_capacity(l), Vec::with_capacity(l));
        for i in 0..l {
            let x = &v[3 * i..3 * i + 3];
            let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = x.iter().map(|&y| (y - m).exp()).collect();
            let s: f32 = e.iter().sum();
            raw.push(e.iter().enumerate().map(|(k, &y)| (y / s) as f64 * (k + 1) as f64).sum());
            // (the first of equal maxima, as torch's argmax)
            let best = (0..3).fold(0, |b, k| if x[k] > x[b] { k } else { b });
            discrete.push(best as i64 + 1);
        }
        (raw, discrete)
    }

    /// complete_deck: the full deck (main, extra codes) given the observed codes and the sizes.
    pub fn complete_deck(&self, observed: &[u32], main_size: usize, extra_size: usize) -> Result<(Vec<u32>, Vec<u32>), String> {
        let v = self.codes.len();
        let mut c_vis = vec![0i64; v];
        for c in observed {
            if let Some(&j) = self.index.get(c)
                && c_vis[j] < 3
            {
                c_vis[j] += 1;
            }
        }
        let main_obs: i64 = (0..v).filter(|&j| !self.is_extra[j]).map(|j| c_vis[j]).sum();
        let extra_obs: i64 = (0..v).filter(|&j| self.is_extra[j]).map(|j| c_vis[j]).sum();
        if main_obs as usize > main_size || extra_obs as usize > extra_size {
            return Err(format!("observed {main_obs}+{extra_obs} cards for a deck of {main_size}+{extra_size}"));
        }
        let target = (main_size + extra_size) as i64;
        let presence = self.presence(&c_vis, target as f32);
        // the non-visible cards by P(presence), the likeliest first (ties by index)
        let mut sorted_nv: Vec<usize> = (0..v).filter(|&j| c_vis[j] == 0).collect();
        sorted_nv.sort_by(|&a, &b| presence[b].partial_cmp(&presence[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        let visible: Vec<usize> = (0..v).filter(|&j| c_vis[j] > 0).collect();
        let k_max = visible.len() + sorted_nv.len();
        let k_init = ((target as f64 / self.avg_copies.max(1e-6)).round_ties_even() as usize).max(1).max(visible.len()).min(k_max);
        // the search for the number of distinct cards K
        type Entry = (usize, Vec<usize>, Vec<f64>, Vec<i64>, i64);
        let mut history: Vec<Entry> = Vec::new();
        let (mut k, mut prev_sign, mut osc) = (k_init as i64, 0i64, 0);
        for _ in 0..MAX_ITER {
            if k < 1 || k as usize > k_max || history.iter().any(|h| h.0 == k as usize) {
                break;
            }
            let n_use = (k as usize).saturating_sub(visible.len()).min(sorted_nv.len());
            let cand: Vec<usize> = visible.iter().chain(&sorted_nv[..n_use]).copied().collect();
            let (raw, discrete) = self.counts(&cand);
            let adj: Vec<i64> = cand.iter().zip(&discrete).map(|(&j, &c)| c.max(c_vis[j])).collect();
            let total: i64 = adj.iter().sum();
            history.push((k as usize, cand, raw, adj, total));
            if total == target {
                break;
            }
            let sign = if total > target { 1 } else { -1 };
            if prev_sign != 0 && sign != prev_sign {
                osc += 1;
                if osc >= 2 {
                    break;
                }
            }
            prev_sign = sign;
            k += if total > target { -1 } else { 1 };
        }
        // the best K: an exact match (the smallest K), else the cheapest forced onto the total
        let exact = history.iter().filter(|h| h.4 == target).min_by_key(|h| h.0);
        let (final_cand, final_counts, raw_of): (Vec<usize>, Vec<i64>, Vec<f64>) = match exact {
            Some(h) => (h.1.clone(), h.3.clone(), h.2.clone()),
            None => {
                // (cost, candidates, counts, expected counts)
                type Forced = (f64, Vec<usize>, Vec<i64>, Vec<f64>);
                let mut best: Option<Forced> = None;
                for h in &history {
                    let (adj, cost) = force_total(&h.1, &h.2, &h.3, &c_vis, target);
                    if best.as_ref().is_none_or(|b| cost < b.0) {
                        best = Some((cost, h.1.clone(), adj, h.2.clone()));
                    }
                }
                let b = best.ok_or("the search produced no candidates")?;
                (b.1, b.2, b.3)
            }
        };
        // per section: the cards in P(presence) order (visible first), truncated to the section
        let per_card: HashMap<usize, i64> = final_cand.iter().copied().zip(final_counts.iter().copied()).collect();
        let raw_by: HashMap<usize, f64> = final_cand.iter().copied().zip(raw_of.iter().copied()).collect();
        let visible_set: BTreeSet<usize> = visible.iter().copied().collect();
        // (per section: its cards with counts, and their total)
        let mut sec: [(Vec<(usize, i64)>, i64); 2] = [(vec![], 0), (vec![], 0)];
        let sizes = [main_size as i64, extra_size as i64];
        let place = |j: usize, sec: &mut [(Vec<(usize, i64)>, i64); 2]| {
            let s = self.is_extra[j] as usize;
            let c = per_card[&j].min(sizes[s] - sec[s].1).max(0);
            if c > 0 {
                sec[s].0.push((j, c));
                sec[s].1 += c;
            }
        };
        for &j in &visible {
            place(j, &mut sec);
        }
        for &j in final_cand.iter().filter(|j| !visible_set.contains(j)) {
            if sec[0].1 >= sizes[0] && sec[1].1 >= sizes[1] {
                break;
            }
            place(j, &mut sec);
        }
        let [(mut main, main_total), (mut extra, extra_total)] = sec;
        let balance = |counts: &mut Vec<(usize, i64)>, mut current: i64, target: i64, is_extra: bool| {
            if current >= target {
                return;
            }
            // first more copies of the section's cards (the highest expected count first), then new
            // cards by P(presence)
            let mut order: Vec<usize> = (0..counts.len()).collect();
            order.sort_by(|&a, &b| {
                let (ra, rb) = (raw_by.get(&counts[a].0).copied().unwrap_or(0.0), raw_by.get(&counts[b].0).copied().unwrap_or(0.0));
                rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
            });
            for i in order {
                while counts[i].1 < 3 && current < target {
                    counts[i].1 += 1;
                    current += 1;
                }
                if current >= target {
                    return;
                }
            }
            for &j in &sorted_nv {
                if current >= target {
                    break;
                }
                if counts.iter().any(|c| c.0 == j) || self.is_extra[j] != is_extra {
                    continue;
                }
                let c = (target - current).min(3);
                counts.push((j, c));
                current += c;
            }
        };
        balance(&mut main, main_total, main_size as i64, false);
        balance(&mut extra, extra_total, extra_size as i64, true);
        let expand = |s: &[(usize, i64)]| s.iter().flat_map(|&(j, n)| std::iter::repeat_n(self.codes[j], n as usize)).collect::<Vec<u32>>();
        Ok((expand(&main), expand(&extra)))
    }

    /// Every card of the vocabulary, the likeliest to be in the deck first (the known ones left out).
    pub fn ranked(&self, canon: &[u32]) -> Vec<u32> {
        let mut c_vis = vec![0i64; self.codes.len()];
        for c in canon {
            if let Some(&j) = self.index.get(c)
                && c_vis[j] < 3
            {
                c_vis[j] += 1;
            }
        }
        let p = self.presence(&c_vis, RANK_SLOTS);
        let mut order: Vec<usize> = (0..p.len()).collect();
        order.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap_or(std::cmp::Ordering::Equal));
        order.into_iter().filter(|&j| c_vis[j] == 0).map(|j| self.codes[j]).collect()
    }
}

/// Greedy +/-1 adjustments so that the counts sum to `target` (complete_deck.force_total): the
/// counts and the squared deviation from the expected counts.
fn force_total(cand: &[usize], raw: &[f64], discrete: &[i64], c_vis: &[i64], target: i64) -> (Vec<i64>, f64) {
    let mut cur = discrete.to_vec();
    let mut total: i64 = cur.iter().sum();
    let vmin: Vec<i64> = cand.iter().map(|&j| c_vis[j].max(1)).collect();
    while total != target {
        let sign = if total > target { -1 } else { 1 };
        let (mut best_i, mut best_d) = (None, f64::INFINITY);
        for i in 0..cand.len() {
            if (sign > 0 && cur[i] >= 3) || (sign < 0 && cur[i] <= vmin[i]) {
                continue;
            }
            let d = (raw[i] - (cur[i] + sign) as f64).powi(2) - (raw[i] - cur[i] as f64).powi(2);
            if d < best_d {
                (best_i, best_d) = (Some(i), d);
            }
        }
        let Some(i) = best_i else { break };
        cur[i] += sign;
        total += sign;
    }
    let cost = (0..cand.len()).map(|i| (raw[i] - cur[i] as f64).powi(2)).sum();
    (cur, cost)
}

type CompleteKey = (Vec<u32>, Vec<u32>, usize, usize, Vec<u32>);
type StandInKey = (Need, Vec<u32>);
type Fills = (Vec<u32>, Vec<u32>);

/// The neural completion (neural_completion.NeuralCompletion).
pub struct NeuralCompletion {
    models: DeckModels,
    cache: Mutex<HashMap<CompleteKey, Fills>>,
    stand_in_cache: Mutex<HashMap<StandInKey, Vec<u32>>>,
}

impl NeuralCompletion {
    /// The process's completion (the models load once).
    pub fn get() -> Result<&'static NeuralCompletion, String> {
        static N: OnceLock<Result<NeuralCompletion, String>> = OnceLock::new();
        N.get_or_init(|| {
            Ok(NeuralCompletion { models: DeckModels::load()?, cache: Mutex::new(HashMap::new()), stand_in_cache: Mutex::new(HashMap::new()) })
        })
        .as_ref()
        .map_err(|e| e.clone())
    }

    pub fn models(&self) -> &DeckModels {
        &self.models
    }

    /// The model's cards of one section beyond the known ones, those usable (the corpus completion's
    /// for the rest): `n` of them.
    fn fill_section(&self, model_cards: &[u32], known: &[u32], n: usize, usable: &dyn Fn(u32) -> bool, spare: &dyn Fn() -> Result<Vec<u32>, String>) -> Result<Vec<u32>, String> {
        let mut rest: HashMap<u32, i64> = HashMap::new();
        for &c in model_cards {
            *rest.entry(c).or_default() += 1;
        }
        for c in known {
            *rest.entry(*c).or_default() -= 1;
        }
        let mut out = Vec::new();
        for &c in model_cards {
            // (as `rest[c] > 0 and not rest.subtract([c]) and usable(c)`: taken from the rest even
            // if not usable)
            let r = rest.entry(c).or_default();
            if *r > 0 {
                *r -= 1;
                if usable(c) {
                    out.push(c);
                }
            }
        }
        if out.len() < n {
            // (in the model's order; what it could not supply, the corpus completion does)
            let mut counts: HashMap<u32, i64> = HashMap::new();
            for &c in known.iter().chain(&out) {
                *counts.entry(c).or_default() += 1;
            }
            for c in spare()? {
                if out.len() >= n {
                    break;
                }
                let k = counts.entry(canonical_code(c)).or_default();
                if *k < 3 {
                    out.push(c);
                    *k += 1;
                }
            }
        }
        out.truncate(n);
        Ok(out)
    }
}

impl Completion for NeuralCompletion {
    fn complete(&self, known_main: &[u32], known_extra: &[u32], n_main: usize, n_extra: usize, present: &[u32]) -> Result<(Vec<u32>, Vec<u32>), String> {
        let p: Vec<u32> = present.iter().copied().collect::<BTreeSet<u32>>().into_iter().collect();
        let mut km_sorted = known_main.to_vec();
        km_sorted.sort();
        let mut ke_sorted = known_extra.to_vec();
        ke_sorted.sort();
        let key = (km_sorted, ke_sorted, n_main, n_extra, p);
        if let Some(r) = self.cache.lock().unwrap().get(&key) {
            return Ok(r.clone());
        }
        let km: Vec<u32> = known_main.iter().map(|&c| canonical_code(c)).collect();
        let ke: Vec<u32> = known_extra.iter().map(|&c| canonical_code(c)).collect();
        let observed: Vec<u32> = km.iter().chain(&ke).copied().collect();
        let (main, extra) = self.models.complete_deck(&observed, km.len() + n_main, ke.len() + n_extra)?;
        let wins = win_condition_codes();
        let mut have: BTreeSet<GlobalKey> = BTreeSet::new();
        for &c in present.iter().chain(&km).chain(&ke) {
            have.extend(global_keys(canonical_code(c)));
        }
        let usable = |code: u32| card(code).is_some() && !wins.contains(&code) && global_keys(code).is_subset(&have);
        let corpus = || CorpusCompletion::get();
        let fill_main = self.fill_section(&main, &km, n_main, &usable, &|| Ok(corpus()?.complete(known_main, known_extra, n_main, 0, present)?.0))?;
        let fill_extra = self.fill_section(&extra, &ke, n_extra, &usable, &|| Ok(corpus()?.complete(known_main, known_extra, 0, n_extra, present)?.1))?;
        let r = (fill_main, fill_extra);
        self.cache.lock().unwrap().insert(key, r.clone());
        Ok(r)
    }

    fn stand_ins(&self, need: Need, known: &[u32]) -> Vec<u32> {
        let canon: Vec<u32> = known.iter().map(|&c| canonical_code(c)).collect();
        let mut key_known = canon.clone();
        key_known.sort();
        let key = (need, key_known);
        if let Some(r) = self.stand_in_cache.lock().unwrap().get(&key) {
            return r.clone();
        }
        let wins = win_condition_codes();
        let mut have: BTreeSet<GlobalKey> = BTreeSet::new();
        for &c in &canon {
            have.extend(global_keys(c));
        }
        let mut out = Vec::new();
        for code in self.models.ranked(&canon) {
            if let Some(c) = card(code)
                && !wins.contains(&code)
                && global_keys(code).is_subset(&have)
                && fits_need(Some(c), need, true)
            {
                out.push(code);
                if out.len() >= 12 {
                    break;
                }
            }
        }
        for c in VanillaCompletion::get().stand_ins(need, &[]) {
            if !out.contains(&c) {
                out.push(c);
            }
        }
        self.stand_in_cache.lock().unwrap().insert(key, out.clone());
        out
    }
}
