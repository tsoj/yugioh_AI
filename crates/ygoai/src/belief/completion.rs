//! Guessing the unknown part of the opponent's deck list (belief/completion.py, corpus.py).
//!
//! [`Completion`] is the interface; [`VanillaCompletion`] fills with effect-less cards,
//! [`CorpusCompletion`] with the cards of the corpus decks that best match the known cards (the
//! numpy index build/deck_index.npz that belief/corpus.py builds).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ygo_core::constants::*;
use ygo_core::types::CardData;

use crate::belief::script_index::{GlobalKey, global_keys, win_condition_codes};
use crate::belief::tracker::Need;
use crate::game::card_db;
use crate::paths::{build_dir, card_scripts_dir};
use crate::pyrandom::PyRandom;

/// Levels a monster Set (or Normal Summoned) with that many Tributes can have at all: effects
/// can lower the Tributes a monster needs, so only a lower bound.
pub fn tribute_levels_any(tributes: u32) -> std::ops::Range<u32> {
    match tributes {
        0 => 0..13,
        1 => 5..13,
        _ => 7..13,
    }
}

pub fn tribute_levels(tributes: u32) -> std::ops::Range<u32> {
    match tributes {
        0 => 0..5,
        1 => 5..7,
        _ => 7..13,
    }
}

/// Monsters that can be Set in the spell/trap zone (EFFECT_MONSTER_SSET, e.g. the "Artifact"
/// monsters), read from the card scripts.
pub fn monster_sset_codes() -> &'static HashSet<u32> {
    static S: OnceLock<HashSet<u32>> = OnceLock::new();
    S.get_or_init(|| {
        let mut out = HashSet::new();
        for sub in ["official", "pre-release", "unofficial"] {
            let Ok(rd) = std::fs::read_dir(card_scripts_dir().join(sub)) else { continue };
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                let Some(code) = name.strip_prefix('c').and_then(|n| n.strip_suffix(".lua")).and_then(|n| n.parse::<u32>().ok()) else {
                    continue;
                };
                if let Ok(b) = std::fs::read(e.path())
                    && String::from_utf8_lossy(&b).contains("EFFECT_MONSTER_SSET")
                {
                    out.insert(code);
                }
            }
        }
        out
    })
}

pub fn is_extra_deck(c: &CardData) -> bool {
    c.type_ & TYPE_MONSTER != 0 && c.type_ & (TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK) != 0
}

/// Whether a card can be the hidden card with requirement `need`: Set in a spell/trap zone, Set
/// in the field zone, Set in a monster zone (with that many Tributes), summoned face-down.
pub fn fits_need(card: Option<&CardData>, need: Need, usual: bool) -> bool {
    let Some(card) = card else { return false };
    match need {
        Need::St => {
            if card.type_ & TYPE_MONSTER != 0 {
                return monster_sset_codes().contains(&card.code);
            }
            card.type_ & (TYPE_SPELL | TYPE_TRAP) != 0 && card.type_ & TYPE_FIELD == 0
        }
        Need::Field => card.type_ & TYPE_SPELL != 0 && card.type_ & TYPE_FIELD != 0,
        Need::Mset(t) => {
            if card.type_ & TYPE_MONSTER == 0 || is_extra_deck(card) {
                return false;
            }
            (if usual { tribute_levels(t) } else { tribute_levels_any(t) }).contains(&card.level)
        }
        Need::Monster => card.type_ & TYPE_MONSTER != 0 && !is_extra_deck(card),
    }
}

/// The card data of a code, if the database knows it.
pub fn card(code: u32) -> Option<&'static CardData> {
    card_db().cards.get(&code)
}

fn is_alt_art(c: &CardData) -> bool {
    c.alias != 0 && (c.alias as i64 - c.code as i64).abs() < 10
}

/// Alternate artworks behave exactly like the original card: the original's code (decks.canonical_code).
pub fn canonical_code(code: u32) -> u32 {
    match card(code) {
        Some(c) if is_alt_art(c) && card_db().cards.contains_key(&c.alias) => c.alias,
        _ => code,
    }
}

/// Given the known cards of a deck, guess the missing ones.
pub trait Completion: Send + Sync {
    /// (main, extra) fills; `present`: the other known cards of the game (both players'). An error
    /// where the Python completion raises RuntimeError (e.g. a filler pool exhausted).
    fn complete(&self, known_main: &[u32], known_extra: &[u32], n_main: usize, n_extra: usize, present: &[u32]) -> Result<(Vec<u32>, Vec<u32>), String>;

    /// Codes for a hidden card of a known kind that was never revealed, the likeliest first
    /// (`known`: the other known cards of its deck); by default effect-less ones.
    fn stand_ins(&self, need: Need, _known: &[u32]) -> Vec<u32> {
        VanillaCompletion::get().stand_ins(need, &[])
    }
}

fn fill(pool: &[u32], known: &[u32], n: usize) -> Result<Vec<u32>, String> {
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &c in known {
        let k = card(c).filter(|d| d.alias != 0).map_or(c, |d| d.alias);
        *counts.entry(k).or_default() += 1;
    }
    let mut out = Vec::new();
    while out.len() < n {
        let mut progressed = false;
        for &code in pool {
            if out.len() >= n {
                break;
            }
            let k = counts.entry(code).or_default();
            if *k < 3 {
                *k += 1;
                out.push(code);
                progressed = true;
            }
        }
        if !progressed {
            return Err("filler pool exhausted".into());
        }
    }
    Ok(out)
}

/// Fills unknown cards with effect-less cards (never interferes with the game).
pub struct VanillaCompletion {
    pub main_pool: Vec<u32>,
    pub extra_pool: Vec<u32>,
    pub settable_monsters_all: Vec<u32>,
    pub settable_spells: Vec<u32>,
}

impl VanillaCompletion {
    pub fn get() -> &'static VanillaCompletion {
        static V: OnceLock<VanillaCompletion> = OnceLock::new();
        V.get_or_init(VanillaCompletion::new)
    }

    fn new() -> VanillaCompletion {
        let db = card_db();
        let wins = win_condition_codes(); // (never: e.g. Exodia pieces, a guessed hand would win)
        let ot = |c: u32| db.ots.get(&c).copied().unwrap_or(0);
        let sorted = |f: &dyn Fn(&CardData) -> bool| -> Vec<u32> {
            let mut v: Vec<u32> = db.cards.values().filter(|c| f(c)).map(|c| c.code).collect();
            v.sort();
            v
        };
        let vanillas = sorted(&|c| c.type_ == TYPE_MONSTER | TYPE_NORMAL && c.alias == 0 && ot(c.code) & 0x3 != 0 && !wins.contains(&c.code));
        let main_pool: Vec<u32> = vanillas.iter().copied().take(60).collect();
        let extra_pool: Vec<u32> = sorted(&|c| c.type_ == TYPE_MONSTER | TYPE_FUSION && c.alias == 0 && ot(c.code) & 0x3 != 0).into_iter().take(30).collect();
        let level = |c: u32| db.cards[&c].level;
        let mut settable_monsters_all = Vec::new();
        for levels in [1..5, 5..7, 7..13] {
            settable_monsters_all.extend(vanillas.iter().copied().filter(|&c| levels.contains(&level(c))).take(6));
        }
        // (then one of each race and Level band: a hidden monster's race can matter)
        for levels in [1..5, 5..7, 7..13] {
            let mut seen_races = HashSet::new();
            for &c in &vanillas {
                let d = &db.cards[&c];
                if levels.contains(&d.level) && !seen_races.contains(&d.race) && !settable_monsters_all.contains(&c) {
                    seen_races.insert(d.race);
                    settable_monsters_all.push(c);
                }
            }
        }
        let normal_st = sorted(&|c| (c.type_ == TYPE_SPELL || c.type_ == TYPE_TRAP) && c.alias == 0 && ot(c.code) & 0x3 != 0);
        // the lowest codes, plus some spread over the whole range
        let step = ((normal_st.len() as i64 - 10) / 30).max(1) as usize;
        let mut settable_spells: Vec<u32> = normal_st.iter().copied().take(10).collect();
        settable_spells.extend(normal_st.iter().copied().skip(10).step_by(step));
        settable_spells.extend(sorted(&|c| c.type_ == TYPE_SPELL | TYPE_FIELD && c.alias == 0 && ot(c.code) & 0x3 != 0).into_iter().take(5));
        VanillaCompletion { main_pool, extra_pool, settable_monsters_all, settable_spells }
    }
}

impl Completion for VanillaCompletion {
    fn complete(&self, known_main: &[u32], known_extra: &[u32], n_main: usize, n_extra: usize, _present: &[u32]) -> Result<(Vec<u32>, Vec<u32>), String> {
        Ok((fill(&self.main_pool, known_main, n_main)?, fill(&self.extra_pool, known_extra, n_extra)?))
    }

    fn stand_ins(&self, need: Need, _known: &[u32]) -> Vec<u32> {
        let pool = match need {
            Need::St | Need::Field => &self.settable_spells,
            Need::Mset(_) => &self.settable_monsters_all,
            Need::Monster => &self.main_pool,
        };
        pool.iter().copied().filter(|&c| fits_need(card(c), need, true)).collect()
    }
}

/// One array of an .npz file (numpy's format: a header, then the data in C order).
fn npy_array(zip: &mut zip::ZipArchive<std::fs::File>, name: &str) -> Result<(String, Vec<u8>), String> {
    let mut f = zip.by_name(name).map_err(|e| format!("{name}: {e}"))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    if buf.len() < 10 || &buf[..6] != b"\x93NUMPY" {
        return Err(format!("{name}: not an npy array"));
    }
    let (hlen, hstart) = if buf[6] == 1 { (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10) } else { (u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize, 12) };
    let header = String::from_utf8_lossy(&buf[hstart..hstart + hlen]).to_string();
    Ok((header, buf[hstart + hlen..].to_vec()))
}

/// Opponent deck completion from a corpus of human deck lists (belief/corpus.py).
pub struct CorpusCompletion {
    deck: Vec<i32>,
    code: Vec<i64>,
    count: Vec<i8>,
    extra: Vec<bool>,
    n: usize,
    /// rows by code (each code's rows in index order)
    rows: HashMap<i64, Vec<usize>>,
    top_k: usize,
}

pub fn index_path() -> PathBuf {
    build_dir().join("deck_index.npz")
}

impl CorpusCompletion {
    /// The completion from the index (build it with belief/corpus.py: build_index).
    pub fn load(path: &Path) -> Result<CorpusCompletion, String> {
        let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut z = zip::ZipArchive::new(f).map_err(|e| e.to_string())?;
        let (_, deck) = npy_array(&mut z, "deck.npy")?;
        let (_, code) = npy_array(&mut z, "code.npy")?;
        let (_, count) = npy_array(&mut z, "count.npy")?;
        let (_, extra) = npy_array(&mut z, "extra.npy")?;
        let (_, n) = npy_array(&mut z, "n.npy")?;
        let deck: Vec<i32> = deck.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
        let code: Vec<i64> = code.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect();
        let count: Vec<i8> = count.iter().map(|&b| b as i8).collect();
        let extra: Vec<bool> = extra.iter().map(|&b| b != 0).collect();
        let n = i64::from_le_bytes(n[..8].try_into().unwrap()) as usize;
        let mut rows: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, &c) in code.iter().enumerate() {
            rows.entry(c).or_default().push(i);
        }
        Ok(CorpusCompletion { deck, code, count, extra, n, rows, top_k: 20 })
    }

    /// The default index, loaded once per process.
    pub fn get() -> Result<&'static CorpusCompletion, String> {
        static C: OnceLock<Result<CorpusCompletion, String>> = OnceLock::new();
        C.get_or_init(|| CorpusCompletion::load(&index_path())).as_ref().map_err(|e| e.clone())
    }

    fn scores(&self, known: &BTreeMap<u32, u32>) -> Vec<f32> {
        let mut scores = vec![0f32; self.n];
        for (&code, &k) in known {
            for &r in self.rows.get(&(code as i64)).map_or(&[][..], |v| &v[..]) {
                scores[self.deck[r] as usize] += (self.count[r] as i64).min(k as i64) as f32;
            }
        }
        scores
    }

    fn fill_kind(&self, weights: &BTreeMap<(u32, bool), f64>, known: &BTreeMap<u32, u32>, n: usize, extra: bool, skip: &dyn Fn(u32) -> bool) -> Vec<u32> {
        let wins = win_condition_codes(); // (never guessed: e.g. Exodia pieces)
        let mut have: HashMap<u32, u32> = known.iter().map(|(&c, &k)| (c, k)).collect();
        let mut ranked: Vec<(f64, u32)> =
            weights.iter().filter(|&(&(code, is_extra), _)| is_extra == extra && !wins.contains(&code) && !skip(code)).map(|(&(code, _), &w)| (w, code)).collect();
        // (sorted(..., reverse=True): by weight, then code, descending)
        ranked.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = Vec::new();
        while out.len() < n {
            let mut progressed = false;
            for &(_, code) in &ranked {
                if out.len() >= n {
                    break;
                }
                let h = have.entry(code).or_default();
                if *h < 3 {
                    *h += 1;
                    out.push(code);
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }
        if out.len() < n {
            // the best decks did not have enough cards of this kind; fall back to vanillas
            let van = VanillaCompletion::get();
            for &code in if extra { &van.extra_pool } else { &van.main_pool } {
                while out.len() < n && have.get(&code).copied().unwrap_or(0) < 3 {
                    *have.entry(code).or_default() += 1;
                    out.push(code);
                }
            }
        }
        out
    }
}

impl Completion for CorpusCompletion {
    fn complete(&self, known_main: &[u32], known_extra: &[u32], n_main: usize, n_extra: usize, present: &[u32]) -> Result<(Vec<u32>, Vec<u32>), String> {
        let mut known: BTreeMap<u32, u32> = BTreeMap::new();
        for &c in known_main.iter().chain(known_extra) {
            *known.entry(canonical_code(c)).or_default() += 1;
        }
        let scores = self.scores(&known);
        let best: Vec<usize> = if scores.iter().copied().fold(f32::NEG_INFINITY, f32::max) <= 0.0 {
            // (drawn from the known cards: the same in every call for the same cards)
            let r = format!("[{}]", known.iter().map(|(c, k)| format!("({c}, {k})")).collect::<Vec<_>>().join(", "));
            let mut rng = PyRandom::new(crc32(r.as_bytes()) as i128);
            rng.sample(&(0..self.n).collect::<Vec<_>>(), self.top_k)
        } else {
            // (np.argsort(-scores, kind="stable")[:top_k])
            let mut order: Vec<usize> = (0..self.n).collect();
            order.sort_by(|&a, &b| (-scores[a]).partial_cmp(&-scores[b]).unwrap_or(std::cmp::Ordering::Equal));
            order.truncate(self.top_k);
            order
        };
        let best_set: HashSet<usize> = best.iter().copied().collect();
        let mut weights: BTreeMap<(u32, bool), f64> = BTreeMap::new();
        for i in 0..self.deck.len() {
            let d = self.deck[i] as usize;
            if best_set.contains(&d) {
                *weights.entry((self.code[i] as u32, self.extra[i])).or_default() += ((scores[d] + 1.0) as f64) * self.count[i] as f64;
            }
        }
        // (no guess adds a duel-wide effect that no known card has brought into the game)
        let have: BTreeSet<GlobalKey> = present.iter().chain(known_main).chain(known_extra).flat_map(|&c| global_keys(canonical_code(c))).collect();
        let adds = |code: u32| !global_keys(code).is_subset(&have);
        Ok((self.fill_kind(&weights, &known, n_main, false, &adds), self.fill_kind(&weights, &known, n_extra, true, &adds)))
    }
}

/// zlib.crc32
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
        }
    }
    !c
}
