//! What a card's script is about, read statically from its text (belief/script_index.py).
//!
//! The scripts declare the archetypes and cards they refer to (`s.listed_series`,
//! `s.listed_names`); and some checks read values of several cards (how many different Pendulum
//! Scales, the highest Rank in the Extra Deck, a sum of Levels) instead of asking each card a
//! yes/no question. Neither needs the Lua to be understood: the declarations narrow down which
//! cards could satisfy a check, and the engine decides whether one does. Whole-corpus scans are
//! done once per process.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use regex::Regex;

use crate::game::card_db;
use crate::paths::{card_scripts_dir, fixed_card_scripts_dir};

fn re(s: &str) -> Regex {
    Regex::new(s).expect("regex")
}

/// calls that read values of a group of cards (a forced filter answer cannot satisfy them: the
/// values themselves are compared)
fn value_checks() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| re(r"GetClassCount|GetMaxGroup|GetMinGroup|CheckWithSum|GetSum\b|aux\.dpcheck|aux\.dncheck|GetBinClassCount|AnnounceNumber\("))
}

fn read(path: &Path) -> Option<String> {
    std::fs::read(path).ok().map(|b| String::from_utf8_lossy(&b).into_owned())
}

/// The scripts of the corpus subdirectories official, pre-release, unofficial (c*.lua).
fn corpus_scripts() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for sub in ["official", "pre-release", "unofficial"] {
        if let Ok(rd) = std::fs::read_dir(card_scripts_dir().join(sub)) {
            for e in rd.flatten() {
                let p = e.path();
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.starts_with('c') && name.ends_with(".lua") {
                    out.push(p);
                }
            }
        }
    }
    out
}

/// The top-level shared scripts (*.lua in the scripts directory).
fn shared_scripts() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(card_scripts_dir())
        .map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "lua") && p.is_file()).collect())
        .unwrap_or_default();
    out.sort();
    out
}

fn code_of(path: &Path) -> Option<u32> {
    path.file_stem()?.to_str()?.get(1..)?.parse().ok()
}

fn constants() -> &'static HashMap<String, u64> {
    static C: OnceLock<HashMap<String, u64>> = OnceLock::new();
    C.get_or_init(|| {
        let r = re(r"(?m)^\s*((?:SET|CARD|RACE)_[A-Z0-9_]+)\s*=\s*(0x[0-9a-fA-F]+|\d+)");
        let mut out = HashMap::new();
        for path in shared_scripts() {
            let Some(text) = read(&path) else { continue };
            for c in r.captures_iter(&text) {
                if let Some(v) = parse_int(&c[2]) {
                    out.insert(c[1].to_string(), v);
                }
            }
        }
        out
    })
}

fn parse_int(s: &str) -> Option<u64> {
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(h) => u64::from_str_radix(h, 16).ok(),
        None => s.parse().ok(),
    }
}

/// The text of a card's script (the corrected one first), empty if there is none.
pub fn script_text(code: u32) -> String {
    let mut paths = vec![fixed_card_scripts_dir().join(format!("c{code}.lua"))];
    for sub in ["official", "pre-release", "unofficial"] {
        paths.push(card_scripts_dir().join(sub).join(format!("c{code}.lua")));
    }
    paths.iter().find_map(|p| read(p)).unwrap_or_default()
}

fn memo<T: Clone>(cache: &'static OnceLock<Mutex<HashMap<u32, T>>>, code: u32, f: impl FnOnce() -> T) -> T {
    let m = cache.get_or_init(Default::default);
    if let Some(v) = m.lock().unwrap().get(&code) {
        return v.clone();
    }
    let v = f();
    m.lock().unwrap().insert(code, v.clone());
    v
}

/// (listed archetype codes, listed card codes, whether it checks card values)
#[derive(Clone, Debug, Default)]
pub struct ScriptInfo {
    pub series: BTreeSet<u32>,
    pub names: BTreeSet<u32>,
    pub value_checks: bool,
}

pub fn script_info(code: u32) -> ScriptInfo {
    static CACHE: OnceLock<Mutex<HashMap<u32, ScriptInfo>>> = OnceLock::new();
    memo(&CACHE, code, || {
        let text = script_text(code);
        let listed = re(r"s\.listed_(series|names)\s*=\s*\{([^}]*)\}");
        let num = re(r"^(?:0x[0-9a-fA-F]+|\d+)$");
        let mut info = ScriptInfo { value_checks: value_checks().is_match(&text), ..ScriptInfo::default() };
        for c in listed.captures_iter(&text) {
            for tok in c[2].split(',').map(str::trim) {
                if tok.is_empty() {
                    continue;
                }
                let value = if tok == "id" {
                    Some(code as u64)
                } else if num.is_match(tok) {
                    parse_int(tok)
                } else {
                    constants().get(tok).copied()
                };
                if let Some(v) = value {
                    if &c[1] == "series" { info.series.insert(v as u32) } else { info.names.insert(v as u32) };
                }
            }
        }
        info
    })
}

/// Card.IsSetCard: the archetype matches, and the sub-archetype bits if any.
pub fn is_set_card(card_setcodes: impl IntoIterator<Item = u16>, setcode: u32) -> bool {
    card_setcodes.into_iter().any(|s| (s as u32 & 0xfff) == (setcode & 0xfff) && (s as u32 & setcode) == setcode)
}

/// The hidden piles the script refers to: (deck, extra).
pub fn piles(code: u32) -> (bool, bool) {
    static CACHE: OnceLock<Mutex<HashMap<u32, (bool, bool)>>> = OnceLock::new();
    memo(&CACHE, code, || {
        let text = script_text(code);
        (text.contains("LOCATION_DECK"), text.contains("LOCATION_EXTRA"))
    })
}

/// Cards the script of `code` refers to (its archetypes and named cards), legal ones, by code.
pub fn candidates(code: u32) -> Vec<u32> {
    let info = script_info(code);
    let db = card_db();
    let mut out: Vec<u32> = db
        .order
        .iter()
        .filter(|&&c| {
            let d = &db.cards[&c];
            let ot = db.ots.get(&c).copied().unwrap_or(0);
            if d.alias != 0 || ot & 0x3 == 0 || c == code {
                return false;
            }
            info.names.contains(&c) || info.series.iter().any(|&s| is_set_card(d.setcodes.iter(), s))
        })
        .copied()
        .collect();
    out.sort();
    out
}

fn scan(f: impl Fn(&str) -> bool) -> HashSet<u32> {
    corpus_scripts().iter().filter_map(|p| read(p).filter(|t| f(t)).and_then(|_| code_of(p))).collect()
}

/// Cards whose scripts give them their own Normal Summon ("summon") or Set ("mset") procedure
/// that can Tribute the opponent's monsters.
pub fn proc_cards(kind: &str) -> &'static HashSet<u32> {
    static SUMMON: OnceLock<HashSet<u32>> = OnceLock::new();
    static MSET: OnceLock<HashSet<u32>> = OnceLock::new();
    let (cell, name) = if kind == "summon" { (&SUMMON, "EFFECT_SUMMON_PROC") } else { (&MSET, "EFFECT_SET_PROC") };
    cell.get_or_init(|| scan(|t| t.contains(name) && t.contains("1-tp") && (t.contains("Release") || t.contains("RELEASE"))))
}

/// Cards whose own Set procedure discards cards as its cost (e.g. Infernal Incinerator).
pub fn discard_set_proc_cards() -> &'static HashSet<u32> {
    static S: OnceLock<HashSet<u32>> = OnceLock::new();
    S.get_or_init(|| scan(|t| (t.contains("EFFECT_SET_PROC") || t.contains("EFFECT_LIMIT_SET_PROC")) && t.contains("REASON_DISCARD")))
}

/// Cards whose own Set procedure Tributes `n` monsters (e.g. The Wicked Avatar: 3).
pub fn set_proc_tribute_cards(n: u32) -> HashSet<u32> {
    static ALL: OnceLock<Vec<(u32, u32, u32)>> = OnceLock::new();
    let all = ALL.get_or_init(|| {
        let r = re(r"AddNormalSetProcedure\(\s*c\s*,[^,]*,[^,]*,\s*(\d+)\s*,\s*(\d+)");
        let mut out = Vec::new();
        for p in corpus_scripts() {
            let (Some(text), Some(code)) = (read(&p), code_of(&p)) else { continue };
            for c in r.captures_iter(&text) {
                if let (Ok(lo), Ok(hi)) = (c[1].parse(), c[2].parse()) {
                    out.push((code, lo, hi));
                }
            }
        }
        out
    });
    all.iter().filter(|&&(_, lo, hi)| lo <= n && n <= hi).map(|&(c, _, _)| c).collect()
}

/// Whether the card's own procedure reveals it together with other cards (e.g. Evil Dragon
/// Cards), as opposed to revealing other cards only (Crimson Nova the Dark Cubic Lord).
pub fn reveals_itself(code: u32) -> bool {
    static CACHE: OnceLock<Mutex<HashMap<u32, bool>>> = OnceLock::new();
    memo(&CACHE, code, || re(r"Group\.FromCards\(c\)|:IsContains\(c\)").is_match(&script_text(code)))
}

/// Whether the card's script has a trigger effect (EFFECT_TYPE_TRIGGER_O/F).
pub fn has_trigger(code: u32) -> bool {
    static CACHE: OnceLock<Mutex<HashMap<u32, bool>>> = OnceLock::new();
    memo(&CACHE, code, || script_text(code).contains("EFFECT_TYPE_TRIGGER_"))
}

/// Whether the card cannot be Normal Summoned or Set by its own script.
pub fn unsummonable(code: u32) -> bool {
    static CACHE: OnceLock<Mutex<HashMap<u32, bool>>> = OnceLock::new();
    memo(&CACHE, code, || re(r"EnableUnsummonable\(|EFFECT_UNSUMMONABLE_CARD|EnableReviveLimit\(").is_match(&script_text(code)))
}

/// Cards that can win the game by themselves (their script calls Duel.Win) and the cards those
/// scripts name (e.g. the Forbidden One pieces).
pub fn win_condition_codes() -> &'static HashSet<u32> {
    static S: OnceLock<HashSet<u32>> = OnceLock::new();
    S.get_or_init(|| {
        let num = re(r"\b(\d{6,9})\b");
        let mut paths = corpus_scripts();
        if let Ok(rd) = std::fs::read_dir(fixed_card_scripts_dir()) {
            paths.extend(rd.flatten().map(|e| e.path()).filter(|p| {
                let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                n.starts_with('c') && n.ends_with(".lua")
            }));
        }
        let mut out = HashSet::new();
        for p in paths {
            let Some(text) = read(&p) else { continue };
            if text.contains("Duel.Win(") {
                if let Some(c) = code_of(&p) {
                    out.insert(c);
                }
                out.extend(num.captures_iter(&text).filter_map(|c| c[1].parse::<u32>().ok()));
            }
        }
        out
    })
}

/// A key of a duel-wide effect: a card's own (by its code) or a shared helper's (by its name).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GlobalKey {
    Code(u32),
    Helper(String),
}

fn global_registration() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| re(r"Effect\.GlobalEffect\(|GlobalCheck\("))
}

/// A call of one of the `global_helpers` (the helper's name: group 1; None without helpers). (A
/// call ends at its name's parenthesis and none starts inside another's name, so one regex finds
/// each helper that a separate one per helper would.)
fn global_helper_calls() -> Option<&'static Regex> {
    static R: OnceLock<Option<Regex>> = OnceLock::new();
    R.get_or_init(|| {
        let helpers = global_helpers();
        (!helpers.is_empty()).then(|| re(&format!(r"\b({})\s*\(", helpers.iter().map(|h| regex::escape(h)).collect::<Vec<_>>().join("|"))))
    })
    .as_ref()
}

/// Builds the whole-corpus scans and regexes once per process (before the clock runs: an engine
/// process plays one game, and each scan reads every script).
pub fn warm_up() {
    value_checks();
    constants();
    proc_cards("summon");
    proc_cards("mset");
    discard_set_proc_cards();
    set_proc_tribute_cards(0);
    win_condition_codes();
    global_registration();
    global_helper_calls();
}

/// Functions of the shared scripts that register a duel-wide effect once per duel for whichever
/// card calls them first, by the last part of their name.
fn global_helpers() -> &'static BTreeSet<String> {
    static S: OnceLock<BTreeSet<String>> = OnceLock::new();
    S.get_or_init(|| {
        let func = re(r"(?ms)^(?:local\s+)?function\s+([\w.:]+)\s*\(.*?^end\b");
        let mut out = BTreeSet::new();
        for p in shared_scripts() {
            let Some(text) = read(&p) else { continue };
            for m in func.captures_iter(&text) {
                let name = m[1].rsplit(['.', ':']).next().unwrap_or("").to_string();
                if name != "GlobalCheck" && global_registration().is_match(&m[0]) {
                    out.insert(name);
                }
            }
        }
        out
    })
}

/// The duel-wide effects a card's script registers when the card is created (see the Python
/// docstring).
pub fn global_keys(code: u32) -> BTreeSet<GlobalKey> {
    static CACHE: OnceLock<Mutex<HashMap<u32, BTreeSet<GlobalKey>>>> = OnceLock::new();
    memo(&CACHE, code, || {
        let text = script_text(code);
        let mut out = BTreeSet::new();
        if global_registration().is_match(&text) {
            out.insert(GlobalKey::Code(code));
        }
        if let Some(calls) = global_helper_calls() {
            for c in calls.captures_iter(&text) {
                out.insert(GlobalKey::Helper(c[1].to_string()));
            }
        }
        out
    })
}

/// The races the script of `code` checks for (IsRace(RACE_...)), as a mask.
pub fn races(code: u32) -> u64 {
    static CACHE: OnceLock<Mutex<HashMap<u32, u64>>> = OnceLock::new();
    memo(&CACHE, code, || {
        let check = re(r"IsRace,?\s*\(?\s*(?:c\s*,\s*)?((?:RACE_[A-Z_]+\s*[|+]?\s*)+)");
        let name = re(r"RACE_[A-Z_]+");
        let mut out = 0u64;
        for g in check.captures_iter(&script_text(code)) {
            for n in name.find_iter(&g[1]) {
                out |= constants().get(n.as_str()).copied().unwrap_or(0);
            }
        }
        out
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_helper_calls_match_one_regex_per_helper() {
        let scripts = corpus_scripts();
        for p in scripts.iter().step_by((scripts.len() / 3000).max(1)) {
            let Some(text) = read(p) else { continue };
            let one: BTreeSet<String> = global_helper_calls().map_or_else(BTreeSet::new, |r| r.captures_iter(&text).map(|c| c[1].to_string()).collect());
            let each: BTreeSet<String> = global_helpers().iter().filter(|h| re(&format!(r"\b{}\s*\(", regex::escape(h))).is_match(&text)).cloned().collect();
            assert_eq!(one, each, "{}", p.display());
        }
    }
}
