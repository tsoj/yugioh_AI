//! Training rows from recorded games: a game (a [`GameRecord`]) plus labels at some of its
//! decisions (the search's value and depth) is the data; features are computed from it by
//! replaying, so a new feature set needs no new search (ygoai-datagen, ygoai-refeature).
//!
//! A row: `game`, `decision` (the response's index in the record), the mover (`player`), `turn`,
//! `turn_player`, `phase`, `prompt`, the labels (`value`, `depth`), `result` (for the mover: 1 win,
//! 0 draw or unfinished, -1 loss) and a field per requested feature set (see [`FEATURE_SETS`]).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use ygo_core::api::QueryInfo;
use ygo_core::constants::*;
use ygo_core::duel::Duel;

use crate::actions::encode;
use crate::game::{GameRecord, run_until_prompt};
use crate::messages::Msg;
use crate::observe::to_core_response;
use crate::search::{eval_features, evaluate_at, mover_features, mover_features2, mover_features3, mover_features4, rich_features};
use crate::state::parse_location_query;

/// The feature sets a row can carry:
/// - `mover3`: the 104 features of the `mover3` evaluation (`search::mover_features3`)
/// - `mover2`: the 87 features of the `mover2` evaluation (`search::mover_features2`)
/// - `mover`: its first 49 (`search::mover_features`)
/// - `eval`: the hand-written evaluation's terms for the mover (`search::eval_features`)
/// - `rich`: `search::rich_features` for the mover
/// - `static`: the evaluation in use (YGOAI_EVAL) for the mover
/// - `state`: both sides as JSON, the mover first (see `side`; the true duel, hidden cards
///   included: not for features as they are)
pub const FEATURE_SETS: [&str; 8] = ["mover4", "mover3", "mover2", "mover", "eval", "rich", "static", "state"];

pub use crate::game::Label;

/// A game of the data: its number (the line of its opening in the book), the record and the labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameData {
    pub game: usize,
    pub record: GameRecord,
    pub labels: Vec<Label>,
    /// (the record's responses are to the prompts as the players saw them, a match's record
    /// (`game::play`, translated by `observe::to_core_response`), not in the core's order as
    /// ygoai-datagen's)
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub viewer_responses: bool,
}

/// One player's side: pile sizes, the monster zones (code, face-up, attack position, ATK, DEF,
/// level, rank, type) and the spell/trap zones (code, face-down, type); empty zones are null.
pub fn side(d: &mut Duel, p: u32) -> Value {
    let q = |d: &mut Duel, loc: u32, flags: u32| parse_location_query(&d.ocg_query_location(&QueryInfo { flags, con: p as u8, loc, seq: 0, overlay_seq: 0 }));
    let mz: Vec<Value> = q(d, LOCATION_MZONE, QUERY_CODE | QUERY_POSITION | QUERY_TYPE | QUERY_LEVEL | QUERY_RANK | QUERY_ATTACK | QUERY_DEFENSE)
        .into_iter()
        .map(|c| match c {
            None => Value::Null,
            Some(c) => {
                let pos = c.position.unwrap_or(0);
                json!([c.code, (pos & POS_FACEUP != 0) as u8, (pos & POS_ATTACK != 0) as u8, c.attack, c.defense, c.level, c.rank, c.type_])
            }
        })
        .collect();
    let sz: Vec<Value> = q(d, LOCATION_SZONE, QUERY_CODE | QUERY_POSITION | QUERY_TYPE)
        .into_iter()
        .map(|c| match c {
            None => Value::Null,
            Some(c) => json!([c.code, (c.position.unwrap_or(0) & POS_FACEDOWN != 0) as u8, c.type_]),
        })
        .collect();
    let n = |d: &mut Duel, loc: u32| d.ocg_query_count(p as u8, loc);
    json!({
        "lp": d.player[p as usize].lp,
        "hand": n(d, LOCATION_HAND), "deck": n(d, LOCATION_DECK), "grave": n(d, LOCATION_GRAVE),
        "removed": n(d, LOCATION_REMOVED), "extra": n(d, LOCATION_EXTRA),
        "mzone": mz, "szone": sz,
    })
}

/// Feature set `set` of the position at `prompt` for its mover.
pub fn features(d: &mut Duel, prompt: &Msg, set: &str) -> Result<Value, String> {
    let me = prompt.player().ok_or("a prompt without a player")?;
    Ok(match set {
        "mover3" => json!(mover_features3(d, me, Some(prompt))),
        "mover4" => json!(mover_features4(d, me, Some(prompt))),
        "mover2" => json!(mover_features2(d, me, Some(prompt))),
        "mover" => json!(mover_features(d, me, Some(prompt))),
        "eval" => json!(eval_features(d, me).to_vec()),
        "rich" => json!(rich_features(d, me)),
        "static" => json!(evaluate_at(d, me, Some(prompt))),
        "state" => json!([side(d, me), side(d, 1 - me)]),
        _ => return Err(format!("unknown feature set {set} (known: {})", FEATURE_SETS.join(", "))),
    })
}

/// The row of a labeled decision (`result` is filled in by [`set_results`]).
pub fn row(d: &mut Duel, game: usize, prompt: &Msg, label: &Label, sets: &[String]) -> Result<Value, String> {
    let [turn, turn_player, phase, ..] = d.ext_get_info();
    let mut r = Map::new();
    for (k, v) in [
        ("game", json!(game)),
        ("decision", json!(label.decision)),
        ("player", json!(label.player)),
        ("turn", json!(turn)),
        ("turn_player", json!(turn_player)),
        ("phase", json!(phase)),
        ("prompt", json!(prompt.name())),
        ("value", json!(label.value)),
        ("depth", json!(label.depth)),
    ] {
        r.insert(k.into(), v);
    }
    for s in sets {
        r.insert(s.clone(), features(d, prompt, s)?);
    }
    Ok(Value::Object(r))
}

/// Each row's `result` for its mover from the record's winner.
pub fn set_results(rows: &mut [Value], record: &GameRecord) {
    let winner = record.result.get("winner").and_then(|w| w.as_u64());
    for r in rows {
        let me = r["player"].as_u64().unwrap_or(0);
        r["result"] = json!(match winner {
            Some(w) if w == me => 1,
            Some(w) if w < 2 => -1,
            _ => 0,
        });
    }
}

/// The rows of a game: its record replayed (no search), features `sets` at each labeled decision.
pub fn replay_rows(g: &GameData, sets: &[String]) -> Result<Vec<Value>, String> {
    let labels: HashMap<usize, &Label> = g.labels.iter().map(|l| (l.decision, l)).collect();
    let mut duel = g.record.setup(None).map_err(|e| e.0)?;
    duel.ocg_start_duel();
    let (mut rows, mut prompt): (Vec<Value>, Option<Msg>) = (Vec::new(), None);
    for (i, entry) in g.record.responses.iter().enumerate() {
        let step = run_until_prompt(&mut duel, prompt.as_ref()).map_err(|e| e.0)?;
        let p = step.prompt.ok_or_else(|| format!("game {}: the duel ended before response {i}", g.game))?;
        if let Some(l) = labels.get(&i) {
            rows.push(row(&mut duel, g.game, &p, l, sets)?);
        }
        let core = if g.viewer_responses {
            to_core_response(&p, entry.player, &entry.response).map_err(|e| format!("game {}: response {i}: {e:?}", g.game))?
        } else {
            entry.response.clone()
        };
        let r = encode(&p, &core).map_err(|e| format!("game {}: response {i}: {e:?}", g.game))?;
        duel.ocg_set_response(&r);
        prompt = Some(p);
    }
    set_results(&mut rows, &g.record);
    Ok(rows)
}
