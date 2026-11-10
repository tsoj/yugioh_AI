//! Measures a trained evaluation's stage calibration (search::StageCalibration): its bias by the
//! stage of the turn, so that the bot can subtract it.
//!
//!   bot-eval-calib MODEL.json RECORDS.jsonl... [--write] [--rows FILE]
//!
//! RECORDS: the games of matches (ygo-arbiter --records). The tool replays each game on the true
//! duel. At each decision that a bot searched (a label), it takes the model's value for the player
//! who decides, without the calibration. Then it fits the results on these values: the logit of a
//! win is slope * value / 1000 plus a constant per stage. A stage's bias is the value at which the
//! fit gives half the games. A drawn game does not count (a game that reaches the turn limit can
//! have thousands of decisions of one stage, more than all other games together). The tool prints each stage's bias and its
//! standard error (too small: the decisions of a game count as independent). If the model has a
//! calibration, the tool also prints the bias of the model's values with it (near 0 if it fits
//! these games). With --write the model file gets the calibration (`stage_calibration`, its other
//! fields unchanged). --rows FILE: one JSON line per decision for other analyses (the game, the
//! decision, its player, the turn, the turn player, the phase, the prompt, the value without the
//! calibration, the label's search value and depth, the winner). A game whose replay fails does
//! not count (its number on stderr).

use std::io::{BufRead, Write};

use bot::search::{LearnedEval, STAGE_PHASES, StageCalibration, stage};
use ygo_arbiter::actions::encode;
use ygo_arbiter::game::{GameRecord, init, run_until_prompt};
use ygo_arbiter::messages::Msg;
use ygo_arbiter::observe::to_core_response;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// (the stages: each phase group for the turn player and for the other player)
const STAGES: usize = STAGE_PHASES.len() * 2;
/// (`fit`: the steps of Newton's method; it converges in fewer)
const NEWTON_STEPS: usize = 30;

/// A searched decision: its stage (group * 2 + other), the mover's value without the calibration
/// and with the model's, and the mover's result (1 a win, 0 a loss).
struct Sample {
    stage: usize,
    uncalibrated: f64,
    calibrated: f64,
    result: f64,
}

/// The searched decisions of a game with a winner (`rows`: their JSON lines); None if the replay
/// fails.
fn samples(record: &GameRecord, game: usize, eval: &LearnedEval, calibrated: Option<&LearnedEval>, rows: &mut Option<impl Write>) -> Option<Vec<Sample>> {
    let Some(winner) = record.result.get("winner").and_then(|w| w.as_u64()) else {
        return Some(Vec::new());
    };
    let mut duel = record.setup(None).ok()?;
    duel.ocg_start_duel();
    let mut out = Vec::new();
    let mut prompt: Option<Msg> = None;
    for (i, entry) in record.responses.iter().enumerate() {
        let p = run_until_prompt(&mut duel, prompt.as_ref()).ok()?.prompt?;
        let player = p.player().unwrap_or(0);
        if player != entry.player {
            return None;
        }
        if let Some(label) = record.labels.iter().find(|l| l.decision == i)
            && let Some((group, other)) = stage(&duel, player)
        {
            let uncalibrated = eval.mover_value_uncalibrated(&mut duel, player, Some(&p));
            out.push(Sample {
                stage: group * 2 + other,
                uncalibrated,
                calibrated: calibrated.map_or(uncalibrated, |e| e.value(&mut duel, player, Some(&p))),
                result: (winner == player as u64) as u8 as f64,
            });
            if let Some(rows) = rows {
                let row = serde_json::json!({
                    "game": game, "decision": i, "player": player, "turn": duel.infos.turn_id, "turn_player": duel.infos.turn_player,
                    "phase": duel.infos.phase, "prompt": p.name(), "static": uncalibrated, "value": label.value, "depth": label.depth, "winner": winner,
                });
                writeln!(rows, "{row}").unwrap();
            }
        }
        let core = to_core_response(&p, player, &entry.response).ok()?;
        duel.ocg_set_response(&encode(&p, &core).ok()?);
        prompt = Some(p);
    }
    Some(out)
}

/// A logistic fit of the results on the values: the slope (the logit of a win per 1000 LP), and
/// per stage the bias (the value with half the wins, in LP) and its standard error.
struct Fit {
    slope: f64,
    bias: [f64; STAGES],
    error: [f64; STAGES],
}

/// The fit of logit(win) = slope * value / 1000 + constant[stage] on (stage, value, result) by
/// Newton's method. The Hessian has the slope's row and column and a diagonal for the constants,
/// so each step solves it directly (`schur`: the slope's entry after the constants are eliminated).
fn fit(data: &[(usize, f64, f64)]) -> Fit {
    let (mut slope, mut constant) = (0.0, [0.0; STAGES]);
    let (mut schur, mut h, mut h_slope) = (0.0, [0.0; STAGES], [0.0; STAGES]);
    for _ in 0..NEWTON_STEPS {
        let (mut g_slope, mut h_slope_slope, mut g) = (0.0, 0.0, [0.0; STAGES]);
        (h, h_slope) = ([0.0; STAGES], [0.0; STAGES]);
        for &(s, value, result) in data {
            let x = value / 1000.0;
            let p = 1.0 / (1.0 + (-(slope * x + constant[s])).exp());
            let q = p * (1.0 - p);
            g_slope += x * (result - p);
            h_slope_slope += q * x * x;
            g[s] += result - p;
            h[s] += q;
            h_slope[s] += q * x;
        }
        let used = |s: &usize| h[*s] > 0.0;
        schur = h_slope_slope - (0..STAGES).filter(used).map(|s| h_slope[s] * h_slope[s] / h[s]).sum::<f64>();
        let step = (g_slope - (0..STAGES).filter(used).map(|s| h_slope[s] * g[s] / h[s]).sum::<f64>()) / schur;
        slope += step;
        for s in (0..STAGES).filter(used) {
            constant[s] += (g[s] - h_slope[s] * step) / h[s];
        }
    }
    let mut out = Fit { slope, bias: [0.0; STAGES], error: [0.0; STAGES] };
    for s in (0..STAGES).filter(|s| h[*s] > 0.0) {
        out.bias[s] = -constant[s] / slope * 1000.0;
        // (the constant's variance: its entry of the inverse Hessian)
        out.error[s] = (1.0 / h[s] + (h_slope[s] / h[s]).powi(2) / schur).sqrt() / slope * 1000.0;
    }
    out
}

fn usage() -> ! {
    eprintln!("usage: bot-eval-calib MODEL.json RECORDS.jsonl... [--write] [--rows FILE]");
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag_value = |flag: &str| args.iter().position(|a| a == flag).map(|i| args[i + 1].clone());
    let rows_path = flag_value("--rows");
    let write = args.iter().any(|a| a == "--write");
    let paths: Vec<&String> = args.iter().filter(|a| !a.starts_with("--") && Some(*a) != rows_path.as_ref()).collect();
    let [model_path, records @ ..] = paths.as_slice() else { usage() };
    if records.is_empty() {
        usage();
    }
    let text = std::fs::read_to_string(model_path).expect("model file");
    // (the model without its calibration, which can be of other weights: the one to measure)
    let mut model: serde_json::Value = serde_json::from_str(&text).expect("model file: JSON");
    let stored = model.as_object_mut().expect("model file: an object").remove("stage_calibration");
    let eval = LearnedEval::from_json(&model.to_string(), model_path);
    let calibrated = stored
        .map(|c| serde_json::from_value::<StageCalibration>(c).expect("stage_calibration"))
        .filter(|c| c.weights_hash == eval.weights_hash())
        .map(|c| {
            let mut e = eval.clone();
            e.stage_calibration = Some(c);
            e
        });

    init(None).unwrap();
    let mut rows = rows_path.map(|p| std::io::BufWriter::new(std::fs::File::create(&p).unwrap_or_else(|e| panic!("{p}: {e}"))));
    let (mut all, mut games, mut counted) = (Vec::new(), 0, 0);
    for path in records {
        let file = std::io::BufReader::new(std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}")));
        for (line, text) in file.lines().enumerate() {
            let record: GameRecord = serde_json::from_str(&text.unwrap()).unwrap_or_else(|e| panic!("{path} line {}: not a record: {e}", line + 1));
            match samples(&record, games, &eval, calibrated.as_ref(), &mut rows) {
                Some(s) => {
                    counted += !s.is_empty() as u64;
                    all.extend(s);
                }
                None => eprintln!("game {games} ({path} line {}): the replay failed", line + 1),
            }
            games += 1;
        }
    }
    assert!(!all.is_empty(), "no searched decisions in the records");

    let measured = fit(&all.iter().map(|s| (s.stage, s.uncalibrated, s.result)).collect::<Vec<_>>());
    let with_stored = calibrated.as_ref().map(|_| fit(&all.iter().map(|s| (s.stage, s.calibrated, s.result)).collect::<Vec<_>>()));
    println!("{model_path}: {} searched decisions of {counted} games with a winner, slope {:.4} per 1000 LP", all.len(), measured.slope);
    println!("{:<8} {:<12} {:>9} {:>8} {:>6}{}", "phase", "mover", "positions", "bias", "+-", if with_stored.is_some() { "   with the model's calibration" } else { "" });
    for s in 0..STAGES {
        let positions = all.iter().filter(|x| x.stage == s).count();
        let mover = ["turn player", "other player"][s % 2];
        let stored = with_stored.as_ref().map_or(String::new(), |f| format!("   {:>+8.0} {:>6.0}", f.bias[s], f.error[s]));
        println!("{:<8} {:<12} {:>9} {:>+8.0} {:>6.0}{stored}", STAGE_PHASES[s / 2], mover, positions, measured.bias[s], measured.error[s]);
    }
    if write {
        let calibration = StageCalibration {
            bias: std::array::from_fn(|group| [measured.bias[group * 2], measured.bias[group * 2 + 1]]),
            slope: measured.slope,
            positions: all.len() as u64,
            games: counted,
            data: records.iter().map(|p| p.as_str()).collect::<Vec<_>>().join(" "),
            weights_hash: eval.weights_hash(),
        };
        let mut model: serde_json::Value = serde_json::from_str(&text).unwrap();
        model["stage_calibration"] = serde_json::to_value(&calibration).unwrap();
        std::fs::write(model_path, serde_json::to_string(&model).unwrap()).expect("write model");
        println!("wrote the calibration to {model_path}");
    }
}
