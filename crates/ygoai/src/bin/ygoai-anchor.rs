//! Measures a trained evaluation's LP anchor (search::LpAnchor): how much its output changes per
//! LP, so that the engine can give its values in LP.
//!
//!   ygoai-anchor MODEL.json ROWS.jsonl [--write] [--max N]
//!
//! ROWS: data rows (ygoai-datagen, ygoai-refeature) with the model's features under its feature
//! set's name, else "mover3", "mover2", "mover_x" or "mover" (or "x"; a model reads the first of
//! a larger set's, as `mover`'s 49 are `mover2`'s first); every row
//! whose two players both have LP_ANCHOR_MIN_LP or more counts (at most N, default 200000,
//! spread over the file). For each, the mover's LP goes up and
//! the opponent's down by LP_ANCHOR_STEP against the reverse; the slope is the mean change of the
//! output per LP of the LP difference, and the factor 1 / slope. With --write the model file gets
//! the anchor (`lp_anchor`, its other fields unchanged).

use ygoai::search::{LP_ANCHOR_MIN_LP, LP_ANCHOR_STEP, LearnedEval, LpAnchor};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: ygoai-anchor MODEL.json ROWS.jsonl [--write] [--max N]");
        std::process::exit(2);
    }
    let (model_path, rows_path) = (&args[1], &args[2]);
    let write = args.iter().any(|a| a == "--write");
    let max: usize = args.iter().position(|a| a == "--max").map_or(200_000, |i| args[i + 1].parse().unwrap());
    let text = std::fs::read_to_string(model_path).expect("model file");
    let eval = LearnedEval::from_json(&text, model_path);
    let (me, opp) = eval.lp_features().expect("an LP anchor needs a mover or mover2 model");

    let rows = std::fs::read_to_string(rows_path).expect("rows file");
    let lines: Vec<&str> = rows.lines().filter(|l| !l.trim().is_empty()).collect();
    let every = lines.len().div_ceil(max * 2).max(1);
    let (mut slopes, mut low) = (Vec::new(), Vec::new());
    for line in lines.iter().step_by(every) {
        let row: serde_json::Value = serde_json::from_str(line).expect("row");
        let Some(x) = [eval.feature_set(), "mover3", "mover2", "mover_x", "mover", "x"].iter().find_map(|k| row.get(*k)).and_then(|v| v.as_array()) else {
            continue;
        };
        // (`mover`'s 49 features are the first of `mover2`'s 87)
        let x: Vec<f64> = x.iter().take(eval.n_features()).map(|v| v.as_f64().unwrap()).collect();
        let (lp_me, lp_opp) = (x[me] * 1000.0, x[opp] * 1000.0);
        let step = LP_ANCHOR_STEP / 1000.0;
        let shifted = |sign: f64| {
            let mut y = x.clone();
            (y[me], y[opp]) = (y[me] + sign * step, y[opp] - sign * step);
            eval.mover_output(&y)
        };
        let slope = (shifted(1.0) - shifted(-1.0)) / (4.0 * LP_ANCHOR_STEP);
        if lp_me >= LP_ANCHOR_MIN_LP && lp_opp >= LP_ANCHOR_MIN_LP {
            if slopes.len() < max {
                slopes.push(slope);
            }
        } else if lp_me > LP_ANCHOR_STEP && lp_opp > LP_ANCHOR_STEP {
            low.push(slope);
        }
    }
    assert!(!slopes.is_empty(), "no rows with both players at {LP_ANCHOR_MIN_LP} LP or more");
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let mut sorted = slopes.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| sorted[((sorted.len() - 1) as f64 * p) as usize];
    let slope = mean(&slopes);
    let anchor = LpAnchor { factor: 1.0 / slope, slope, positions: slopes.len() as u64, data: rows_path.clone(), weights_hash: eval.weights_hash() };
    println!(
        "{model_path}: slope {slope:.6} points per LP over {} positions (10/50/90%: {:.6} {:.6} {:.6}; below {LP_ANCHOR_MIN_LP} LP: {:.6} over {}), factor {:.3}, hash {}",
        slopes.len(),
        q(0.1),
        q(0.5),
        q(0.9),
        mean(&low),
        low.len(),
        anchor.factor,
        anchor.weights_hash
    );
    if write {
        let mut model: serde_json::Value = serde_json::from_str(&text).unwrap();
        model["lp_anchor"] = serde_json::to_value(&anchor).unwrap();
        std::fs::write(model_path, serde_json::to_string(&model).unwrap()).expect("write model");
        println!("wrote the anchor to {model_path}");
    }
}
