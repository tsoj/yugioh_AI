//! The imitation policy's bot ("imitation"): the network of scripts/policy/train.py (policynet,
//! on the CPU) chooses among the options of each decision, as the player sees it
//! (`policy_features`), without a search. A game's cards are named by one of the text tables'
//! pseudo-name seeds, drawn per game. Prompts it has no options for (and a single option) the
//! rules answer.
//!
//! $BOT_POLICY_NET: an export_policy.py file; $BOT_POLICY_TABLES: an embed_tables.py file.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use policynet::{Inputs, Policy, Tables};
use ygo_arbiter::actions::{Response, check_sum};
use ygo_arbiter::game::{Bot, BotError, InitInfo};
use ygo_arbiter::messages::Msg;
use ygo_arbiter::pyrandom::PyRandom;
use ygo_arbiter::state::Snapshot;

use crate::heuristic::Heuristic;
use crate::policy_features::*;

/// The network and the tables, loaded once per process.
fn loaded() -> Result<&'static (Policy, Tables), String> {
    static LOADED: OnceLock<Result<(Policy, Tables), String>> = OnceLock::new();
    LOADED
        .get_or_init(|| {
            let var = |name: &str| std::env::var_os(name).ok_or_else(|| format!("imitation: ${name} is not set"));
            let net = std::fs::read(var("BOT_POLICY_NET")?).map_err(|e| format!("imitation: {e}"))?;
            let (policy, _) = Policy::from_bytes(&net)?;
            let tables = Tables::load(std::path::Path::new(&var("BOT_POLICY_TABLES")?))?;
            if tables.dim != policy.sizes.emb_dim {
                return Err(format!("imitation: tables of {} dimensions, the net reads {}", tables.dim, policy.sizes.emb_dim));
            }
            Ok((policy, tables))
        })
        .as_ref()
        .map_err(|e| e.clone())
}

pub struct ImitationBot {
    rng: PyRandom,
    seen: Seen,
    codes: Codes,
    deck: Vec<u32>,
    /// the game's pseudo-name seed (a table's)
    names: usize,
    /// text vectors by code, and by (code, string), under the game's names
    cards: HashMap<u32, Option<Arc<[f32]>>>,
    strings: HashMap<(u32, u32), Option<Arc<[f32]>>>,
    fallback: Heuristic,
    /// the last prompt answered: asked again, the rules answer
    last: Option<Msg>,
}

impl ImitationBot {
    pub fn new(seed: i64) -> Result<ImitationBot, String> {
        loaded()?;
        Ok(ImitationBot {
            rng: PyRandom::new(seed as i128),
            seen: Seen::new(0),
            codes: Codes::default(),
            deck: Vec::new(),
            names: 0,
            cards: HashMap::new(),
            strings: HashMap::new(),
            fallback: Heuristic::new(seed as i128, 0),
            last: None,
        })
    }

    fn card(&mut self, tables: &Tables, index: i64) -> Option<Arc<[f32]>> {
        let code = *self.codes.codes.get((index - 1) as usize)?;
        let names = self.names;
        self.cards.entry(code).or_insert_with(|| tables.card(names, code).map(Arc::from)).clone()
    }

    fn string(&mut self, tables: &Tables, index: i64, string: i64) -> Option<Arc<[f32]>> {
        if index <= 0 || string >= S_CARD {
            return None;
        }
        let code = *self.codes.codes.get((index - 1) as usize)?;
        let names = self.names;
        self.strings.entry((code, string as u32)).or_insert_with(|| tables.string(names, code, string as u32).map(Arc::from)).clone()
    }

    /// The options' scores (train.Batcher's truncations; options beyond them: -inf).
    fn scores(&mut self, d: &Decision) -> Vec<f32> {
        let (policy, tables) = loaded().expect("loaded at creation");
        let s = policy.sizes;
        let lt = d.tok.len().min(s.max_tokens);
        let lo = d.opt.len().min(s.max_options);
        let tok: Vec<[i64; TOK]> = d.tok[..lt]
            .iter()
            .map(|t| {
                let mut r = t.map(|x| x as i64);
                if r[T_REF] >= s.max_tokens as i64 {
                    r[T_REF] = -1;
                }
                r
            })
            .collect();
        let opt: Vec<[i64; OPT]> = d.opt[..lo]
            .iter()
            .map(|(o, _)| {
                let mut r = o.map(|x| x as i64);
                if r[O_TOKEN] >= lt as i64 {
                    r[O_TOKEN] = -1;
                }
                r
            })
            .collect();
        let card = |b: &mut Self, i: i64| if i > 0 { b.card(tables, i) } else { None };
        let tok_text: Vec<_> = tok.iter().map(|t| card(self, t[T_CODE])).collect();
        let ref_text: Vec<_> = tok.iter().map(|t| if t[T_REF] >= 0 { card(self, tok[t[T_REF] as usize][T_CODE]) } else { None }).collect();
        let opt_desc_text: Vec<_> = opt.iter().map(|o| card(self, o[O_DESC_CODE])).collect();
        let opt_str_text: Vec<_> = opt.iter().map(|o| self.string(tables, o[O_DESC_CODE], o[O_DESC_STR])).collect();
        let h = d.hdr;
        let glob_text = [H_HINT_CODE, H_DESC_CODE, H_CODE, H_PREV_CODE].map(|c| card(self, h[c]));
        let glob_str_text = [(H_HINT_CODE, H_HINT_STR), (H_DESC_CODE, H_DESC_STR)].map(|(c, s)| self.string(tables, h[c], h[s]));
        let inputs = Inputs {
            hdr: h,
            tok,
            opt,
            tok_text: tok_text.iter().map(|x| x.as_deref()).collect(),
            ref_text: ref_text.iter().map(|x| x.as_deref()).collect(),
            opt_desc_text: opt_desc_text.iter().map(|x| x.as_deref()).collect(),
            opt_str_text: opt_str_text.iter().map(|x| x.as_deref()).collect(),
            glob_text: [glob_text[0].as_deref(), glob_text[1].as_deref(), glob_text[2].as_deref(), glob_text[3].as_deref()],
            glob_str_text: [glob_str_text[0].as_deref(), glob_str_text[1].as_deref()],
        };
        let mut scores = policy.scores(&inputs);
        scores.resize(d.opt.len(), f32::NEG_INFINITY);
        scores
    }

    /// The response of the best options (by `order`, the best first) and the chosen options.
    fn response(prompt: &Msg, d: &Decision, order: &[usize]) -> Option<(Response, Vec<usize>)> {
        let cards: Vec<(usize, usize)> = order.iter().filter_map(|&k| if let Act::Card(i) = d.opt[k].1 { Some((k, i)) } else { None }).collect();
        match prompt {
            Msg::SelectCard { min, .. } => {
                let take = &cards[..(*min).max(1) as usize];
                Some((Response::cards(take.iter().map(|&(_, i)| i).collect()), take.iter().map(|&(k, _)| k).collect()))
            }
            Msg::SelectTribute { min, max, cards: list, .. } => {
                // (the best cards until they release enough)
                let (mut picked, mut total) = (Vec::new(), 0);
                for &(k, i) in &cards {
                    if total >= *min || picked.len() >= *max as usize {
                        break;
                    }
                    total += list[i].release_param.unwrap_or(1);
                    picked.push((k, i));
                }
                (total >= *min).then(|| (Response::cards(picked.iter().map(|&(_, i)| i).collect()), picked.iter().map(|&(k, _)| k).collect()))
            }
            Msg::SelectSum { .. } => {
                // (the first valid selection of a depth-first search over the cards, the best first)
                fn dfs(m: &Msg, cards: &[(usize, usize)], start: usize, chosen: &mut Vec<(usize, usize)>, steps: &mut usize) -> bool {
                    *steps += 1;
                    if *steps > 20_000 {
                        return false;
                    }
                    if !chosen.is_empty() && check_sum(m, &chosen.iter().map(|&(_, i)| i).collect::<Vec<_>>()) {
                        return true;
                    }
                    for j in start..cards.len() {
                        chosen.push(cards[j]);
                        if dfs(m, cards, j + 1, chosen, steps) {
                            return true;
                        }
                        chosen.pop();
                    }
                    false
                }
                let mut chosen = Vec::new();
                dfs(prompt, &cards, 0, &mut chosen, &mut 0).then(|| (Response::cards(chosen.iter().map(|&(_, i)| i).collect()), chosen.iter().map(|&(k, _)| k).collect()))
            }
            Msg::SelectPlace { count, .. } | Msg::SelectDisfield { count, .. } => {
                let places: Vec<(usize, [u32; 3])> = order.iter().filter_map(|&k| if let Act::Place(p) = d.opt[k].1 { Some((k, p)) } else { None }).take(*count as usize).collect();
                Some((Response { places: Some(places.iter().map(|&(_, p)| p).collect()), ..Response::default() }, places.iter().map(|&(k, _)| k).collect()))
            }
            _ => match &d.opt[order[0]].1 {
                Act::Respond(r) => Some((r.clone(), vec![order[0]])),
                _ => None,
            },
        }
    }
}

impl Bot for ImitationBot {
    fn name(&self) -> String {
        "imitation".into()
    }
    fn wants_state(&self) -> bool {
        true
    }
    fn start(&mut self, player: u32, init: &InitInfo) -> Result<(), BotError> {
        let (_, tables) = loaded().map_err(BotError::Failed)?;
        self.seen = Seen::new(player);
        self.codes = Codes::default();
        self.deck = init.deck.main.clone();
        for &c in init.deck.main.iter().chain(&init.deck.extra) {
            self.codes.of(c);
        }
        self.names = self.rng.randrange(tables.seeds());
        self.cards.clear();
        self.strings.clear();
        self.fallback = Heuristic::new(self.rng.randrange(1 << 30) as i128, player);
        self.last = None;
        Ok(())
    }
    fn observe(&mut self, msg: &Msg) {
        self.seen.message(msg);
        self.fallback.observe_msg(msg);
    }
    fn respond(&mut self, prompt: &Msg, state: Option<&Snapshot>, _clock: Option<(f64, f64)>) -> Result<Response, BotError> {
        let again = self.last.as_ref() == Some(prompt);
        self.last = Some(prompt.clone());
        let view = state.ok_or_else(|| BotError::Failed("imitation: no state".into()))?;
        let d = decision(&mut self.codes, &self.seen, view, prompt, &Response::default(), &self.deck);
        let chosen = match &d {
            Some(d) if !again && !d.opt.is_empty() => {
                let scores = if d.opt.len() > 1 { self.scores(d) } else { vec![0.0] };
                let mut order: Vec<usize> = (0..d.opt.len()).collect();
                order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
                Self::response(prompt, d, &order)
            }
            _ => None,
        };
        let Some((resp, picked)) = chosen else {
            self.seen.decided(None);
            return self.fallback.respond_to(prompt, state);
        };
        self.seen.decided(picked.first().map(|&k| d.as_ref().unwrap().chosen_key(k)));
        Ok(resp)
    }
}
