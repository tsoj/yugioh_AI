//! Built-in bots: random, greedy, setter and mixed (the heuristic player: crate::heuristic). They
//! make the choices of the former Python bots with the same seeds (Python's random numbers:
//! ygo_arbiter::pyrandom), so that the belief checks' games stay the games of their baselines.

use std::collections::HashMap;

use ygo_core::constants::*;

use ygo_arbiter::actions::{Command, Response, battle_options, idle_options};
use ygo_arbiter::game::{Bot, BotError};
use ygo_arbiter::messages::Msg;
use ygo_arbiter::pyrandom::PyRandom;
use ygo_arbiter::random::{RandomBot, pick, random_response};
use ygo_arbiter::state::Snapshot;

pub const MAX_SAME_ACTIVATION_PER_TURN: u32 = 3;

/// How often each effect was activated this turn, by (code, location, sequence, description),
/// capped at `MAX_SAME_ACTIVATION_PER_TURN` (so that unlimited effects cannot loop forever); reset
/// by NEW_TURN.
#[derive(Clone, Default)]
pub struct ActivationCaps(HashMap<(u32, u32, u32, Option<u64>), u32>);

impl ActivationCaps {
    fn key(e: &ygo_arbiter::messages::Card) -> (u32, u32, u32, Option<u64>) {
        (e.code, e.loc, e.seq, e.desc)
    }

    pub fn allowed(&self, e: &ygo_arbiter::messages::Card) -> bool {
        self.0.get(&Self::key(e)).copied().unwrap_or(0) < MAX_SAME_ACTIVATION_PER_TURN
    }

    pub fn note(&mut self, e: &ygo_arbiter::messages::Card) {
        *self.0.entry(Self::key(e)).or_insert(0) += 1;
    }

    pub fn observe(&mut self, msg: &Msg) {
        if let Msg::NewTurn { .. } = msg {
            self.0.clear();
        }
    }
}

/// Activates everything it can, summons, sets, always attacks; random where it has no opinion.
pub struct GreedyBot {
    rng: PyRandom,
    caps: ActivationCaps,
}

impl GreedyBot {
    pub fn new(seed: i64) -> GreedyBot {
        GreedyBot { rng: PyRandom::new(seed as i128), caps: ActivationCaps::default() }
    }
}

impl Bot for GreedyBot {
    fn name(&self) -> String {
        "greedy".into()
    }
    fn observe(&mut self, msg: &Msg) {
        self.caps.observe(msg);
    }
    fn respond(&mut self, prompt: &Msg, _state: Option<&Snapshot>, _clock: Option<(f64, f64)>) -> Result<Response, BotError> {
        match prompt {
            Msg::SelectIdlecmd { activatable, spsummonable, .. } => {
                let opts = idle_options(prompt);
                for command in [Command::Activate, Command::Spsummon, Command::Summon, Command::Sset, Command::Mset, Command::ToBp, Command::ToEp] {
                    // (the uses per turn are capped of activations and Special Summons)
                    let list = match command {
                        Command::Activate => Some(activatable),
                        Command::Spsummon => Some(spsummonable),
                        _ => None,
                    };
                    let cands: Vec<Response> = opts
                        .iter()
                        .filter(|o| o.kind == Some(command) && list.is_none_or(|l| self.caps.allowed(&l[o.index.unwrap() as usize])))
                        .cloned()
                        .collect();
                    if !cands.is_empty() {
                        let choice = pick(&mut self.rng, &cands)?;
                        if let Some(l) = list {
                            self.caps.note(&l[choice.index.unwrap() as usize]);
                        }
                        return Ok(choice);
                    }
                }
                pick(&mut self.rng, &opts)
            }
            Msg::SelectBattlecmd { activatable, .. } => {
                let opts = battle_options(prompt);
                let acts: Vec<Response> =
                    opts.iter().filter(|o| o.kind == Some(Command::Activate) && self.caps.allowed(&activatable[o.index.unwrap() as usize])).cloned().collect();
                if !acts.is_empty() {
                    let choice = pick(&mut self.rng, &acts)?;
                    self.caps.note(&activatable[choice.index.unwrap() as usize]);
                    return Ok(choice);
                }
                let attacks: Vec<Response> = opts.iter().filter(|o| o.kind == Some(Command::Attack)).cloned().collect();
                if !attacks.is_empty() {
                    return pick(&mut self.rng, &attacks);
                }
                for command in [Command::ToM2, Command::ToEp] {
                    if let Some(o) = opts.iter().find(|o| o.kind == Some(command)) {
                        return Ok(o.clone());
                    }
                }
                pick(&mut self.rng, &opts)
            }
            Msg::SelectChain { chains, forced, .. } => {
                let cands: Vec<usize> = (0..chains.len()).filter(|&i| self.caps.allowed(&chains[i])).collect();
                if !cands.is_empty() {
                    let i = *self.rng.choice(&cands);
                    self.caps.note(&chains[i]);
                    return Ok(Response::index(i));
                }
                Ok(if *forced != 0 { Response::index(0) } else { Response::pass() })
            }
            Msg::SelectEffectyn { .. } | Msg::SelectYesno { .. } => Ok(Response::yes(true)),
            Msg::SelectPosition { positions, .. } if positions & POS_FACEUP_ATTACK != 0 => {
                Ok(Response { position: Some(POS_FACEUP_ATTACK as i64), ..Response::default() })
            }
            Msg::SelectCard { cancelable, min, max, cards, .. } if *cancelable != 0 && (*max).min(cards.len() as u32) >= (*min).max(1) => {
                // never cancel: pick the maximum amount of cards
                let n = cards.len() as i64;
                let k = (*max as i64).min(n) as usize;
                let mut idx = self.rng.sample(&(0..n).collect::<Vec<_>>(), k);
                idx.sort();
                Ok(Response { indices: Some(idx), ..Response::default() })
            }
            Msg::SelectUnselectCard { select_cards, .. } if !select_cards.is_empty() => Ok(Response::index(self.rng.randrange(select_cards.len()))),
            _ => random_response(prompt, &mut self.rng, 0.0),
        }
    }
}

/// Plays defensively: sets monsters and spells/traps face-down, activates only sometimes in its
/// own turn, responds in chain windows (e.g. with its set traps), attacks now and then.
pub struct SetterBot {
    rng: PyRandom,
    caps: ActivationCaps,
}

impl SetterBot {
    pub fn new(seed: i64) -> SetterBot {
        SetterBot { rng: PyRandom::new(seed as i128), caps: ActivationCaps::default() }
    }
}

impl Bot for SetterBot {
    fn name(&self) -> String {
        "setter".into()
    }
    fn observe(&mut self, msg: &Msg) {
        self.caps.observe(msg);
    }
    fn respond(&mut self, prompt: &Msg, _state: Option<&Snapshot>, _clock: Option<(f64, f64)>) -> Result<Response, BotError> {
        let rng = &mut self.rng;
        let of = |opts: &[Response], command: Command| -> Vec<Response> { opts.iter().filter(|o| o.kind == Some(command)).cloned().collect() };
        match prompt {
            Msg::SelectIdlecmd { activatable, .. } => {
                let opts = idle_options(prompt);
                let acts: Vec<Response> = of(&opts, Command::Activate).into_iter().filter(|o| self.caps.allowed(&activatable[o.index.unwrap() as usize])).collect();
                if !acts.is_empty() && rng.random() < 0.3 {
                    let choice = pick(rng, &acts)?;
                    self.caps.note(&activatable[choice.index.unwrap() as usize]);
                    return Ok(choice);
                }
                for command in [Command::Mset, Command::Sset, Command::Summon] {
                    let cands = of(&opts, command);
                    if !cands.is_empty() && rng.random() < 0.8 {
                        return pick(rng, &cands);
                    }
                }
                if let Some(to_bp) = of(&opts, Command::ToBp).first()
                    && rng.random() < 0.5
                {
                    return Ok(to_bp.clone());
                }
                if let Some(to_ep) = of(&opts, Command::ToEp).first() {
                    return Ok(to_ep.clone());
                }
                pick(rng, &opts)
            }
            Msg::SelectBattlecmd { .. } => {
                let opts = battle_options(prompt);
                let attacks = of(&opts, Command::Attack);
                if !attacks.is_empty() && rng.random() < 0.5 {
                    return pick(rng, &attacks);
                }
                for command in [Command::ToM2, Command::ToEp] {
                    if let Some(o) = opts.iter().find(|o| o.kind == Some(command)) {
                        return Ok(o.clone());
                    }
                }
                pick(rng, &opts)
            }
            Msg::SelectChain { chains, forced, .. } => {
                let cands: Vec<usize> = (0..chains.len()).filter(|&i| self.caps.allowed(&chains[i])).collect();
                if !cands.is_empty() && rng.random() < 0.7 {
                    let i = *rng.choice(&cands);
                    self.caps.note(&chains[i]);
                    return Ok(Response::index(i));
                }
                Ok(if *forced != 0 { Response::index(0) } else { Response::pass() })
            }
            Msg::SelectPosition { positions, .. } if let Some(position) = [POS_FACEDOWN_DEFENSE, POS_FACEUP_DEFENSE].into_iter().find(|p| positions & p != 0) => {
                Ok(Response { position: Some(position as i64), ..Response::default() })
            }
            _ => random_response(prompt, rng, 0.0),
        }
    }
}

/// Greedy, but a random response instead 30% of the time (one random number generator for both).
pub struct MixedBot(GreedyBot);

impl MixedBot {
    pub fn new(seed: i64) -> MixedBot {
        MixedBot(GreedyBot::new(seed))
    }
}

impl Bot for MixedBot {
    fn name(&self) -> String {
        "mixed".into()
    }
    fn observe(&mut self, msg: &Msg) {
        self.0.observe(msg);
    }
    fn respond(&mut self, prompt: &Msg, state: Option<&Snapshot>, clock: Option<(f64, f64)>) -> Result<Response, BotError> {
        if self.0.rng.random() < 0.3 {
            return random_response(prompt, &mut self.0.rng, 0.05);
        }
        self.0.respond(prompt, state, clock)
    }
}

/// A built-in bot by name (random, greedy, setter, mixed, heuristic), seeded as
/// make_bot(name, seed).
pub fn make_bot(name: &str, seed: i64) -> Option<Box<dyn Bot + Send>> {
    Some(match name {
        "random" => Box::new(RandomBot::new(seed)),
        "greedy" => Box::new(GreedyBot::new(seed)),
        "setter" => Box::new(SetterBot::new(seed)),
        "mixed" => Box::new(MixedBot::new(seed)),
        "heuristic" => Box::new(crate::heuristic::HeuristicBot::new(seed)),
        _ => return None,
    })
}
