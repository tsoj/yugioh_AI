//! Built-in agents: random, greedy, setter and mixed (the heuristic player: crate::heuristic). They
//! make the choices of the former Python agents with the same seeds (Python's random numbers:
//! crate::pyrandom), so that the belief checks' games stay the games of their baselines.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use ygo_core::constants::*;

use crate::actions::{Command, DeclCard, Response, available_places, battle_options, check_sum, idle_options, is_declarable};
use crate::game::{Agent, AgentError, card_db};
use crate::messages::Msg;
use crate::pyrandom::PyRandom;
use crate::state::Snapshot;

/// The codes an ANNOUNCE_CARD filter allows, in the card database's order.
pub fn declarable_codes(opcodes: &[u64]) -> Vec<u32> {
    static CACHE: OnceLock<Mutex<HashMap<Vec<u64>, Vec<u32>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(v) = cache.lock().unwrap().get(opcodes) {
        return v.clone();
    }
    let db = card_db();
    let codes: Vec<u32> = db
        .order
        .iter()
        .filter(|code| {
            let c = &db.cards[code];
            let card = DeclCard {
                code: c.code,
                alias: c.alias,
                type_: c.type_,
                race: c.race,
                attribute: c.attribute,
                setcodes: c.setcodes.iter().collect(),
            };
            is_declarable(&card, opcodes)
        })
        .copied()
        .collect();
    cache.lock().unwrap().insert(opcodes.to_vec(), codes.clone());
    codes
}

fn low(param: u32) -> u32 {
    let (o1, o2) = (param & 0xFFFF, param >> 16);
    if o2 != 0 { o1.min(o2) } else { o1 }
}

/// A selection for a SELECT_SUM prompt by depth-first search in a random order.
pub fn random_sum_selection(m: &Msg, rng: &mut PyRandom, budget: usize) -> Option<Vec<usize>> {
    let Msg::SelectSum { mode, acc, max, must_cards, cards, .. } = m else { return None };
    let n = cards.len();
    let mut order: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut order);
    let must_low: u32 = must_cards.iter().map(|c| low(c.param.unwrap_or(0))).sum();
    struct S<'a> {
        m: &'a Msg,
        order: Vec<usize>,
        steps: usize,
        budget: usize,
        mode: u32,
        max: usize,
        acc: u32,
        params: Vec<u32>,
    }
    fn dfs(s: &mut S, start: usize, chosen: &mut Vec<usize>, low_sum: u32) -> Option<Vec<usize>> {
        s.steps += 1;
        if s.steps > s.budget {
            return None;
        }
        if !chosen.is_empty() && check_sum(s.m, chosen) {
            return Some(chosen.clone());
        }
        if s.mode == 0 && (chosen.len() >= s.max || low_sum >= s.acc) {
            return None; // exact sums cannot shrink by adding cards
        }
        for k in start..s.order.len() {
            let c = s.order[k];
            chosen.push(c);
            let res = dfs(s, k + 1, chosen, low_sum + low(s.params[c]));
            chosen.pop();
            if res.is_some() {
                return res;
            }
        }
        None
    }
    let mut s = S { m, order, steps: 0, budget, mode: *mode, max: *max as usize, acc: *acc, params: cards.iter().map(|c| c.param.unwrap_or(0)).collect() };
    dfs(&mut s, 0, &mut Vec::new(), must_low)
}

/// A uniformly-ish random valid response to a prompt.
pub fn random_response(m: &Msg, rng: &mut PyRandom, cancel_prob: f64) -> Result<Response, AgentError> {
    use Msg::*;
    let fail = |t: &str| AgentError::Failed(format!("InvalidResponse: no random response for {t}"));
    Ok(match m {
        SelectIdlecmd { .. } => rng.choice(&idle_options(m)).clone(),
        SelectBattlecmd { .. } => rng.choice(&battle_options(m)).clone(),
        SelectEffectyn { .. } | SelectYesno { .. } => Response::yes(rng.random() < 0.5),
        SelectOption { options, .. } => Response::index(rng.randrange(options.len())),
        AnnounceNumber { values, .. } => Response::index(rng.randrange(values.len())),
        SelectCard { cancelable, min, max, cards, .. } => {
            if *cancelable != 0 && rng.random() < cancel_prob {
                return Ok(Response { cancel: Some(true), ..Response::default() });
            }
            let n = cards.len() as i64;
            let (lo, hi) = (*min as i64, (*max as i64).min(n));
            let k = if hi >= lo { rng.randint(if hi >= 1 { lo.max(1) } else { 0 }, hi) } else { lo };
            let mut idx = rng.sample(&(0..n).collect::<Vec<_>>(), k as usize);
            idx.sort();
            Response { indices: Some(idx), ..Response::default() }
        }
        SelectTribute { cancelable, min, max, cards, .. } => {
            if *cancelable != 0 && rng.random() < cancel_prob {
                return Ok(Response { cancel: Some(true), ..Response::default() });
            }
            let mut order: Vec<i64> = (0..cards.len() as i64).collect();
            rng.shuffle(&mut order);
            let (mut chosen, mut total) = (Vec::new(), 0u32);
            for i in order {
                if total >= *min && !chosen.is_empty() {
                    break;
                }
                chosen.push(i);
                total += cards[i as usize].release_param.unwrap_or(0);
            }
            chosen.truncate((*max).max(1) as usize);
            chosen.sort();
            Response { indices: Some(chosen), ..Response::default() }
        }
        SelectSum { .. } => {
            let mut sel: Vec<i64> = random_sum_selection(m, rng, 20000).unwrap_or_default().into_iter().map(|i| i as i64).collect();
            sel.sort();
            Response { indices: Some(sel), ..Response::default() }
        }
        SelectUnselectCard { finishable, cancelable, select_cards, unselect_cards, .. } => {
            if (*finishable != 0 || *cancelable != 0) && (select_cards.is_empty() || rng.random() < 0.5) {
                return Ok(Response { finish: Some(true), ..Response::default() });
            }
            if !select_cards.is_empty() {
                Response::index(rng.randrange(select_cards.len()))
            } else {
                Response::index(select_cards.len() + rng.randrange(unselect_cards.len()))
            }
        }
        SelectChain { chains, forced, .. } => {
            let n = chains.len();
            if *forced != 0 {
                Response::index(rng.randrange(n))
            } else {
                match rng.randrange(n + 1) {
                    0 => Response::pass(),
                    k => Response::index(k - 1),
                }
            }
        }
        SelectPlace { player, count, flag } | SelectDisfield { player, count, flag } => {
            let places = available_places(*player, *flag);
            let count = (*count as usize).min(places.len());
            Response { places: Some(rng.sample(&places, count)), ..Response::default() }
        }
        SelectPosition { positions, .. } => {
            let options: Vec<i64> = [POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK, POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE]
                .into_iter()
                .filter(|p| positions & p != 0)
                .map(|p| p as i64)
                .collect();
            Response { position: Some(*rng.choice(&options)), ..Response::default() }
        }
        SelectCounter { count, cards, .. } => {
            let mut counts = vec![0i64; cards.len()];
            for _ in 0..*count {
                let cands: Vec<usize> = (0..cards.len()).filter(|&i| counts[i] < cards[i].counters.unwrap_or(0) as i64).collect();
                counts[*rng.choice(&cands)] += 1;
            }
            Response { counts: Some(counts), ..Response::default() }
        }
        SortCard { cards, .. } | SortChain { cards, .. } => {
            let mut order: Vec<i64> = (0..cards.len() as i64).collect();
            rng.shuffle(&mut order);
            Response { order: Some(order), ..Response::default() }
        }
        AnnounceRace { count, available, .. } => {
            let bits: Vec<u64> = (0..64).map(|i| 1u64 << i).filter(|b| available & b != 0).collect();
            let k = (*count as usize).min(bits.len());
            Response { mask: Some(rng.sample(&bits, k).iter().sum()), ..Response::default() }
        }
        AnnounceAttrib { count, available, .. } => {
            let bits: Vec<u64> = (0..32).map(|i| 1u64 << i).filter(|b| *available as u64 & b != 0).collect();
            let k = (*count as usize).min(bits.len());
            Response { mask: Some(rng.sample(&bits, k).iter().sum()), ..Response::default() }
        }
        AnnounceCard { values, .. } => {
            let codes = declarable_codes(values);
            Response { code: Some(if codes.is_empty() { 0 } else { *rng.choice(&codes) as i64 }), ..Response::default() }
        }
        RockPaperScissors { .. } => Response { hand: Some(rng.randint(1, 3)), ..Response::default() },
        _ => return Err(fail(m.name())),
    })
}

/// Picks uniformly random valid responses.
pub struct RandomAgent {
    rng: PyRandom,
}

impl RandomAgent {
    pub fn new(seed: i64) -> RandomAgent {
        RandomAgent { rng: PyRandom::new(seed as i128) }
    }
}

impl Agent for RandomAgent {
    fn name(&self) -> String {
        "random".into()
    }
    fn respond(&mut self, prompt: &Msg, _state: Option<&Snapshot>, _clock: Option<(f64, f64)>) -> Result<Response, AgentError> {
        random_response(prompt, &mut self.rng, 0.05)
    }
}

pub const MAX_SAME_ACTIVATION_PER_TURN: u32 = 3;

/// How often each effect was activated this turn, by (code, location, sequence, description),
/// capped at `MAX_SAME_ACTIVATION_PER_TURN` (so that unlimited effects cannot loop forever); reset
/// by NEW_TURN.
#[derive(Clone, Default)]
pub struct ActivationCaps(HashMap<(u32, u32, u32, Option<u64>), u32>);

impl ActivationCaps {
    fn key(e: &crate::messages::Card) -> (u32, u32, u32, Option<u64>) {
        (e.code, e.loc, e.seq, e.desc)
    }

    pub fn allowed(&self, e: &crate::messages::Card) -> bool {
        self.0.get(&Self::key(e)).copied().unwrap_or(0) < MAX_SAME_ACTIVATION_PER_TURN
    }

    pub fn note(&mut self, e: &crate::messages::Card) {
        *self.0.entry(Self::key(e)).or_insert(0) += 1;
    }

    pub fn observe(&mut self, msg: &Msg) {
        if let Msg::NewTurn { .. } = msg {
            self.0.clear();
        }
    }
}

/// A random one of `opts` (Python's random.choice, which fails on an empty list).
fn pick<T: Clone>(rng: &mut PyRandom, opts: &[T]) -> Result<T, AgentError> {
    if opts.is_empty() {
        return Err(AgentError::Failed("IndexError: Cannot choose from an empty sequence".into()));
    }
    Ok(rng.choice(opts).clone())
}

/// Activates everything it can, summons, sets, always attacks; random where it has no opinion.
pub struct GreedyAgent {
    rng: PyRandom,
    caps: ActivationCaps,
}

impl GreedyAgent {
    pub fn new(seed: i64) -> GreedyAgent {
        GreedyAgent { rng: PyRandom::new(seed as i128), caps: ActivationCaps::default() }
    }
}

impl Agent for GreedyAgent {
    fn name(&self) -> String {
        "greedy".into()
    }
    fn observe(&mut self, msg: &Msg) {
        self.caps.observe(msg);
    }
    fn respond(&mut self, prompt: &Msg, _state: Option<&Snapshot>, _clock: Option<(f64, f64)>) -> Result<Response, AgentError> {
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
pub struct SetterAgent {
    rng: PyRandom,
    caps: ActivationCaps,
}

impl SetterAgent {
    pub fn new(seed: i64) -> SetterAgent {
        SetterAgent { rng: PyRandom::new(seed as i128), caps: ActivationCaps::default() }
    }
}

impl Agent for SetterAgent {
    fn name(&self) -> String {
        "setter".into()
    }
    fn observe(&mut self, msg: &Msg) {
        self.caps.observe(msg);
    }
    fn respond(&mut self, prompt: &Msg, _state: Option<&Snapshot>, _clock: Option<(f64, f64)>) -> Result<Response, AgentError> {
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
pub struct MixedAgent(GreedyAgent);

impl MixedAgent {
    pub fn new(seed: i64) -> MixedAgent {
        MixedAgent(GreedyAgent::new(seed))
    }
}

impl Agent for MixedAgent {
    fn name(&self) -> String {
        "mixed".into()
    }
    fn observe(&mut self, msg: &Msg) {
        self.0.observe(msg);
    }
    fn respond(&mut self, prompt: &Msg, state: Option<&Snapshot>, clock: Option<(f64, f64)>) -> Result<Response, AgentError> {
        if self.0.rng.random() < 0.3 {
            return random_response(prompt, &mut self.0.rng, 0.05);
        }
        self.0.respond(prompt, state, clock)
    }
}

/// A built-in agent by name (random, greedy, setter, mixed, heuristic), seeded as
/// make_agent(name, seed).
pub fn make_agent(name: &str, seed: i64) -> Option<Box<dyn Agent + Send>> {
    Some(match name {
        "random" => Box::new(RandomAgent::new(seed)),
        "greedy" => Box::new(GreedyAgent::new(seed)),
        "setter" => Box::new(SetterAgent::new(seed)),
        "mixed" => Box::new(MixedAgent::new(seed)),
        "heuristic" => Box::new(crate::heuristic::HeuristicAgent::new(seed)),
        _ => return None,
    })
}
