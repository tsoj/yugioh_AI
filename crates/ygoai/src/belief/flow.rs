//! Exact hypothesis construction as an integer flow problem (belief/flow.py; see there for the
//! model). The model is built in the same order as the Python one (commodities, edges, columns,
//! rows) and solved by the same HiGHS version, so it has the same solution.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use ygo_core::constants::*;

use crate::belief::completion::{Completion, VanillaCompletion, card, fits_need, monster_sset_codes};
use crate::belief::tracker::{Hypothesis, Kind, Need, PoolLoc, Result, ShuffleOutcome, Tracker, TrackerError, VarId};

const WILD_MAIN: &str = "main";
const WILD_EXTRA: &str = "extra";
const WILDS: [&str; 8] = [WILD_MAIN, WILD_EXTRA, "st", "field", "monster", "mset0", "mset1", "mset2"];
/// identity sets without known codes are modelled only while small
const MAX_UNTAGGED_ALLOWED: usize = 8;

fn err<T>(s: impl Into<String>) -> Result<T> {
    Err(TrackerError(s.into()))
}

fn need_wilds(need: Need) -> Vec<&'static str> {
    match need {
        Need::Mset(t) => vec![["mset0", "mset1", "mset2"][t.min(2) as usize]],
        Need::Monster => vec!["monster", "mset0", "mset1", "mset2"],
        Need::St => vec!["st"],
        Need::Field => vec!["field"],
    }
}

fn wild_need(wild: &str) -> Option<Need> {
    if let Some(t) = wild.strip_prefix("mset") {
        return t.parse().ok().map(Need::Mset);
    }
    match wild {
        "st" => Some(Need::St),
        "field" => Some(Need::Field),
        "monster" => Some(Need::Monster),
        _ => None,
    }
}

/// What a commodity is: a known code, a typed wildcard, or a tag (ref var id, code).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    Code(u32),
    Wild(&'static str),
    Tag(VarId, u32),
}

/// A commodity: (owner (None for tags), key).
type Commodity = (Option<u32>, Key);

/// What a value of code_of is: a code or a wildcard's name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CodeOf {
    Code(u32),
    Wild(&'static str),
}

fn code_of_key(k: Key) -> CodeOf {
    match k {
        Key::Code(c) | Key::Tag(_, c) => CodeOf::Code(c),
        Key::Wild(w) => CodeOf::Wild(w),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Producer {
    Atoms(u32, Kind),
    Created(u32),
    Pool(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EdgeKey {
    producer: Producer,
    /// the consuming pool (None: the sink)
    consumer: Option<usize>,
    /// a constrained slot's own edge
    var: Option<VarId>,
    owner: u32,
    /// unconstrained slots: (kind, pool location, seen owner)
    group: Option<(Kind, Option<PoolLoc>, Option<u32>)>,
}

/// An ordered multiset (collections.Counter in insertion order).
#[derive(Default, Clone)]
struct Counter {
    order: Vec<u32>,
    n: HashMap<u32, u32>,
}

impl Counter {
    fn add(&mut self, k: u32, by: u32) {
        let e = self.n.entry(k).or_insert(0);
        if *e == 0 && !self.order.contains(&k) {
            self.order.push(k);
        }
        *e += by;
    }
    fn get(&self, k: u32) -> u32 {
        self.n.get(&k).copied().unwrap_or(0)
    }
    /// Counter.elements(): each key as often as it counts, in insertion order.
    fn elements(&self) -> Vec<u32> {
        self.order.iter().flat_map(|&k| std::iter::repeat_n(k, self.get(k) as usize)).collect()
    }
}

pub struct FlowSolver<'a> {
    pub t: &'a mut Tracker,
    pub completion: Option<&'a dyn Completion>,
}

/// The card's original code for alternate artworks (Solver._base).
pub fn base(code: u32) -> u32 {
    match card(code) {
        Some(c) if c.alias != 0 && (c.alias as i64 - c.code as i64).abs() < 10 => c.alias,
        _ => code,
    }
}

fn is_extra(code: u32) -> bool {
    card(code).is_some_and(crate::belief::completion::is_extra_deck)
}

fn fits(code: u32, need: Need, usual: bool) -> bool {
    fits_need(card(code), need, usual)
}

impl<'a> FlowSolver<'a> {
    pub fn new(t: &'a mut Tracker, completion: Option<&'a dyn Completion>) -> Self {
        FlowSolver { t, completion }
    }

    /// Cards for slots whose requirement differs in kind from those of all the cards they can be.
    fn joint_need_codes(&self) -> BTreeSet<u32> {
        let t = &self.t;
        let mut out = BTreeSet::new();
        for v in &t.vars {
            let (Some(need), Kind::Claim, Some(pool)) = (v.need, v.kind, v.pool) else { continue };
            let needs: Vec<Option<Need>> = t.pools[pool].members.iter().map(|&m| t.vars[m].need).collect();
            if needs.contains(&None) || needs.iter().all(|n| n.unwrap().tag() == need.tag()) {
                continue;
            }
            for &code in monster_sset_codes() {
                let c = card(code);
                if c.is_some() && fits_need(c, need, false) && needs.iter().all(|n| fits_need(c, n.unwrap(), false)) {
                    out.insert(code);
                }
            }
        }
        out
    }

    pub fn solve(self) -> Result<Hypothesis> {
        let viewer = self.t.viewer;
        // (a guess which revealed card a slot is that none of them can be)
        for vi in 0..self.t.vars.len() {
            let v = &self.t.vars[vi];
            if let (Some(allowed), Some(need)) = (&v.allowed, v.need) {
                let codes: Vec<Option<u32>> =
                    allowed.iter().filter(|&&a| a < self.t.vars.len()).map(|&a| self.t.vars[a].fixed_code.filter(|&c| c != 0).or(self.t.vars[a].code)).collect();
                if !codes.is_empty() && codes.iter().all(|c| c.is_some_and(|c| c != 0 && !fits_need(card(c), need, false))) {
                    self.t.vars[vi].allowed = None;
                }
            }
        }
        let t: &Tracker = self.t;
        let mut consumer: HashMap<VarId, usize> = HashMap::new();
        for pool in &t.pools {
            for &m in &pool.members {
                consumer.insert(m, pool.index);
            }
        }
        let mut known: BTreeSet<u32> = t.vars.iter().filter_map(|v| v.known_code()).filter(|&c| c != 0).collect();
        known.extend(self.joint_need_codes());
        // cards named by hints that no card of that player is known to be: its deck contains a copy
        let own_list: HashSet<Option<u32>> = t.atoms[viewer as usize].iter().map(|&a| t.vars[a].fixed_code).collect();
        let owned: HashSet<(u32, Option<u32>)> = t.vars.iter().map(|v| (v.owner, v.known_code())).collect();
        let mut contains = Vec::new();
        for &(p, code) in &t.contains {
            let c = card(code);
            if p == viewer || c.is_none() || c.unwrap().type_ & TYPE_TOKEN != 0 || own_list.contains(&Some(code)) || owned.contains(&(p, Some(code))) {
                continue;
            }
            contains.push((p, code));
            known.insert(code);
        }
        let mut commodities: Vec<Commodity> = Vec::new();
        let mut cindex: HashMap<Commodity, usize> = HashMap::new();
        let mut com = |owner: Option<u32>, key: Key, commodities: &mut Vec<Commodity>| -> usize {
            *cindex.entry((owner, key)).or_insert_with(|| {
                commodities.push((owner, key));
                commodities.len() - 1
            })
        };
        let mut main_codes: [Vec<usize>; 2] = Default::default();
        let mut extra_codes: [Vec<usize>; 2] = Default::default();
        let mut token_codes: [Vec<usize>; 2] = Default::default();
        for &code in &known {
            let c = card(code);
            for p in 0..2u32 {
                let k = com(Some(p), Key::Code(code), &mut commodities);
                if c.is_some_and(|c| c.type_ & TYPE_TOKEN != 0) {
                    token_codes[p as usize].push(k);
                } else if is_extra(code) {
                    extra_codes[p as usize].push(k);
                } else {
                    main_codes[p as usize].push(k);
                }
            }
        }
        let opp = 1 - viewer;
        let main_wilds: Vec<usize> = WILDS.iter().filter(|&&w| w != WILD_EXTRA).map(|&w| com(Some(opp), Key::Wild(w), &mut commodities)).collect();
        let extra_wilds = vec![com(Some(opp), Key::Wild(WILD_EXTRA), &mut commodities)];

        // tags: physical cards the stream singles out (identity constraints refer to them)
        let mut refs: BTreeSet<VarId> = BTreeSet::new();
        for v in &t.vars {
            if let Some(s) = v.same_as {
                refs.insert(s);
            }
            refs.extend(v.not_same.iter().copied());
            if let Some(a) = &v.allowed {
                refs.extend(a.iter().copied().filter(|&a| a < t.vars.len()));
            }
        }
        let known_code = |v: VarId| t.vars[v].known_code().filter(|&c| c != 0);
        let mut same_code: HashMap<VarId, u32> = HashMap::new();
        for v in &t.vars {
            if let Some(s) = v.same_as
                && let Some(c) = known_code(v.id)
            {
                same_code.entry(s).or_insert(c);
            }
        }
        let mut tag_of: BTreeMap<VarId, usize> = BTreeMap::new();
        let mut tags_by_code: HashMap<u32, Vec<usize>> = HashMap::new();
        for &rid in &refs {
            let Some(code) = known_code(rid).or(same_code.get(&rid).copied()) else { continue };
            let k = com(None, Key::Tag(rid, code), &mut commodities);
            tag_of.insert(rid, k);
            tags_by_code.entry(code).or_default().push(k);
        }
        drop(com);
        let commodities = commodities;
        let tag_rid = |k: usize| match commodities[k].1 {
            Key::Tag(r, _) => r,
            _ => unreachable!(),
        };
        let mut ref_tags: HashMap<VarId, Vec<usize>> = HashMap::new();
        for (&rid, &k) in &tag_of {
            let Key::Tag(_, code) = commodities[k].1 else { unreachable!() };
            ref_tags.insert(rid, tags_by_code[&code].iter().copied().filter(|&q| tag_rid(q) <= rid).collect());
        }
        let code_of = |k: usize| code_of_key(commodities[k].1);

        // who may own the card in a slot: wildcards stand for the opponent's cards
        let mut owners: HashMap<VarId, BTreeSet<u32>> = HashMap::new();
        for v in &t.vars {
            if v.kind != Kind::Claim {
                owners.insert(v.id, BTreeSet::from([v.owner]));
            }
        }
        for pool in &t.pools {
            let ow: BTreeSet<u32> = pool.members.iter().flat_map(|m| owners.get(m).cloned().unwrap_or_default()).collect();
            for &c in &pool.claims {
                owners.insert(c, ow.clone());
            }
        }
        let tag_owner = |k: usize| -> Option<u32> {
            let r = &t.vars[tag_rid(k)];
            if let Some(o) = r.seen_owner {
                return Some(o);
            }
            let ow = owners.get(&r.id).cloned().unwrap_or_else(|| BTreeSet::from([r.owner]));
            if ow.len() == 1 { ow.iter().next().copied() } else { None }
        };
        let slot_owners = |v: VarId| -> Vec<u32> {
            let x = &t.vars[v];
            let mut ow = owners.get(&v).cloned().unwrap_or_else(|| BTreeSet::from([x.owner]));
            if let Some(s) = x.seen_owner {
                // (shown by a state view)
                ow = BTreeSet::from([s]);
            }
            ow.into_iter().collect()
        };
        let commodities_key = |k: usize| commodities[k].1;
        // (every commodity a domain can name exists by now)
        let lookup = |owner: Option<u32>, key: Key| -> usize { cindex[&(owner, key)] };
        let plain_domain = |v: VarId| -> Vec<usize> {
            let x = &t.vars[v];
            let ows = slot_owners(v);
            if let Some(code) = x.known_code().filter(|&c| c != 0) {
                return ows.iter().map(|&p| lookup(Some(p), Key::Code(code))).collect();
            }
            let opp_ok = ows.contains(&opp);
            let mains: Vec<usize> = ows.iter().flat_map(|&p| main_codes[p as usize].iter().copied()).collect();
            let extras: Vec<usize> = ows.iter().flat_map(|&p| extra_codes[p as usize].iter().copied()).collect();
            let tokens: Vec<usize> = ows.iter().flat_map(|&p| token_codes[p as usize].iter().copied()).collect();
            let pool_loc = x.pool.map(|p| t.pools[p].location);
            let mut dom: Vec<usize> = match x.kind {
                Kind::Main => [mains, main_wilds.clone()].concat(),
                Kind::Extra => [extras, extra_wilds.clone()].concat(),
                _ if matches!(pool_loc, Some(PoolLoc::Loc(l)) if l == LOCATION_DECK || l == LOCATION_HAND) => {
                    [mains, if opp_ok { main_wilds.clone() } else { vec![] }].concat()
                }
                _ if pool_loc == Some(PoolLoc::Loc(LOCATION_EXTRA)) => [extras, if opp_ok { extra_wilds.clone() } else { vec![] }].concat(),
                _ => [mains, extras, tokens, if opp_ok { [main_wilds.clone(), extra_wilds.clone()].concat() } else { vec![] }].concat(),
            };
            if let Some(need) = x.need {
                let ok_wilds = need_wilds(need);
                let wilds: Vec<usize> = dom.iter().copied().filter(|&k| matches!(commodities_key(k), Key::Wild(w) if ok_wilds.contains(&w))).collect();
                let codes: Vec<usize> = dom.iter().copied().filter(|&k| matches!(commodities_key(k), Key::Code(c) if fits(c, need, false))).collect();
                dom = [wilds, codes].concat();
            }
            if !x.excluded.is_empty() {
                dom.retain(|&k| !matches!(commodities_key(k), Key::Code(c) if x.excluded.contains(&c)));
            }
            dom
        };
        let domain = |v: VarId| -> Vec<usize> {
            if let Some(ts) = ref_tags.get(&v) {
                return ts.clone();
            }
            let dom = plain_domain(v);
            let ows = slot_owners(v);
            let mut extra: Vec<usize> = Vec::new();
            for &k in &dom {
                if let Key::Code(c) = commodities[k].1 {
                    for &q in tags_by_code.get(&c).map_or(&[][..], |v| &v[..]) {
                        if (tag_owner(q).is_none() || ows.contains(&tag_owner(q).unwrap())) && !extra.contains(&q) {
                            extra.push(q);
                        }
                    }
                }
            }
            [dom, extra].concat()
        };

        // slots in identity constraints need their own variables
        let mut linked: HashSet<VarId> = tag_of.keys().copied().collect();
        for v in &t.vars {
            if v.allowed.is_some() || v.same_as.is_some() || !v.not_same.is_empty() {
                linked.insert(v.id);
            }
            if let Some(a) = &v.allowed
                && a.len() <= MAX_UNTAGGED_ALLOWED
            {
                linked.extend(a.iter().copied().filter(|&a| a < t.vars.len()));
            }
        }
        let mut edge_keys: Vec<EdgeKey> = Vec::new();
        let mut edges: HashMap<EdgeKey, Vec<VarId>> = HashMap::new();
        for v in &t.vars {
            if v.kind == Kind::Claim && v.pool.is_none() {
                continue;
            }
            let producer = match v.kind {
                Kind::Main | Kind::Extra => Producer::Atoms(v.owner, v.kind),
                Kind::Created => Producer::Created(v.owner),
                Kind::Claim => Producer::Pool(v.pool.unwrap()),
            };
            let constrained = v.code.is_some() || v.need.is_some() || !v.excluded.is_empty() || v.fixed_code.is_some() || v.kind == Kind::Created || linked.contains(&v.id);
            // (slots grouped only with slots of the same seen owner: their domains differ)
            let key = EdgeKey {
                producer,
                consumer: consumer.get(&v.id).copied(),
                var: constrained.then_some(v.id),
                owner: v.owner,
                group: (!constrained).then(|| (v.kind, v.pool.map(|p| t.pools[p].location), v.seen_owner)),
            };
            let e = edges.entry(key).or_insert_with(|| {
                edge_keys.push(key);
                Vec::new()
            });
            e.push(v.id);
        }

        // variables x[e, k]
        let mut fixed_counts: Vec<(usize, Counter)> = Vec::new();
        let (mut cols, mut lo, mut hi, mut cost): (Vec<(usize, usize)>, Vec<f64>, Vec<f64>, Vec<f64>) = Default::default();
        for (ei, key) in edge_keys.iter().enumerate() {
            let vs = &edges[key];
            let mut dom = domain(vs[0]);
            if dom.is_empty() {
                return err(format!("no possible card for {}", t.vars[vs[0]].describe()));
            }
            let n = vs.len() as f64;
            let viewer_atoms = matches!(key.producer, Producer::Atoms(o, _) if o == viewer);
            if viewer_atoms {
                let mut codes = Counter::default();
                for &v in vs {
                    codes.add(t.vars[v].fixed_code.unwrap_or(0), 1);
                }
                dom.retain(|&k| matches!(code_of(k), CodeOf::Code(c) if codes.get(c) > 0));
                fixed_counts.push((ei, codes));
            }
            let v0 = &t.vars[vs[0]];
            let holder = if v0.kind == Kind::Claim && v0.pool.is_some() { t.pools[v0.pool.unwrap()].player } else { v0.owner };
            let opp_atom = matches!(key.producer, Producer::Atoms(o, _) if o != viewer);
            for k in dom {
                cols.push((ei, k));
                lo.push(0.0);
                hi.push(n);
                let what = commodities[k].1;
                let is_str = matches!(what, Key::Wild(_));
                let c1 = if !opp_atom {
                    0.0
                } else if !is_str {
                    1.0
                } else if matches!(what, Key::Wild(w) if w == WILD_MAIN || w == WILD_EXTRA) {
                    0.0
                } else {
                    0.01
                };
                // (a known card whose Level needs fewer Tributes than usual: only if nothing else fits)
                // (a tag's key is no card code: it never fits, as in the Python model)
                let c2 = match (v0.need, what) {
                    (Some(need), Key::Code(c)) if !fits(c, need, true) => 2.0,
                    (Some(_), Key::Tag(..)) => 2.0,
                    _ => 0.0,
                };
                // (a card a player holds is their own unless something shows otherwise)
                let c3 = if commodities[k].0.is_some_and(|o| o != holder) { 0.05 } else { 0.0 };
                cost.push(c1 + c2 + c3);
            }
        }
        let colmap: HashMap<(usize, usize), usize> = cols.iter().enumerate().map(|(j, &c)| (c, j)).collect();
        let ncol = cols.len();
        let mut by_edge: Vec<Vec<usize>> = vec![Vec::new(); edge_keys.len()];
        for (j, &(ei, _)) in cols.iter().enumerate() {
            by_edge[ei].push(j);
        }
        let mut rows: Vec<(Vec<(usize, f64)>, f64, f64)> = Vec::new();
        // capacity of every edge
        for (ei, key) in edge_keys.iter().enumerate() {
            let n = edges[key].len() as f64;
            rows.push((by_edge[ei].iter().map(|&j| (j, 1.0)).collect(), n, n));
        }
        // conservation per pool and commodity
        let mut flow_order: Vec<(usize, usize)> = Vec::new();
        let mut flow: HashMap<(usize, usize), Vec<(usize, f64)>> = HashMap::new();
        let mut add_flow = |key: (usize, usize), term: (usize, f64), flow: &mut HashMap<(usize, usize), Vec<(usize, f64)>>| {
            flow.entry(key)
                .or_insert_with(|| {
                    flow_order.push(key);
                    Vec::new()
                })
                .push(term);
        };
        for (j, &(ei, k)) in cols.iter().enumerate() {
            let key = &edge_keys[ei];
            if let Some(c) = key.consumer {
                add_flow((c, k), (j, 1.0), &mut flow);
            }
            if let Producer::Pool(p) = key.producer {
                add_flow((p, k), (j, -1.0), &mut flow);
            }
        }
        for key in &flow_order {
            rows.push((flow[key].clone(), 0.0, 0.0));
        }
        // the viewer's own deck list: exactly its cards (tagged or not)
        for (ei, codes) in &fixed_counts {
            let mut per: HashMap<u32, Vec<usize>> = HashMap::new();
            for &j in &by_edge[*ei] {
                if let CodeOf::Code(c) = code_of(cols[j].1) {
                    per.entry(c).or_default().push(j);
                }
            }
            for &code in &codes.order {
                let n = codes.get(code) as f64;
                rows.push((per.get(&code).map_or(vec![], |js| js.iter().map(|&j| (j, 1.0)).collect()), n, n));
            }
        }
        // the deck of the player a hint named a card for contains it
        for &(p, code) in &contains {
            let terms = cols
                .iter()
                .enumerate()
                .filter(|&(_, &(ei, k))| matches!(edge_keys[ei].producer, Producer::Atoms(o, _) if o == p) && code_of(k) == CodeOf::Code(code))
                .map(|(j, _)| (j, 1.0))
                .collect();
            rows.push((terms, 1.0, 3.0));
        }
        // at most 3 copies of a card per opponent deck (by base code, tags included)
        let mut limit_order: Vec<u32> = Vec::new();
        let mut limit: HashMap<u32, Vec<usize>> = HashMap::new();
        for (j, &(ei, k)) in cols.iter().enumerate() {
            if let Producer::Atoms(o, _) = edge_keys[ei].producer
                && o != viewer
                && let CodeOf::Code(c) = code_of(k)
            {
                let b = base(c);
                limit
                    .entry(b)
                    .or_insert_with(|| {
                        limit_order.push(b);
                        Vec::new()
                    })
                    .push(j);
            }
        }
        for b in &limit_order {
            rows.push((limit[b].iter().map(|&j| (j, 1.0)).collect(), 0.0, 3.0));
        }
        // a tag exists (is supplied by one card of a deck list) iff its ref carries it
        let mut edge_of: HashMap<VarId, usize> = HashMap::new();
        for (ei, key) in edge_keys.iter().enumerate() {
            if let Some(v) = key.var {
                edge_of.insert(v, ei);
            }
        }
        for (&rid, &k) in &tag_of {
            let supply: Vec<(usize, f64)> =
                cols.iter().enumerate().filter(|&(_, &(ei, kk))| kk == k && matches!(edge_keys[ei].producer, Producer::Atoms(..))).map(|(j, _)| (j, 1.0)).collect();
            let Some(&e) = edge_of.get(&rid) else { return err(format!("no edge for tagged slot {rid}")) };
            let Some(&jr) = colmap.get(&(e, k)) else { continue };
            rows.push(([supply, vec![(jr, -1.0)]].concat(), 0.0, 0.0));
        }
        let var_cols = |v: VarId| -> Vec<(usize, usize)> {
            match edge_of.get(&v) {
                None => vec![],
                Some(&ei) => by_edge[ei].iter().map(|&j| (cols[j].1, j)).collect(),
            }
        };
        let get = |m: &[(usize, usize)], k: usize| m.iter().find(|x| x.0 == k).map(|x| x.1);
        let ninf = f64::NEG_INFINITY;
        for v in &t.vars {
            if !linked.contains(&v.id) {
                continue;
            }
            let mine = var_cols(v.id);
            if let Some(s) = v.same_as
                && let Some(ts) = ref_tags.get(&s)
            {
                // the same card: the same tag
                let other = var_cols(s);
                for &k in ts {
                    let mut terms = get(&other, k).map_or(vec![], |j| vec![(j, -1.0)]);
                    if let Some(j) = get(&mine, k) {
                        terms.push((j, 1.0));
                    }
                    rows.push((terms, 0.0, 0.0));
                }
            }
            for &x in &v.not_same {
                let Some(ts) = ref_tags.get(&x) else { continue };
                let other = var_cols(x);
                for &k in ts {
                    if let (Some(a), Some(b)) = (get(&mine, k), get(&other, k)) {
                        rows.push((vec![(a, 1.0), (b, 1.0)], ninf, 1.0));
                    }
                }
            }
            if let Some(allowed) = &v.allowed {
                if !allowed.iter().all(|a| ref_tags.contains_key(a)) && allowed.len() <= MAX_UNTAGGED_ALLOWED {
                    // (members without a known code) the slot's card is one of them: whatever it
                    // carries, one of them carries too
                    let ms: Vec<VarId> = allowed.iter().copied().filter(|&a| a < t.vars.len()).collect();
                    for &(k, j) in &mine {
                        let mut terms = vec![(j, 1.0)];
                        terms.extend(ms.iter().filter_map(|&m| get(&var_cols(m), k)).map(|j| (j, -1.0)));
                        rows.push((terms, ninf, 0.0));
                    }
                }
                let ms: Vec<VarId> = allowed.iter().copied().filter(|a| ref_tags.contains_key(a)).collect();
                if !ms.is_empty() && ms.len() == allowed.len() {
                    let ts: BTreeSet<usize> = ms.iter().flat_map(|m| ref_tags[m].iter().copied()).collect();
                    for &k in &ts {
                        let mut terms: Vec<(usize, f64)> = ms.iter().filter_map(|&m| get(&var_cols(m), k)).map(|j| (j, -1.0)).collect();
                        if let Some(j) = get(&mine, k) {
                            terms.push((j, 1.0));
                        }
                        rows.push((terms, ninf, 0.0));
                    }
                    rows.push((ts.iter().filter_map(|&k| get(&mine, k)).map(|j| (j, 1.0)).collect(), 1.0, 1.0));
                }
            }
        }
        // (the CSC matrix scipy makes of the rows: duplicates summed, rows ascending per column)
        let mut by_col: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); ncol];
        for (r, (terms, _, _)) in rows.iter().enumerate() {
            for &(j, val) in terms {
                *by_col[j].entry(r).or_insert(0.0) += val;
            }
        }
        let mut mip = highs::Mip {
            cost,
            col_lower: lo,
            col_upper: hi,
            row_lower: rows.iter().map(|r| r.1).collect(),
            row_upper: rows.iter().map(|r| r.2).collect(),
            start: vec![0],
            index: vec![],
            value: vec![],
        };
        for col in &by_col {
            for (&r, &val) in col {
                mip.index.push(r as i32);
                mip.value.push(val);
            }
            mip.start.push(mip.index.len() as i32);
        }
        let x = match highs::solve(&mip, 600.0, 1) {
            Ok(x) => x,
            Err(status) => return err(format!("observations are inconsistent (flow model: HiGHS model status {status})")),
        };
        let x: Vec<i64> = x.iter().map(|v| v.round_ties_even() as i64).collect();
        // commodity of every slot
        let mut vcom: HashMap<VarId, usize> = HashMap::new();
        for (ei, key) in edge_keys.iter().enumerate() {
            let mut it = edges[key].iter();
            for &j in &by_edge[ei] {
                for _ in 0..x[j].max(0) {
                    if let Some(&v) = it.next() {
                        vcom.insert(v, cols[j].1);
                    }
                }
            }
        }
        self.realize(vcom, commodities)
    }

    fn descends(&self, mut v: Option<VarId>, ancestor: VarId) -> bool {
        while let Some(x) = v {
            if x == ancestor {
                return true;
            }
            v = if self.t.vars[x].kind == Kind::Claim { self.t.vars[x].assigned } else { None };
        }
        false
    }

    fn realize(self, vcom: HashMap<VarId, usize>, commodities: Vec<Commodity>) -> Result<Hypothesis> {
        // 1. every pool: claims get members of the same commodity (hint identity first)
        for pi in 0..self.t.pools.len() {
            let mut free: HashMap<usize, Vec<VarId>> = HashMap::new();
            for &m in &self.t.pools[pi].members {
                free.entry(vcom[&m]).or_default().push(m);
            }
            let mut claims = self.t.pools[pi].claims.clone();
            claims.sort_by_key(|&c| {
                let x = &self.t.vars[c];
                (x.same_as.is_none(), x.allowed.is_none(), x.not_same.is_empty(), x.prefer.is_empty())
            });
            for c in claims {
                let available = free.get(&vcom[&c]).cloned().unwrap_or_default();
                let x = self.t.vars[c].clone();
                let mut cands = available.clone();
                if !x.prefer.is_empty() {
                    // prefer members that are one of these physical cards
                    let ok: Vec<VarId> = available.iter().copied().filter(|&m| x.prefer.iter().any(|&p| self.descends(Some(m), p))).collect();
                    cands = [ok.clone(), available.iter().copied().filter(|m| !ok.contains(m)).collect()].concat();
                }
                if !x.not_same.is_empty() {
                    // prefer members that are none of these physical cards
                    let ok: Vec<VarId> = available.iter().copied().filter(|&m| !x.not_same.iter().any(|&p| self.descends(Some(m), p))).collect();
                    cands = [ok.clone(), available.iter().copied().filter(|m| !ok.contains(m)).collect()].concat();
                }
                let allows = |m: VarId| -> bool {
                    match &x.allowed {
                        None => true,
                        Some(a) => a.contains(&m) || a.iter().any(|&a| a < self.t.vars.len() && self.descends(Some(m), a)),
                    }
                };
                let pick = match x.same_as {
                    Some(s) => cands.iter().copied().find(|&m| self.descends(Some(m), s)).or_else(|| cands.iter().copied().find(|&m| allows(m))),
                    None => cands.iter().copied().find(|&m| allows(m)),
                };
                let Some(pick) = pick else { return err(format!("no member for {} in pool {pi}", x.describe())) };
                let avail = free.get_mut(&vcom[&c]).unwrap();
                let pos = avail.iter().position(|&m| m == pick).unwrap();
                avail.remove(pos);
                self.t.vars[c].assigned = Some(pick);
            }
        }

        // 2. codes of atoms: known ones directly, wildcards from the completion
        let vanilla: &dyn Completion = VanillaCompletion::get();
        let comp: &dyn Completion = self.completion.unwrap_or(vanilla);
        let mut unknown = 0;
        // codes excluded for a slot are excluded for the card it realizes to
        let mut excluded: HashMap<VarId, BTreeSet<u32>> = HashMap::new();
        for &v in &self.t.exclusions {
            let mut a = Some(v);
            while let Some(x) = a.filter(|&x| self.t.vars[x].kind == Kind::Claim) {
                a = self.t.vars[x].assigned;
            }
            if let Some(a) = a {
                excluded.entry(a).or_default().extend(self.t.vars[v].excluded.iter().copied());
            }
        }
        let key_code = |k: usize| -> Option<u32> {
            match commodities[k].1 {
                Key::Code(c) | Key::Tag(_, c) => Some(c),
                Key::Wild(_) => None,
            }
        };
        for p in 0..2 {
            for &a in &self.t.atoms[p].clone() {
                let v = &self.t.vars[a];
                self.t.vars[a].resolved = v.fixed_code.or_else(|| key_code(vcom[&a]));
            }
        }
        let present: Vec<u32> = (0..2).flat_map(|p| self.t.atoms[p].iter().filter_map(|&a| self.t.vars[a].resolved).collect::<Vec<_>>()).collect();
        for p in 0..2 {
            let atoms = self.t.atoms[p].clone();
            let wild: Vec<VarId> = atoms.iter().copied().filter(|&a| self.t.vars[a].resolved.is_none()).collect();
            if wild.is_empty() {
                continue;
            }
            unknown += wild.len();
            let mut counts = Counter::default();
            for &a in &atoms {
                if let Some(r) = self.t.vars[a].resolved {
                    counts.add(base(r), 1);
                }
            }
            let wild_name = |a: VarId| match commodities[vcom[&a]].1 {
                Key::Wild(w) => w,
                _ => unreachable!(),
            };
            let typed: Vec<VarId> = wild.iter().copied().filter(|&a| !matches!(wild_name(a), WILD_MAIN | WILD_EXTRA)).collect();
            for a in typed {
                let need = wild_need(wild_name(a)).unwrap();
                let ex = excluded.get(&a).cloned().unwrap_or_default();
                let mut cands: Vec<u32> = comp.stand_ins(need, &counts.elements()).into_iter().filter(|&c| counts.get(base(c)) < 3 && !ex.contains(&c)).collect();
                if matches!(need, Need::Mset(_)) && comp.stand_ins(need, &[]).iter().any(|c| ex.contains(c)) {
                    // (once a stand-in was ruled out: the races of the player's known monsters first)
                    let mut races: HashMap<u64, i64> = HashMap::new();
                    for &r in &counts.order {
                        if counts.get(r) > 0
                            && let Some(c) = card(r).filter(|c| c.type_ & TYPE_MONSTER != 0)
                        {
                            *races.entry(c.race).or_default() += 1;
                        }
                    }
                    cands.sort_by_key(|&c| card(c).map_or(0, |d| -races.get(&d.race).copied().unwrap_or(0)));
                }
                if cands.is_empty() {
                    return err(format!("no stand-in for a hidden card of kind ('{}',)", need.tag()));
                }
                // (different stand-ins first: checks can count different names)
                cands.sort_by_key(|&c| counts.get(base(c)) > 0);
                self.t.vars[a].resolved = Some(cands[0]);
                counts.add(base(cands[0]), 1);
            }
            let plain: Vec<VarId> = wild.iter().copied().filter(|&a| self.t.vars[a].resolved.is_none()).collect();
            if !plain.is_empty() {
                let res = |k: Kind| atoms.iter().filter(|&&a| self.t.vars[a].kind == k).filter_map(|&a| self.t.vars[a].resolved).collect::<Vec<_>>();
                let (known_main, known_extra) = (res(Kind::Main), res(Kind::Extra));
                let n_of = |k: Kind| plain.iter().filter(|&&a| self.t.vars[a].kind == k).count();
                let (n_main, n_extra) = (n_of(Kind::Main), n_of(Kind::Extra));
                let Some(completion) = self.completion else { return err("unknown opponent cards but no deck completion available") };
                // (plus a spare code per slot with ruled out codes)
                let spare = |k: Kind| plain.iter().filter(|&&a| self.t.vars[a].kind == k && excluded.get(&a).is_some_and(|e| !e.is_empty())).count();
                let (fill_main, fill_extra) = match completion.complete(&known_main, &known_extra, n_main + spare(Kind::Main), n_extra + spare(Kind::Extra), &present) {
                    Ok(f) => f,
                    Err(_) => completion.complete(&known_main, &known_extra, n_main, n_extra, &present).map_err(TrackerError)?,
                };
                let (mut fm, mut fe) = (fill_main, fill_extra);
                let mut order = plain.clone();
                order.sort_by_key(|a| excluded.get(a).is_none_or(|e| e.is_empty()));
                for a in order {
                    let lst = if self.t.vars[a].kind == Kind::Main { &mut fm } else { &mut fe };
                    let ex = excluded.get(&a).cloned().unwrap_or_default();
                    if lst.is_empty() {
                        return err("the completion gave too few cards");
                    }
                    let k = if !ex.contains(&lst[0]) { 0 } else { lst.iter().position(|c| !ex.contains(c)).unwrap_or(0) };
                    self.t.vars[a].resolved = Some(lst.remove(k));
                }
            }
        }
        for vi in 0..self.t.vars.len() {
            let v = &self.t.vars[vi];
            if v.kind == Kind::Created && v.resolved.is_none() {
                let key = vcom.get(&vi).and_then(|&k| key_code(k));
                self.t.vars[vi].resolved = v.code.filter(|&c| c != 0).or(key);
            }
        }
        // 3. the rest as in the search based solver (orders, shuffles, copy ranks)
        finish(self.t, unknown)
    }
}

/// The value of a var if already determined (Solver._known).
pub fn known(t: &Tracker, mut v: VarId) -> Option<u32> {
    loop {
        let x = &t.vars[v];
        if let Some(c) = x.fixed_code {
            return Some(c);
        }
        if let Some(c) = x.resolved {
            return Some(c);
        }
        match x.assigned {
            None => return x.code,
            Some(a) => v = a,
        }
    }
}

fn initial_index(t: &Tracker, player: u32, location: u32) -> Option<usize> {
    t.initial_pools.iter().position(|&p| t.pools[p].player == player && t.pools[p].location == PoolLoc::Loc(location))
}

/// Orders, shuffles and copy ranks from a complete assignment (Solver.finish).
pub fn finish(t: &Tracker, unknown: usize) -> Result<Hypothesis> {
    let final_ = |v: VarId| -> Result<u32> { known(t, v).ok_or_else(|| TrackerError(format!("unresolved {}", t.vars[v].describe()))) };
    let atom = |mut v: Option<VarId>| -> Option<VarId> {
        while let Some(x) = v {
            if t.vars[x].kind != Kind::Claim {
                break;
            }
            v = t.vars[x].assigned;
        }
        v.filter(|&x| t.vars[x].kind != Kind::Created)
    };
    let mut ranks: HashMap<VarId, u32> = HashMap::new();
    for p in 0..2 {
        let mut seen: HashMap<(Kind, u32), u32> = HashMap::new();
        for &a in &t.atoms[p] {
            let k = (t.vars[a].kind, final_(a)?);
            let r = seen.get(&k).copied().unwrap_or(0);
            ranks.insert(a, r);
            seen.insert(k, r + 1);
        }
    }
    let rank_list = |vs: &[VarId]| -> Option<Vec<u32>> { vs.iter().map(|&v| atom(Some(v)).map(|a| ranks[&a])).collect() };
    let owner_list = |vs: &[VarId]| -> Option<Vec<u32>> { vs.iter().map(|&v| atom(Some(v)).map(|a| t.vars[a].owner)).collect() };
    let mut h = Hypothesis { unknown_atoms: unknown, ..Hypothesis::default() };
    for p in 0..2u32 {
        let di = initial_index(t, p, LOCATION_DECK).ok_or_else(|| TrackerError("no initial deck".into()))?;
        h.deck_ranks.push(rank_list(&t.pools[t.initial_pools[di]].claims));
        let extra_atoms: Vec<VarId> = t.atoms[p as usize].iter().copied().filter(|&a| t.vars[a].kind == Kind::Extra).collect();
        h.extra_ranks.push(match initial_index(t, p, LOCATION_EXTRA) {
            None => rank_list(&extra_atoms),
            Some(ei) => rank_list(&t.pools[t.initial_pools[ei]].claims),
        });
    }
    for p in 0..2u32 {
        let of_kind = |k: Kind| -> Result<Vec<u32>> { t.atoms[p as usize].iter().filter(|&&a| t.vars[a].kind == k).map(|&a| final_(a)).collect() };
        let (main, extra) = (of_kind(Kind::Main)?, of_kind(Kind::Extra)?);
        let di = initial_index(t, p, LOCATION_DECK).unwrap();
        h.deck_orders.push(t.pools[t.initial_pools[di]].claims.iter().map(|&c| final_(c)).collect::<Result<_>>()?);
        h.extra_orders.push(match initial_index(t, p, LOCATION_EXTRA) {
            None => extra.clone(), // (the viewer's own extra deck, in its order)
            Some(ei) => t.pools[t.initial_pools[ei]].claims.iter().map(|&c| final_(c)).collect::<Result<_>>()?,
        });
        h.decks.push((main, extra));
    }
    for &pi in &t.shuffle_events {
        let pool = &t.pools[pi];
        h.shuffles.push(ShuffleOutcome {
            player: pool.player,
            location: pool.location,
            codes: pool.claims.iter().map(|&c| final_(c)).collect::<Result<_>>()?,
            ranks: rank_list(&pool.claims),
            owners: owner_list(&pool.claims),
        });
    }
    for &pi in &t.set_shuffle_events {
        let pool = &t.pools[pi];
        h.set_card_shuffles.push(pool.claims.iter().map(|&c| final_(c)).collect::<Result<_>>()?);
        h.set_card_shuffle_ranks.push((rank_list(&pool.claims), owner_list(&pool.claims)));
    }
    for &v in &t.exclusions {
        let f = final_(v)?;
        if t.vars[v].excluded.contains(&f) {
            return err(format!("{} is {f}, which was excluded", t.vars[v].describe()));
        }
    }
    h.var_codes = t.vars.iter().map(|v| (v.id, known(t, v.id))).collect();
    h.observed = t.observations.iter().map(|&(v, _)| v).collect();
    Ok(h)
}

impl Tracker {
    /// The hypothesis (belief/tracker.py Tracker.solve: the flow solver).
    pub fn solve(&mut self, completion: Option<&dyn Completion>) -> Result<Hypothesis> {
        FlowSolver::new(self, completion).solve()
    }
}
