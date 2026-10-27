//! Games for Python: a new game's record, what a player knows before the duel, and replays of
//! recorded games (both players' observation streams, or the god view of what happened).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ygo_core::duel::Duel;
use ygoai::actions::Response;
use ygoai::game::{Deck, DuelConfig, GameError, GameRecord, ObservationLog, Observer, replay};
use ygoai::messages::Msg;
use ygoai::observe::to_core_response;
use ygoai::state::{Snapshot, capture};

use crate::convert::{de, py, py_items};

fn err(e: GameError) -> PyErr {
    PyValueError::new_err(e.0)
}

/// A new game's record of the decks (game::new_record): the core's seed and both deck orders from
/// `seed`.
#[pyfunction]
fn new_record<'py>(p: Python<'py>, decks: &Bound<'py, PyAny>, seed: i64) -> PyResult<Bound<'py, PyAny>> {
    let decks: Vec<Deck> = de(decks)?;
    py(p, &ygoai::game::new_record(&decks, seed, &DuelConfig::default()))
}

/// What `player` knows before the duel of `record` starts (game::init_info).
#[pyfunction]
fn init_info<'py>(p: Python<'py>, record: &Bound<'py, PyAny>, player: u32) -> PyResult<Bound<'py, PyAny>> {
    let record: GameRecord = de(record)?;
    let duel = record.setup(None).map_err(err)?;
    py(p, &ygoai::game::init_info(&record, &duel, player))
}

/// Both players' observation streams of a recorded game, replayed with the card creation order
/// of `creation_seed` (None: the game's own), and the error the replay stopped at (None: none;
/// the streams then end there).
#[pyfunction]
#[pyo3(signature = (record, creation_seed=None))]
fn replay_streams<'py>(p: Python<'py>, record: &Bound<'py, PyAny>, creation_seed: Option<i128>) -> PyResult<(Vec<Vec<Bound<'py, PyAny>>>, Option<String>)> {
    let record: GameRecord = de(record)?;
    let mut logs = [ObservationLog::new(0, true), ObservationLog::new(1, true)];
    let [log0, log1] = &mut logs;
    let error = p.detach(|| replay(&record, &mut [log0, log1], None, creation_seed).err()).map(|e| e.0);
    Ok((logs.iter().map(|log| py_items(p, &log.items)).collect::<PyResult<_>>()?, error))
}

/// One event of a replay's god view.
enum Event {
    Msg(Msg),
    /// with the full state (`state::capture`), if asked for
    Prompt(Msg, Option<Snapshot>),
    /// the player's response and the core's version of it
    Response(u32, Response, Option<Response>),
}

/// Records a replay's events, and before each the length of a viewer's observation stream.
struct Events {
    snapshots: bool,
    events: Vec<Event>,
    viewer: Option<ObservationLog>,
    positions: Vec<usize>,
    prompt: Option<Msg>,
}

impl Events {
    fn mark(&mut self) {
        if let Some(log) = &self.viewer {
            self.positions.push(log.items.len());
        }
    }
}

impl Observer for Events {
    fn message(&mut self, msg: &Msg) {
        self.mark();
        self.events.push(Event::Msg(msg.clone()));
        if let Some(log) = &mut self.viewer {
            log.message(msg);
        }
    }
    fn prompt(&mut self, prompt: &Msg, duel: &mut Duel) {
        self.mark();
        // (the queries on a copy: the game goes on exactly as it was played)
        let snapshot = self.snapshots.then(|| capture(&mut duel.clone()));
        self.events.push(Event::Prompt(prompt.clone(), snapshot));
        if let Some(log) = &mut self.viewer {
            log.prompt(prompt, duel);
        }
        self.prompt = Some(prompt.clone());
    }
    fn response(&mut self, player: u32, response: &Response) {
        self.mark();
        let core = self.prompt.as_ref().and_then(|p| to_core_response(p, player, response).ok());
        self.events.push(Event::Response(player, response.clone(), core));
        if let Some(log) = &mut self.viewer {
            log.response(player, response);
        }
    }
}

/// The god view of a recorded game: its events ("msg", msg) / ("prompt", prompt, the full state
/// if `snapshots`, else None) / ("response", player, response, the core's response), and with a
/// `viewer` the length of its observation stream before each event (else None).
#[pyfunction]
#[pyo3(signature = (record, viewer=None, snapshots=false))]
fn replay_events<'py>(p: Python<'py>, record: &Bound<'py, PyAny>, viewer: Option<u32>, snapshots: bool) -> PyResult<(Vec<Bound<'py, PyAny>>, Option<Vec<usize>>)> {
    let record: GameRecord = de(record)?;
    let mut events = Events { snapshots, events: Vec::new(), viewer: viewer.map(|v| ObservationLog::new(v, false)), positions: Vec::new(), prompt: None };
    p.detach(|| replay(&record, &mut [&mut events], None, None).map(drop)).map_err(err)?;
    let items = events
        .events
        .iter()
        .map(|e| {
            Ok(match e {
                Event::Msg(m) => ("msg", py(p, m)?).into_pyobject(p)?.into_any(),
                Event::Prompt(m, s) => ("prompt", py(p, m)?, py(p, s)?).into_pyobject(p)?.into_any(),
                Event::Response(player, r, core) => ("response", player, py(p, r)?, py(p, core)?).into_pyobject(p)?.into_any(),
            })
        })
        .collect::<PyResult<_>>()?;
    Ok((items, viewer.map(|_| events.positions)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(new_record, m)?)?;
    m.add_function(wrap_pyfunction!(init_info, m)?)?;
    m.add_function(wrap_pyfunction!(replay_streams, m)?)?;
    m.add_function(wrap_pyfunction!(replay_events, m)?)?;
    Ok(())
}
