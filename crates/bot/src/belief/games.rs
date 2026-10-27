//! The belief checks' test games: a game between two built-in bots, and both players'
//! observation streams recorded while it is played.

use crate::belief_bots::make_bot;
use ygo_arbiter::game::{Deck, DuelConfig, GameError, GameRecord, ObsItem, ObservationLog, PlayOptions, new_record, play};
use ygo_arbiter::messages::Msg;

/// The test game `seed` of the decks and the bots named (seeded 2 seed and 2 seed + 1), and each
/// player's observation stream with the state views, up to the WIN message (a replay stops there;
/// the game itself goes on to the end of its step).
pub fn play_with_streams(seed: i64, decks: [Deck; 2], bots: [&str; 2]) -> Result<(GameRecord, [Vec<ObsItem>; 2]), GameError> {
    let mut record = new_record(&decks, seed, &DuelConfig::default());
    let bot = |p: usize| make_bot(bots[p], Some(2 * seed + p as i64)).map_err(GameError);
    let (mut a, mut b) = (bot(0)?, bot(1)?);
    let mut logs = [ObservationLog::new(0, true), ObservationLog::new(1, true)];
    let [log0, log1] = &mut logs;
    play(&mut record, &mut [a.as_mut(), b.as_mut()], &mut [log0, log1], &mut PlayOptions::default())?;
    let streams = logs.map(|log| {
        let mut items = log.items;
        if let Some(win) = items.iter().position(|it| matches!(it, ObsItem::Msg(Msg::Win { .. }))) {
            items.truncate(win + 1);
        }
        items
    });
    Ok((record, streams))
}
