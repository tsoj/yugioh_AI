//! The columns of a decision's rows (bot::policy_features, scripts/policy/data.py).

// -- header columns (i64) ---------------------------------------------------------------------
pub const H_GAME: usize = 0;
pub const H_DECISION: usize = 1;
pub const H_PLAYER: usize = 2;
pub const H_TURN: usize = 3;
/// 0: the player's own turn
pub const H_TURN_PLAYER: usize = 4;
pub const H_PHASE: usize = 5;
pub const H_LP_ME: usize = 6;
pub const H_LP_OPP: usize = 7;
pub const H_PROMPT: usize = 8;
pub const H_MIN: usize = 9;
pub const H_MAX: usize = 10;
/// bit 0 forced (a chain), bit 1 cancelable, bit 2 finishable
pub const H_FLAGS: usize = 11;
/// the last selection hint (HINT_SELECTMSG) to the player: a card's string (code index, string
/// number) or a system string (0, its number)
pub const H_HINT_CODE: usize = 12;
pub const H_HINT_STR: usize = 13;
/// a yes/no prompt's question, the same way
pub const H_DESC_CODE: usize = 14;
pub const H_DESC_STR: usize = 15;
/// the search's depth and value (-1 and 0: not searched, the rules answered)
pub const H_DEPTH: usize = 16;
pub const H_VALUE: usize = 17;
/// for the player: 1 win, 0 draw or unfinished, -1 loss
pub const H_RESULT: usize = 18;
pub const H_TOK_START: usize = 19;
pub const H_N_TOK: usize = 20;
pub const H_OPT_START: usize = 21;
pub const H_N_OPT: usize = 22;
/// the player's previous chosen option (its kind, -1 none, and its card's code index)
pub const H_PREV_KIND: usize = 23;
pub const H_PREV_CODE: usize = 24;
pub const H_DECK_ME: usize = 25;
pub const H_DECK_OPP: usize = 26;
pub const H_EXTRA_OPP: usize = 27;
/// SELECT_POSITION's card (code index)
pub const H_CODE: usize = 28;
pub const HDR: usize = 29;

// -- token columns (i16) ----------------------------------------------------------------------
pub const T_CODE: usize = 0;
/// 0 the player, 1 the opponent
pub const T_CON: usize = 1;
/// 0 the player, 1 the opponent, 2 unknown
pub const T_OWNER: usize = 2;
pub const T_LOC: usize = 3;
pub const T_SEQ: usize = 4;
/// bit 0 face-up, bit 1 attack position, bit 2 public (revealed)
pub const T_POS: usize = 5;
/// -1: unknown or '?'
pub const T_ATK: usize = 6;
pub const T_DEF: usize = 7;
pub const T_BASE_ATK: usize = 8;
pub const T_BASE_DEF: usize = 9;
/// level, rank or link rating
pub const T_LEVEL: usize = 10;
pub const T_SCALE: usize = 11;
pub const T_COUNTERS: usize = 12;
pub const T_OVERLAYS: usize = 13;
/// another token of the decision: an Xyz material's monster, an Equip Card's monster, a chain
/// link's card (-1 none)
pub const T_REF: usize = 14;
/// `STATUS_BITS`' bits
pub const T_STATUS: usize = 15;
/// a deck summary's copies; a chain link's number (1 first)
pub const T_COUNT: usize = 16;
pub const TOK: usize = 17;

// -- option columns (i32) ---------------------------------------------------------------------
pub const O_KIND: usize = 0;
/// the option's card (a token, -1 none)
pub const O_TOKEN: usize = 1;
/// a zone (player 0/1 * 16 + 8 for the Spell/Trap Zones + sequence), a position, a number, an
/// attribute or type bit, a sum's value
pub const O_AUX: usize = 2;
/// the option's text (an activation's effect, an option): code index and string number, or 0
/// and a system string
pub const O_DESC_CODE: usize = 3;
pub const O_DESC_STR: usize = 4;
/// 1: in the player's response
pub const O_CHOSEN: usize = 5;
pub const OPT: usize = 6;
