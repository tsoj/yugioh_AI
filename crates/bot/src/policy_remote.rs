//! The imitation bot's scores from another process (scripts/policy/rl.py, on the GPU): game
//! threads queue their decisions; a server thread sends them in batches and hands back the scores.
//!
//! Over a byte stream, little-endian. A batch: u32 1, u32 n, then columns: per decision u32 id,
//! u32 net, u32 tokens, u32 options, u32 codes, u32 strings; the hdr rows (i64 x HDR), token rows
//! (i16 x TOK), option rows (i32 x OPT), code rows (i32: the game's code indices' rows of the
//! tables, -1 none) and string rows (i32 x 3: code index, string, its row of the tables), each
//! decision's after the previous one's. The answer: u32 n, the ids (u32), the scores' counts
//! (u32), the scores (f32; train.Batcher's options, those beyond it -inf). The end: u32 2, u32
//! length, a JSON summary.
//!
//! `Client`: a bot process's decisions, one at a time, as batches of one over a Unix socket
//! (scripts/policy/serve.py).

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::policy_features::*;

/// A decision as the bot sees it, for network `net`, with its game's text rows.
pub struct Request {
    pub net: u32,
    pub row: Row,
    pub code_rows: Vec<i32>,
    pub string_rows: Vec<[i32; 3]>,
}

type Slot = Arc<(Mutex<Option<Vec<f32>>>, Condvar)>;

struct Queue {
    waiting: Vec<(Request, Slot)>,
    /// game threads still playing
    workers: usize,
    /// since when a decision waits
    since: Option<Instant>,
}

pub struct Remote {
    queue: Mutex<Queue>,
    changed: Condvar,
}

impl Remote {
    pub fn new(workers: usize) -> Remote {
        Remote { queue: Mutex::new(Queue { waiting: Vec::new(), workers, since: None }), changed: Condvar::new() }
    }

    /// The request's scores (blocks until the server has them).
    pub fn score(&self, request: Request) -> Vec<f32> {
        let slot: Slot = Arc::new((Mutex::new(None), Condvar::new()));
        {
            let mut q = self.queue.lock().unwrap();
            q.since.get_or_insert_with(Instant::now);
            q.waiting.push((request, slot.clone()));
        }
        self.changed.notify_one();
        let (answer, ready) = &*slot;
        let mut a = answer.lock().unwrap();
        loop {
            if let Some(s) = a.take() {
                return s;
            }
            a = ready.wait(a).unwrap();
        }
    }

    /// A game thread is done.
    pub fn worker_done(&self) {
        self.queue.lock().unwrap().workers -= 1;
        self.changed.notify_one();
    }

    /// The next batch: once half the playing threads wait (the others still play their
    /// previous answers), or the first waited `wait`; empty: all threads are done.
    fn batch(&self, wait: Duration) -> Vec<(Request, Slot)> {
        let mut q = self.queue.lock().unwrap();
        loop {
            let ready = match q.since {
                Some(t) => q.waiting.len() * 2 >= q.workers || t.elapsed() >= wait,
                None => q.workers == 0,
            };
            if ready {
                q.since = None;
                return std::mem::take(&mut q.waiting);
            }
            q = match q.since {
                Some(t) => self.changed.wait_timeout(q, wait.saturating_sub(t.elapsed())).unwrap().0,
                None => self.changed.wait(q).unwrap(),
            };
        }
    }

    /// Serves the batches over `out` and their answers from `input` until all threads are done;
    /// then sends `summary()`.
    pub fn serve(&self, input: &mut impl Read, out: &mut impl Write, wait: Duration, summary: impl FnOnce() -> String) -> std::io::Result<()> {
        loop {
            let batch = self.batch(wait);
            if batch.is_empty() {
                let s = summary();
                out.write_all(&2u32.to_le_bytes())?;
                out.write_all(&(s.len() as u32).to_le_bytes())?;
                out.write_all(s.as_bytes())?;
                return out.flush();
            }
            write_batch(out, batch.iter().map(|(r, _)| r))?;
            for (id, s) in read_answer(input)? {
                let (answer, ready) = &*batch[id as usize].1;
                *answer.lock().unwrap() = Some(s);
                ready.notify_one();
            }
        }
    }
}

/// A connection to a server (one request at a time).
pub struct Client {
    conn: Mutex<UnixStream>,
}

impl Client {
    pub fn connect(path: &std::path::Path) -> std::io::Result<Client> {
        Ok(Client { conn: Mutex::new(UnixStream::connect(path)?) })
    }

    pub fn score(&self, request: &Request) -> std::io::Result<Vec<f32>> {
        let mut conn = self.conn.lock().unwrap();
        write_batch(&mut *conn, [request].into_iter())?;
        let answer = read_answer(&mut *conn)?;
        answer.into_iter().next().map(|(_, s)| s).ok_or_else(|| std::io::Error::other("an empty answer"))
    }
}

fn write_batch<'a>(out: &mut impl Write, batch: impl Iterator<Item = &'a Request> + Clone) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    let mut put = |b: &[u8]| buf.extend_from_slice(b);
    put(&1u32.to_le_bytes());
    put(&(batch.clone().count() as u32).to_le_bytes());
    for (i, r) in batch.clone().enumerate() {
        for x in [i as u32, r.net, r.row.tok.len() as u32, r.row.opt.len() as u32, r.code_rows.len() as u32, r.string_rows.len() as u32] {
            put(&x.to_le_bytes());
        }
    }
    batch.clone().flat_map(|r| r.row.hdr).for_each(|x| put(&x.to_le_bytes()));
    batch.clone().flat_map(|r| r.row.tok.iter().flatten()).for_each(|x| put(&x.to_le_bytes()));
    batch.clone().flat_map(|r| r.row.opt.iter().flatten()).for_each(|x| put(&x.to_le_bytes()));
    batch.clone().flat_map(|r| &r.code_rows).for_each(|x| put(&x.to_le_bytes()));
    batch.flat_map(|r| r.string_rows.iter().flatten()).for_each(|x| put(&x.to_le_bytes()));
    out.write_all(&buf)?;
    out.flush()
}

/// An answer's (id, scores).
fn read_answer(input: &mut impl Read) -> std::io::Result<Vec<(u32, Vec<f32>)>> {
    let n = read_u32s(input, 1)?[0] as usize;
    let ids = read_u32s(input, n)?;
    let counts = read_u32s(input, n)?;
    let mut scores = vec![0u8; counts.iter().sum::<u32>() as usize * 4];
    input.read_exact(&mut scores)?;
    let mut at = 0;
    Ok(ids
        .into_iter()
        .zip(counts)
        .map(|(id, k)| {
            let s = scores[at..at + k as usize * 4].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            at += k as usize * 4;
            (id, s)
        })
        .collect())
}

fn read_u32s(input: &mut impl Read, n: usize) -> std::io::Result<Vec<u32>> {
    let mut b = vec![0u8; n * 4];
    input.read_exact(&mut b)?;
    Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
}
