//! The imitation policy (scripts/policy/train.py's `Policy`) on burn: batch 1, on the CPU. Its
//! weights come from scripts/policy/export_policy.py, the text vectors it reads (cards and card
//! strings under a game's pseudo-names) from scripts/policy/embed_tables.py (`Tables`). A
//! decision's rows are bot::policy_features'.

pub mod cols;
mod file;
mod tables;

use burn::prelude::*;
use burn::tensor::activation::{gelu, softmax};

use cols::*;
pub use file::Check;
use file::Tensors;
pub use tables::Tables;

/// A decision's rows as the model reads them (train.Batcher's, one decision): at most
/// `max_tokens` tokens, `max_options` options, references beyond them -1; and the text vectors
/// the rows refer to (None: no text).
pub struct Inputs<'a> {
    pub hdr: [i64; HDR],
    pub tok: Vec<[i64; TOK]>,
    pub opt: Vec<[i64; OPT]>,
    /// per token: its card's text, its reference's card's text
    pub tok_text: Vec<Option<&'a [f32]>>,
    pub ref_text: Vec<Option<&'a [f32]>>,
    /// per option: its text's card, its card string
    pub opt_desc_text: Vec<Option<&'a [f32]>>,
    pub opt_str_text: Vec<Option<&'a [f32]>>,
    /// the hint's, the question's, H_CODE's and the previous choice's cards
    pub glob_text: [Option<&'a [f32]>; 4],
    /// the hint's and the question's card strings
    pub glob_str_text: [Option<&'a [f32]>; 2],
}

/// The model's sizes (export_policy.py's "consts" and train.py's arguments).
#[derive(Clone, Copy)]
pub struct Sizes {
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub emb_dim: usize,
    pub n_prompts: usize,
    pub sys_buckets: usize,
    pub max_tokens: usize,
    pub max_options: usize,
    pub locs: usize,
    pub kinds: usize,
    pub strings: bool,
}

/// A linear layer, as PyTorch's nn.Linear.
struct Linear {
    /// (in, out)
    weight: Tensor<2>,
    /// (1, out)
    bias: Tensor<2>,
}

impl Linear {
    fn load(t: &mut Tensors, name: &str, device: &Device) -> Result<Linear, String> {
        // (PyTorch's weight is (out, in))
        Ok(Linear { weight: t.matrix(&format!("{name}.weight"), device)?.transpose(), bias: t.row(&format!("{name}.bias"), device)? })
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.matmul(self.weight.clone()) + self.bias.clone()
    }
}

/// nn.LayerNorm over the last dimension (eps 1e-5).
struct LayerNorm {
    weight: Tensor<2>,
    bias: Tensor<2>,
}

impl LayerNorm {
    fn load(t: &mut Tensors, name: &str, device: &Device) -> Result<LayerNorm, String> {
        Ok(LayerNorm { weight: t.row(&format!("{name}.weight"), device)?, bias: t.row(&format!("{name}.bias"), device)? })
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let centered = x.clone() - x.mean_dim(1);
        let var = centered.clone().powi_scalar(2).mean_dim(1);
        centered / (var + 1e-5).sqrt() * self.weight.clone() + self.bias.clone()
    }
}

/// nn.TransformerEncoderLayer(norm_first=True, activation="gelu"): self-attention and the
/// feed-forward block, each on the layer norm of its input, added to it.
struct EncoderLayer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    /// q, k, v in one: (dim, 3 dim)
    in_proj: Linear,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
    heads: usize,
}

impl EncoderLayer {
    fn load(t: &mut Tensors, name: &str, heads: usize, device: &Device) -> Result<EncoderLayer, String> {
        let in_proj = Linear {
            weight: t.matrix(&format!("{name}.self_attn.in_proj_weight"), device)?.transpose(),
            bias: t.row(&format!("{name}.self_attn.in_proj_bias"), device)?,
        };
        Ok(EncoderLayer {
            norm1: LayerNorm::load(t, &format!("{name}.norm1"), device)?,
            norm2: LayerNorm::load(t, &format!("{name}.norm2"), device)?,
            in_proj,
            out_proj: Linear::load(t, &format!("{name}.self_attn.out_proj"), device)?,
            linear1: Linear::load(t, &format!("{name}.linear1"), device)?,
            linear2: Linear::load(t, &format!("{name}.linear2"), device)?,
            heads,
        })
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let dim = x.dims()[1];
        let hd = dim / self.heads;
        let qkv = self.in_proj.forward(self.norm1.forward(x.clone()));
        let scale = 1.0 / (hd as f32).sqrt();
        let heads: Vec<Tensor<2>> = (0..self.heads)
            .map(|h| {
                let q = qkv.clone().narrow(1, h * hd, hd);
                let k = qkv.clone().narrow(1, dim + h * hd, hd);
                let v = qkv.clone().narrow(1, 2 * dim + h * hd, hd);
                softmax(q.matmul(k.transpose()) * scale, 1).matmul(v)
            })
            .collect();
        let x = x + self.out_proj.forward(Tensor::cat(heads, 1));
        let ff = self.linear2.forward(gelu(self.linear1.forward(self.norm2.forward(x.clone()))));
        x + ff
    }
}

/// An embedding table (nn.Embedding) as rows.
struct Embedding {
    rows: Vec<f32>,
    dim: usize,
}

impl Embedding {
    fn load(t: &mut Tensors, name: &str) -> Result<Embedding, String> {
        let (shape, rows) = t.floats(&format!("{name}.weight"))?;
        Ok(Embedding { rows, dim: shape[1] })
    }

    fn row(&self, i: i64) -> &[f32] {
        &self.rows[i as usize * self.dim..(i as usize + 1) * self.dim]
    }

    /// Adds row `i` to `out`.
    fn add(&self, i: i64, out: &mut [f32]) {
        out.iter_mut().zip(self.row(i)).for_each(|(o, x)| *o += x);
    }
}

/// The policy network.
pub struct Policy {
    pub sizes: Sizes,
    device: Device,
    text: Linear,
    text_ref: Linear,
    text_glob: Vec<Linear>,
    text_desc: Linear,
    /// (with strings) per option, and the hint's and the question's
    str_opt: Option<Linear>,
    str_glob: Vec<Linear>,
    unknown: Vec<f32>,
    con: Embedding,
    owner: Embedding,
    locseq: Embedding,
    pos: Embedding,
    level: Embedding,
    scale: Embedding,
    counters: Embedding,
    overlays: Embedding,
    count: Embedding,
    status: Linear,
    stats: Linear,
    has_ref: Embedding,
    g_kind: Vec<f32>,
    prompt: Embedding,
    tp: Embedding,
    phase: Embedding,
    turn: Embedding,
    minmax: Embedding,
    flags: Embedding,
    sys_hint: Embedding,
    sys_desc: Embedding,
    str_hint: Embedding,
    str_desc: Embedding,
    prev_kind: Embedding,
    g_num: Linear,
    o_kind: Embedding,
    o_aux: Embedding,
    o_sys: Embedding,
    o_str: Embedding,
    o_marker: Vec<f32>,
    trunk: Vec<EncoderLayer>,
    norm: LayerNorm,
    head1: Linear,
    head2: Linear,
}

/// `train.bucket`: clamped to 0..n.
fn bucket(x: i64, n: usize) -> i64 {
    x.clamp(0, n as i64 - 1)
}

impl Policy {
    /// The model of an export_policy.py file, and its checks.
    pub fn from_bytes(bytes: &[u8]) -> Result<(Policy, Vec<Check>), String> {
        let (header, mut t) = file::decode(bytes)?;
        let device = Device::default();
        let d = &device;
        let sizes = header.sizes();
        let vector = |t: &mut Tensors, name: &str| t.floats(name).map(|x| x.1);
        let p = Policy {
            sizes,
            text: Linear::load(&mut t, "text", d)?,
            text_ref: Linear::load(&mut t, "text_ref", d)?,
            text_glob: (0..4).map(|i| Linear::load(&mut t, &format!("text_glob.{i}"), d)).collect::<Result<_, _>>()?,
            text_desc: Linear::load(&mut t, "text_desc", d)?,
            str_opt: if sizes.strings { Some(Linear::load(&mut t, "str_opt", d)?) } else { None },
            str_glob: if sizes.strings { (0..2).map(|i| Linear::load(&mut t, &format!("str_glob.{i}"), d)).collect::<Result<_, _>>()? } else { Vec::new() },
            unknown: vector(&mut t, "unknown")?,
            con: Embedding::load(&mut t, "con")?,
            owner: Embedding::load(&mut t, "owner")?,
            locseq: Embedding::load(&mut t, "locseq")?,
            pos: Embedding::load(&mut t, "pos")?,
            level: Embedding::load(&mut t, "level")?,
            scale: Embedding::load(&mut t, "scale")?,
            counters: Embedding::load(&mut t, "counters")?,
            overlays: Embedding::load(&mut t, "overlays")?,
            count: Embedding::load(&mut t, "count")?,
            status: Linear::load(&mut t, "status", d)?,
            stats: Linear::load(&mut t, "stats", d)?,
            has_ref: Embedding::load(&mut t, "has_ref")?,
            g_kind: vector(&mut t, "g_kind")?,
            prompt: Embedding::load(&mut t, "prompt")?,
            tp: Embedding::load(&mut t, "tp")?,
            phase: Embedding::load(&mut t, "phase")?,
            turn: Embedding::load(&mut t, "turn")?,
            minmax: Embedding::load(&mut t, "minmax")?,
            flags: Embedding::load(&mut t, "flags")?,
            sys_hint: Embedding::load(&mut t, "sys_hint")?,
            sys_desc: Embedding::load(&mut t, "sys_desc")?,
            str_hint: Embedding::load(&mut t, "str_hint")?,
            str_desc: Embedding::load(&mut t, "str_desc")?,
            prev_kind: Embedding::load(&mut t, "prev_kind")?,
            g_num: Linear::load(&mut t, "g_num", d)?,
            o_kind: Embedding::load(&mut t, "o_kind")?,
            o_aux: Embedding::load(&mut t, "o_aux")?,
            o_sys: Embedding::load(&mut t, "o_sys")?,
            o_str: Embedding::load(&mut t, "o_str")?,
            o_marker: vector(&mut t, "o_marker")?,
            trunk: (0..sizes.layers).map(|i| EncoderLayer::load(&mut t, &format!("trunk.layers.{i}"), sizes.heads, d)).collect::<Result<_, _>>()?,
            norm: LayerNorm::load(&mut t, "norm", d)?,
            head1: Linear::load(&mut t, "head.0", d)?,
            head2: Linear::load(&mut t, "head.2", d)?,
            device,
        };
        t.finish()?;
        Ok((p, header.checks))
    }

    /// A matrix (rows, cols) of row-major values.
    fn matrix(&self, values: Vec<f32>, rows: usize, cols: usize) -> Tensor<2> {
        Tensor::from_data(TensorData::new(values, [rows, cols]), &self.device)
    }

    /// `lin` of each text (rows; none: a zero row), as values.
    fn texts(&self, lin: &Linear, texts: &[Option<&[f32]>]) -> Vec<f32> {
        let (n, e) = (texts.len(), self.sizes.emb_dim);
        let mut x = vec![0f32; n * e];
        for (i, t) in texts.iter().enumerate() {
            if let Some(t) = t {
                x[i * e..(i + 1) * e].copy_from_slice(t);
            }
        }
        let mut out: Vec<f32> = lin.forward(self.matrix(x, n, e)).into_data().try_to_vec::<f32>().expect("f32");
        let dim = self.sizes.dim;
        for (i, t) in texts.iter().enumerate() {
            if t.is_none() {
                out[i * dim..(i + 1) * dim].fill(0.0);
            }
        }
        out
    }

    /// `lin` of rows of `k` values each.
    fn linear(&self, lin: &Linear, values: Vec<f32>, k: usize) -> Vec<f32> {
        let n = values.len() / k;
        lin.forward(self.matrix(values, n, k)).into_data().try_to_vec::<f32>().expect("f32")
    }

    /// The options' scores (train.Policy.forward).
    pub fn scores(&self, x: &Inputs) -> Vec<f32> {
        let s = &self.sizes;
        let dim = s.dim;
        let (lt, lo) = (x.tok.len(), x.opt.len());
        let add = |out: &mut [f32], v: &[f32]| out.iter_mut().zip(v).for_each(|(o, x)| *o += x);

        // tokens
        let mut tok = self.texts(&self.text, &x.tok_text);
        let refs = self.texts(&self.text_ref, &x.ref_text);
        let mut bits = Vec::with_capacity(lt * 8);
        let mut stats = Vec::with_capacity(lt * 8);
        for t in &x.tok {
            for i in 0..8 {
                bits.push(((t[T_STATUS] >> i) & 1) as f32);
            }
            let st = [t[T_ATK], t[T_DEF], t[T_BASE_ATK], t[T_BASE_DEF]];
            stats.extend(st.iter().map(|&v| v.max(0) as f32 / 3000.0));
            stats.extend(st.iter().map(|&v| (v >= 0) as i32 as f32));
        }
        let status = self.linear(&self.status, bits, 8);
        let stat = self.linear(&self.stats, stats, 8);
        for (i, t) in x.tok.iter().enumerate() {
            let out = &mut tok[i * dim..(i + 1) * dim];
            if x.tok_text[i].is_none() {
                add(out, &self.unknown);
            }
            self.con.add(t[T_CON].clamp(0, 1), out);
            self.owner.add(t[T_OWNER].clamp(0, 2), out);
            self.locseq.add(t[T_LOC].clamp(0, s.locs as i64 - 1) * 32 + t[T_SEQ].clamp(0, 31), out);
            self.pos.add(t[T_POS].clamp(0, 7), out);
            self.level.add(bucket(t[T_LEVEL], 16), out);
            self.scale.add(bucket(t[T_SCALE], 16), out);
            self.counters.add(bucket(t[T_COUNTERS], 8), out);
            self.overlays.add(bucket(t[T_OVERLAYS], 8), out);
            self.count.add(bucket(t[T_COUNT], 8), out);
            add(out, &status[i * dim..(i + 1) * dim]);
            add(out, &stat[i * dim..(i + 1) * dim]);
            add(out, &refs[i * dim..(i + 1) * dim]);
            self.has_ref.add((t[T_REF] >= 0) as i64, out);
        }

        // the global token
        let h = &x.hdr;
        let mut g = self.g_kind.clone();
        self.prompt.add(h[H_PROMPT].clamp(0, s.n_prompts as i64 - 1), &mut g);
        self.tp.add(h[H_TURN_PLAYER].clamp(0, 1), &mut g);
        self.phase.add(((h[H_PHASE].max(1) as f32).log2() as i64).clamp(0, 10), &mut g);
        self.turn.add(bucket(h[H_TURN], 32), &mut g);
        self.minmax.add(bucket(h[H_MIN], 8), &mut g);
        self.minmax.add(8 + bucket(h[H_MAX], 8), &mut g);
        self.flags.add(bucket(h[H_FLAGS], 8), &mut g);
        let sys = |code: i64, string: i64| code == 0 && string > 0;
        let buckets = s.sys_buckets as i64;
        if sys(h[H_HINT_CODE], h[H_HINT_STR]) {
            self.sys_hint.add(h[H_HINT_STR] % buckets, &mut g);
        } else {
            self.str_hint.add(bucket(h[H_HINT_STR], 32), &mut g);
        }
        if sys(h[H_DESC_CODE], h[H_DESC_STR]) {
            self.sys_desc.add(h[H_DESC_STR] % buckets, &mut g);
        } else {
            self.str_desc.add(bucket(h[H_DESC_STR], 32), &mut g);
        }
        self.prev_kind.add(h[H_PREV_KIND] + 1, &mut g);
        let num = vec![h[H_LP_ME] as f32 / 8000.0, h[H_LP_OPP] as f32 / 8000.0, h[H_DECK_ME] as f32 / 40.0, h[H_DECK_OPP] as f32 / 40.0, h[H_EXTRA_OPP] as f32 / 15.0];
        add(&mut g, &self.linear(&self.g_num, num, 5));
        for (lin, t) in self.text_glob.iter().zip(&x.glob_text) {
            add(&mut g, &self.texts(lin, &[*t]));
        }
        for (lin, t) in self.str_glob.iter().zip(&x.glob_str_text) {
            add(&mut g, &self.texts(lin, &[*t]));
        }

        // options
        let mut opt = self.texts(&self.text_desc, &x.opt_desc_text);
        if let Some(lin) = &self.str_opt {
            add(&mut opt, &self.texts(lin, &x.opt_str_text));
        }
        for (i, o) in x.opt.iter().enumerate() {
            let out = &mut opt[i * dim..(i + 1) * dim];
            let kind = o[O_KIND].clamp(0, s.kinds as i64 - 1);
            add(out, &self.o_marker);
            self.o_kind.add(kind, out);
            self.o_aux.add(kind * 64 + o[O_AUX].clamp(0, 63), out);
            if sys(o[O_DESC_CODE], o[O_DESC_STR]) {
                self.o_sys.add(o[O_DESC_STR] % buckets, out);
            } else {
                self.o_str.add(bucket(o[O_DESC_STR], 32), out);
            }
            if o[O_TOKEN] >= 0 {
                let r = o[O_TOKEN] as usize;
                let row: Vec<f32> = tok[r * dim..(r + 1) * dim].to_vec();
                add(out, &row);
            }
        }

        // the trunk over [global, tokens, options]
        let mut seq = g;
        seq.extend_from_slice(&tok);
        seq.extend_from_slice(&opt);
        let mut y = self.matrix(seq, 1 + lt + lo, dim);
        for layer in &self.trunk {
            y = layer.forward(y);
        }
        let y = self.norm.forward(y);
        let ty: Vec<f32> = y.clone().narrow(0, 1, lt.max(1)).into_data().try_to_vec::<f32>().expect("f32");
        let oy = y.narrow(0, 1 + lt, lo);
        let mut refs = vec![0f32; lo * dim];
        for (i, o) in x.opt.iter().enumerate() {
            if o[O_TOKEN] >= 0 && lt > 0 {
                let r = o[O_TOKEN] as usize;
                refs[i * dim..(i + 1) * dim].copy_from_slice(&ty[r * dim..(r + 1) * dim]);
            }
        }
        let both = Tensor::cat(vec![oy, self.matrix(refs, lo, dim)], 1);
        self.head2.forward(gelu(self.head1.forward(both))).into_data().try_to_vec::<f32>().expect("f32")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The export's checks: the same scores as PyTorch (POLICY_NET: an export_policy.py file).
    #[test]
    fn checks() {
        let Some(path) = std::env::var_os("POLICY_NET") else {
            eprintln!("POLICY_NET not set: skipped");
            return;
        };
        let (p, checks) = Policy::from_bytes(&std::fs::read(path).unwrap()).unwrap();
        for c in &checks {
            let scores = p.scores(&c.inputs());
            let worst = scores.iter().zip(&c.scores).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            eprintln!("{} options, max |diff| {worst:.5}", scores.len());
            assert!(worst < 1e-2, "{scores:?} vs {:?}", c.scores);
        }
    }
}
