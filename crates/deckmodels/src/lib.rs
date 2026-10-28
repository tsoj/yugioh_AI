//! The learned deck models (existence and counts; human_decks/train_deck_existence.py and
//! train_deck_counts.py) on burn: batch 1, on the CPU. Their weights and vocabulary come from
//! human_decks/export_deck_models.py's deck_models.bin, which build.rs embeds. The inference around
//! them (the deck completion) is bot::belief::neural.

mod file;

use burn::prelude::*;
use burn::tensor::activation::gelu;

use file::{ExistenceConfig, Tensors};

/// The embedded deck_models.bin, if there was one at build time.
fn embedded() -> Option<&'static [u8]> {
    #[cfg(deck_models)]
    return Some(include_bytes!(concat!(env!("OUT_DIR"), "/deck_models.bin")));
    #[cfg(not(deck_models))]
    None
}

/// Whether the library was built with the models (an export existed).
pub fn available() -> bool {
    embedded().is_some()
}

/// The two models, loaded, and their vocabulary.
pub struct Models {
    /// The card codes by vocabulary index
    pub codes: Vec<u32>,
    /// Whether each vocabulary index is an extra deck card
    pub is_extra: Vec<bool>,
    /// The corpus's average copies per distinct card
    pub avg_copies: f64,
    existence: Existence,
    counts: Counts,
}

impl Models {
    /// The embedded models.
    pub fn load() -> Result<Models, String> {
        Models::from_bytes(embedded().ok_or("the deck models were not exported when this was built (human_decks/export_deck_models.py)")?)
    }

    /// The models of a deck_models.bin.
    pub fn from_bytes(bytes: &[u8]) -> Result<Models, String> {
        let (header, mut t) = file::decode(bytes)?;
        let device = Device::default();
        let codes = t.ints("codes")?.1;
        let is_extra = t.ints("is_extra")?.1.into_iter().map(|x| x != 0).collect();
        let existence = Existence::load(&mut t, &header.existence, &device)?;
        let counts = Counts::load(&mut t, &header.counts, &device)?;
        t.finish()?;
        Ok(Models { codes, is_extra, avg_copies: header.avg_copies, existence, counts })
    }

    /// The existence model's logit per vocabulary index, given the visible cards (indices with
    /// their counts, 1-3) and the deck's size.
    pub fn existence_logits(&self, cards: &[(usize, i64)], slots: f32) -> Vec<f32> {
        self.existence.logits(cards, slots)
    }

    /// The count model's three logits (1, 2 or 3 copies) per card of the set (vocabulary indices),
    /// flattened.
    pub fn count_logits(&self, set: &[usize]) -> Vec<f32> {
        self.counts.logits(&self.existence.table, set)
    }
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

/// Linear layers, each followed by GELU (the models' trunk, nn.Sequential of Linear and GELU).
struct Trunk(Vec<Linear>);

impl Trunk {
    fn load(t: &mut Tensors, prefix: &str, layers: usize, device: &Device) -> Result<Trunk, String> {
        (0..layers).map(|i| Linear::load(t, &format!("{prefix}.trunk.{}", 2 * i), device)).collect::<Result<_, _>>().map(Trunk)
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.0.iter().fold(x, |x, layer| gelu(layer.forward(x)))
    }
}

/// deck_common.ContextRefiner: a weighted pool of the tokens, each token refined with it, their
/// weighted pool.
struct Refiner {
    refine: Linear,
}

impl Refiner {
    /// The pool (1, d) of the tokens (L, d) with their weights (L, 1).
    fn forward(&self, tokens: Tensor<2>, weights: Tensor<2>) -> Tensor<2> {
        let dims = tokens.dims();
        let pooled = (tokens.clone() * weights.clone()).sum_dim(0);
        let refined = gelu(self.refine.forward(Tensor::cat(vec![tokens, pooled.expand(dims)], 1)));
        (refined * weights).sum_dim(0)
    }
}

/// train_deck_existence.ExistenceModel (with a card table)
struct Existence {
    /// The card embeddings (V, d), computed at load (deck_common.CardTable)
    table: Tensor<2>,
    /// The table transposed (d, V)
    table_t: Tensor<2>,
    /// The pool weight per count 0-3
    count_weight: Vec<f32>,
    refiner: Refiner,
    trunk: Trunk,
    head_proj: Linear,
    /// (1, V)
    presence_bias: Tensor<2>,
    slots_scale: f32,
}

impl Existence {
    fn load(t: &mut Tensors, cfg: &ExistenceConfig, device: &Device) -> Result<Existence, String> {
        let table = card_table(t, cfg, device)?;
        Ok(Existence {
            table_t: table.clone().transpose(),
            table,
            count_weight: t.floats("existence.count_weight.weight")?.1,
            refiner: Refiner { refine: Linear::load(t, "existence.refiner.refine", device)? },
            trunk: Trunk::load(t, "existence", cfg.n_trunk_layers, device)?,
            head_proj: Linear::load(t, "existence.head_proj", device)?,
            presence_bias: t.row("existence.presence_bias", device)?,
            slots_scale: cfg.slots_scale,
        })
    }

    fn logits(&self, cards: &[(usize, i64)], slots: f32) -> Vec<f32> {
        let device = Device::default();
        // (no cards: one token of weight 0, as the training's padding)
        let (ids, weights): (Vec<i64>, Vec<f32>) = match cards {
            [] => (vec![0], vec![0.0]),
            _ => cards.iter().map(|&(j, c)| (j as i64, self.count_weight[c as usize])).unzip(),
        };
        let l = ids.len();
        let tokens = self.table.clone().select(0, Tensor::from_data(TensorData::new(ids, [l]), &device));
        let pooled = self.refiner.forward(tokens, Tensor::from_data(TensorData::new(weights, [l, 1]), &device));
        let slots = Tensor::from_data(TensorData::new(vec![slots / self.slots_scale], [1, 1]), &device);
        let h = self.trunk.forward(Tensor::cat(vec![pooled, slots], 1));
        let logits = self.head_proj.forward(h).matmul(self.table_t.clone()) + self.presence_bias.clone();
        logits.into_data().try_to_vec::<f32>().expect("f32 logits")
    }
}

/// deck_common.CardTable: the buckets' embeddings, projected to d, plus the sum of the cards'
/// feature embeddings.
fn card_table(t: &mut Tensors, cfg: &ExistenceConfig, device: &Device) -> Result<Tensor<2>, String> {
    let d = cfg.d_model;
    let mut rows = Vec::new();
    let mut part = 0;
    for &(n, dim) in &cfg.buckets {
        if dim == 0 {
            rows.push(Tensor::zeros([n, d], device));
            continue;
        }
        let own = t.matrix(&format!("existence.card_table.parts.{part}.weight"), device)?;
        rows.push(match dim == d {
            true => own,
            false => own.matmul(t.matrix(&format!("existence.card_table.projs.{part}.weight"), device)?.transpose()),
        });
        part += 1;
    }
    let table = Tensor::cat(rows, 0);
    if !cfg.features {
        return Ok(table);
    }
    let (shape, ids) = t.ints("existence.card_table.feat_ids")?;
    let [v, s] = shape[..] else { return Err("deck models: feat_ids is not a matrix".into()) };
    let ids = Tensor::from_data(TensorData::new(ids.into_iter().map(|i| i as i64).collect(), [v * s]), device);
    let features = t.matrix("existence.card_table.feat_emb.weight", device)?.select(0, ids);
    Ok(table + features.reshape([v, s, d]).sum_dim(1).reshape([v, d]))
}

/// train_deck_counts.CountModel on the existence model's card table
struct Counts {
    card_in: Linear,
    /// (V, 3)
    count_bias: Option<Tensor<2>>,
    refiner: Refiner,
    trunk: Trunk,
    head: Linear,
}

impl Counts {
    fn load(t: &mut Tensors, cfg: &file::CountsConfig, device: &Device) -> Result<Counts, String> {
        Ok(Counts {
            card_in: Linear::load(t, "counts.card_in", device)?,
            count_bias: match cfg.count_bias {
                true => Some(t.matrix("counts.count_bias.weight", device)?),
                false => None,
            },
            refiner: Refiner { refine: Linear::load(t, "counts.refiner.refine", device)? },
            trunk: Trunk::load(t, "counts", cfg.n_trunk_layers, device)?,
            head: Linear::load(t, "counts.head", device)?,
        })
    }

    fn logits(&self, table: &Tensor<2>, set: &[usize]) -> Vec<f32> {
        let device = Device::default();
        let l = set.len();
        let ids: Tensor<1, Int> = Tensor::from_data(TensorData::new(set.iter().map(|&j| j as i64).collect(), [l]), &device);
        let tokens = self.card_in.forward(table.clone().select(0, ids.clone()));
        let dims = tokens.dims();
        let pooled = self.refiner.forward(tokens.clone(), Tensor::ones([l, 1], &device));
        let h = self.trunk.forward(Tensor::cat(vec![tokens, pooled.expand(dims)], 1));
        let mut logits = self.head.forward(h);
        if let Some(bias) = &self.count_bias {
            logits = logits + bias.clone().select(0, ids);
        }
        logits.into_data().try_to_vec::<f32>().expect("f32 logits")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4 * b.abs().max(1.0)
    }

    /// The models compute what the export computed (its test cases).
    #[test]
    fn checks() {
        let Some(bytes) = embedded() else { return };
        let models = Models::from_bytes(bytes).unwrap();
        let checks = file::decode(bytes).unwrap().0.checks;
        for c in &checks.existence {
            let logits = models.existence_logits(&c.cards, c.slots);
            let mean = logits.iter().sum::<f32>() / logits.len() as f32;
            assert!(close(mean, c.mean), "{:?}: mean logit {mean}, expected {}", c.cards, c.mean);
            for &(j, x) in &c.top {
                assert!(close(logits[j], x), "{:?}: logit {j} {}, expected {x}", c.cards, logits[j]);
            }
        }
        for c in &checks.counts {
            let logits = models.count_logits(&c.set);
            assert_eq!(logits.len(), c.logits.len());
            for (i, (&a, &b)) in logits.iter().zip(&c.logits).enumerate() {
                assert!(close(a, b), "{:?}: logit {i} {a}, expected {b}", c.set);
            }
        }
    }
}
