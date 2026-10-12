//! The learned deck models (existence and counts): burn code that build.rs generates from their ONNX
//! export (download_human_decks/export_deck_models.py), the weights embedded. The inference around
//! them (the deck completion) is ygoai::belief::neural; this crate offers the two forward passes on
//! vocabulary indices, batch 1, on the CPU.

/// Whether the library was built with the models (an export existed).
pub fn available() -> bool {
    cfg!(deck_models)
}

/// The export's deck_models.json: the vocabulary (card codes by index), is_extra by index, the
/// corpus's average copies per distinct card.
pub fn meta_json() -> Option<&'static str> {
    #[cfg(deck_models)]
    return Some(include_str!(concat!(env!("OUT_DIR"), "/deck_models.json")));
    #[cfg(not(deck_models))]
    None
}

#[cfg(deck_models)]
mod generated {
    #[allow(clippy::all, unused)]
    pub mod deck_existence {
        include!(concat!(env!("OUT_DIR"), "/model/deck_existence.rs"));
    }
    #[allow(clippy::all, unused)]
    pub mod deck_counts {
        include!(concat!(env!("OUT_DIR"), "/model/deck_counts.rs"));
    }
}

/// The two models, loaded.
pub struct Models {
    #[cfg(deck_models)]
    ex: generated::deck_existence::Model,
    #[cfg(deck_models)]
    cnt: generated::deck_counts::Model,
}

#[cfg(deck_models)]
impl Models {
    pub fn load() -> Result<Models, String> {
        Ok(Models { ex: generated::deck_existence::Model::default(), cnt: generated::deck_counts::Model::default() })
    }

    /// The existence model's logit per vocabulary index, given the visible cards (indices with
    /// their counts, 1-3) and the deck's size.
    pub fn existence_logits(&self, cards: &[(usize, i64)], slots: f32) -> Vec<f32> {
        use burn::prelude::*;
        let l = cards.len().max(1);
        let (mut ids, mut counts, mut kpm) = (vec![0i64; l], vec![0i64; l], vec![true; l]);
        for (i, &(j, c)) in cards.iter().enumerate() {
            (ids[i], counts[i], kpm[i]) = (j as i64, c, false);
        }
        let d = Device::default();
        let logits = self.ex.forward(
            Tensor::<2, Int>::from_data(TensorData::new(ids, [1, l]), &d),
            Tensor::<2, Int>::from_data(TensorData::new(counts, [1, l]), &d),
            Tensor::<2, Bool>::from_data(TensorData::new(kpm, [1, l]), &d),
            Tensor::<1>::from_data(TensorData::new(vec![slots], [1]), &d),
        );
        logits.into_data().try_to_vec::<f32>().expect("f32 logits")
    }

    /// The count model's three logits (1, 2 or 3 copies) per card of the set (vocabulary indices),
    /// flattened.
    pub fn count_logits(&self, set: &[usize]) -> Vec<f32> {
        use burn::prelude::*;
        let l = set.len();
        let d = Device::default();
        let logits = self.cnt.forward(
            Tensor::<2, Int>::from_data(TensorData::new(set.iter().map(|&j| j as i64).collect::<Vec<_>>(), [1, l]), &d),
            Tensor::<2, Bool>::from_data(TensorData::new(vec![false; l], [1, l]), &d),
        );
        logits.into_data().try_to_vec::<f32>().expect("f32 logits")
    }
}

#[cfg(not(deck_models))]
impl Models {
    pub fn load() -> Result<Models, String> {
        Err("the deck models were not exported when this was built (download_human_decks/export_deck_models.py)".into())
    }
    pub fn existence_logits(&self, _cards: &[(usize, i64)], _slots: f32) -> Vec<f32> {
        unreachable!()
    }
    pub fn count_logits(&self, _set: &[usize]) -> Vec<f32> {
        unreachable!()
    }
}
