//! HiGHS 1.12.0, the version scipy 1.18 bundles, for the same solutions as
//! `scipy.optimize.milp` on the same model (belief::flow).

use std::ffi::{CString, c_char, c_void};

type HighsInt = i32;

unsafe extern "C" {
    fn Highs_create() -> *mut c_void;
    fn Highs_destroy(highs: *mut c_void);
    fn Highs_setBoolOptionValue(highs: *mut c_void, option: *const c_char, value: HighsInt) -> HighsInt;
    fn Highs_setIntOptionValue(highs: *mut c_void, option: *const c_char, value: HighsInt) -> HighsInt;
    fn Highs_setDoubleOptionValue(highs: *mut c_void, option: *const c_char, value: f64) -> HighsInt;
    #[allow(clippy::too_many_arguments)]
    fn Highs_passMip(
        highs: *mut c_void,
        num_col: HighsInt,
        num_row: HighsInt,
        num_nz: HighsInt,
        a_format: HighsInt,
        sense: HighsInt,
        offset: f64,
        col_cost: *const f64,
        col_lower: *const f64,
        col_upper: *const f64,
        row_lower: *const f64,
        row_upper: *const f64,
        a_start: *const HighsInt,
        a_index: *const HighsInt,
        a_value: *const f64,
        integrality: *const HighsInt,
    ) -> HighsInt;
    fn Highs_run(highs: *mut c_void) -> HighsInt;
    fn Highs_getModelStatus(highs: *const c_void) -> HighsInt;
    fn Highs_getSolution(highs: *const c_void, col_value: *mut f64, col_dual: *mut f64, row_value: *mut f64, row_dual: *mut f64) -> HighsInt;
}

const STATUS_ERROR: HighsInt = -1;
const MATRIX_COLWISE: HighsInt = 1;
const SENSE_MINIMIZE: HighsInt = 1;
const VAR_INTEGER: HighsInt = 1;
/// kHighsModelStatusOptimal
pub const MODEL_OPTIMAL: HighsInt = 7;

/// An integer program: minimize `cost . x` subject to `row_lower <= A x <= row_upper` and
/// `col_lower <= x <= col_upper`, every x integer; A column-wise (CSC: `start`, `index`, `value`).
pub struct Mip {
    pub cost: Vec<f64>,
    pub col_lower: Vec<f64>,
    pub col_upper: Vec<f64>,
    pub row_lower: Vec<f64>,
    pub row_upper: Vec<f64>,
    pub start: Vec<i32>,
    pub index: Vec<i32>,
    pub value: Vec<f64>,
}

/// The solution, or the model status (a HiGHS kHighsModelStatus code) if not optimal.
pub fn solve(mip: &Mip, time_limit: f64, threads: i32) -> Result<Vec<f64>, i32> {
    let ncol = mip.cost.len();
    let nrow = mip.row_lower.len();
    assert!(mip.col_lower.len() == ncol && mip.col_upper.len() == ncol && mip.row_upper.len() == nrow);
    assert!(mip.start.len() == ncol + 1 && mip.index.len() == mip.value.len());
    let integrality = vec![VAR_INTEGER; ncol];
    let opt = |s: &str| CString::new(s).unwrap();
    // SAFETY: the pointers are to vectors of the sizes HiGHS is told (checked above), alive for the
    // calls; the instance is destroyed before returning.
    unsafe {
        let h = Highs_create();
        // (the options scipy.optimize.milp passes for belief/flow.py)
        Highs_setBoolOptionValue(h, opt("log_to_console").as_ptr(), 0);
        Highs_setDoubleOptionValue(h, opt("time_limit").as_ptr(), time_limit);
        Highs_setIntOptionValue(h, opt("threads").as_ptr(), threads);
        let st = Highs_passMip(
            h,
            ncol as HighsInt,
            nrow as HighsInt,
            mip.value.len() as HighsInt,
            MATRIX_COLWISE,
            SENSE_MINIMIZE,
            0.0,
            mip.cost.as_ptr(),
            mip.col_lower.as_ptr(),
            mip.col_upper.as_ptr(),
            mip.row_lower.as_ptr(),
            mip.row_upper.as_ptr(),
            mip.start.as_ptr(),
            mip.index.as_ptr(),
            mip.value.as_ptr(),
            integrality.as_ptr(),
        );
        if st == STATUS_ERROR || Highs_run(h) == STATUS_ERROR {
            let status = Highs_getModelStatus(h);
            Highs_destroy(h);
            return Err(status);
        }
        let status = Highs_getModelStatus(h);
        if status != MODEL_OPTIMAL {
            Highs_destroy(h);
            return Err(status);
        }
        let mut x = vec![0.0; ncol];
        let mut col_dual = vec![0.0; ncol];
        let mut row_value = vec![0.0; nrow];
        let mut row_dual = vec![0.0; nrow];
        Highs_getSolution(h, x.as_mut_ptr(), col_dual.as_mut_ptr(), row_value.as_mut_ptr(), row_dual.as_mut_ptr());
        Highs_destroy(h);
        Ok(x)
    }
}
