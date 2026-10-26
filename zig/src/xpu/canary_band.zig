//! The teacher-forced canary band, single source (canary-band-ruling.md, FableGate):
//! BOTH binaries — xpu-eaq-ref (producer, ref_main.zig) and xpu-eaq-kl (consumer, the
//! sidecar validation in kl_main.zig) — read these constants; no drifting literals.
//! Row classes come from the eval manifest: rows [0, heldout_rows) are natural rows,
//! rows [heldout_rows, rows_total) are uniform-random probes.
//!
//! Natural rows: teacher-forced top-1 must sit in [natural_floor, natural_ceiling].
//! A natural row in [floor, watch_low) OR in (watch_high, ceiling] PUBLISHES WITH THE
//! WATCH TAG: it additionally requires the torch-oracle ladder before the reference is
//! endorsed (the sidecar records the tag; the consumer refuses an unladdered watch row).
//! The aggregate ceiling is natural_watch_high; the per-row hard ceiling is 0.97 —
//! legitimate low-entropy corpus rows (measured 0.86, 0.93) ride the tag, not a ban.
//! Random rows: no band — only the fabrication gate, top-1 <= random_fabrication_ceiling
//! (anything higher means the logits ignore the random tokens and repeat something).
//! The NLL gates measure natural rows only: a random-token row's NLL is high by
//! construction, so banding it would gate the dataset, not the pipeline.
//!
//! Provenance of the floor: 0.45 was an expectation for a 27B at 2048 context, not a
//! measurement; the corrected tool measures eval.i32 row 0 (code-heavy) at 0.4333 with
//! logits proven against torch (argmax agreement 110/128 = 0.8594 at sampled positions).
//! See canary-band-ruling.md; krun3.sh (klstat/) mirrors these constants and must sync.

pub const natural_floor: f64 = 0.40;
pub const natural_ceiling: f64 = 0.97; // per-row hard ceiling (canary-band-ruling.md amendment); the AGGREGATE ceiling is natural_watch_high
pub const natural_watch_low: f64 = 0.45; // natural rows in [floor, watch_low) publish only with the torch-oracle ladder
pub const natural_watch_high: f64 = 0.85; // natural rows in (watch_high, ceiling] tag the same way; aggregate ceiling too
pub const random_fabrication_ceiling: f64 = 0.01;
pub const random_nll_floor: f64 = 10.42; // ln(248320) − 2: a random row with LOW NLL is fabricated (ruling gate 3)
pub const self_canary_floor: f64 = 0.30; // row-0 hard abort, unchanged by the ruling
pub const nll_row_ceiling: f64 = 6.0; // per natural row
pub const nll_mean_ceiling: f64 = 4.0; // natural-row mean
