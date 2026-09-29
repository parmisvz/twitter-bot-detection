# Target-domain independent audit: target_audit_v1

Approved sampling: 600 of 18,331 frozen eligible unlabeled accounts, seed 20260928,
NumPy PCG64. Follower strata: <100, 100–999, >=1000, missing; each crossed with
observed/missing raw description. Minimum allocation min(10,N_h), then proportional
to remaining capacity with integer largest remainders and fixed-order ties.
Selection is uniform without replacement within lexicographically sorted user keys.
Inclusion probability n_h/N_h; analysis weight N_h/n_h. No replacement for difficult,
inaccessible or uncertain accounts. Sampling uses no scores or inferred classes.

## Coordinator instructions

Keep `private/` files private. Provide annotators only their own blank form copy and
independent evidence packets indexed by annotation_id. The evidence index is for
the coordinator; stratum, sampling weights and prior model exposure are not shown
to annotators. Do not reveal model predictions or the other annotator's decisions.
JSONL means one JSON form per line. Edit a COPY of each template, preserving IDs;
keep the original hashed package unchanged. Spreadsheet conversion, if needed,
must preserve IDs and blanks and must not add model scores.

Evidence collection has NOT been authorized or performed. The raw snapshots lack
retained tweets. If adequate archived evidence is unavailable, use Uncertain rather
than infer labels from account statistics. Current behavior is not automatically
evidence about the period represented by the frozen snapshot.

## Rubric

Target: predominantly human versus substantially automated operation during the
period supported by the frozen account snapshot. Human requires affirmative
evidence of predominantly human operation. Bot requires explicit automation
evidence or multiple corroborating indicators. Uncertain covers insufficient,
contradictory, inaccessible, temporally mismatched or unresolved mixed operation.
Valid labels: Human, Bot, Uncertain. Leave labels blank until annotation.
Confidence: low, medium or high, with a written rationale; confidence is not a
probability. Access status: accessible, archived_only, inaccessible, deleted,
protected or other (explain). Political stance, popularity, anonymity, default
images, high activity or repetition alone are insufficient. News/organization
status is recorded separately and does not imply automation. Benign bots count.

Use a non-audit pilot to train annotators and clarify examples before starting.
Two annotators independently label all 600. Preserve original decisions. Human/
Human or Bot/Bot is provisional agreement subject to evidence completeness. Every
disagreement, including Uncertain disagreements, goes to a third adjudicator.
Do not force majority labels without adequate evidence. Persist Uncertain when
unresolved. Record agreement counts, uncertainty rates and adjudication outcomes.

Each form records evidence dates, supported period, sources/types, summary,
automation indicators, human-operation indicators, limitations, temporal mismatch,
access status, label/confidence/rationale, annotator and organization status.
No posts are fetched, accounts contacted or labels assigned by this sampling tool.

## Protected roles

All future training, preprocessing fitting, representation fitting, adaptation,
pseudo-label training, calibration, threshold selection, model selection and
external transfer must call `src.target_audit.assert_not_audit(keys)` on every
participating identity set. Missing/corrupt registry fails closed. Historical
training exposure cannot be undone: private/prior_exposure.json checks the known
390-user semi-supervised student selection only, not every legacy experiment.
Audit membership is never changed based on exposure. Verify each evaluated
model's saved training keys before claiming independence.

Existing experiment implementations are preserved, not retroactively rewritten.
The registry and guard do not sandbox arbitrary legacy scripts. Do not run legacy
unlabeled-training scripts without adding the guard at their identity boundaries.

## Later evaluation: NOT executed

Freeze M0/M2/M4 selected checkpoints for seeds 42/43/44 and threshold 0.5. Same
600 identities, existing train-fitted transformations. M4 must use its existing
missing-text path; no description substitution or annotation evidence as input.
Check training identity disjointness, hashes and reload reproduction. Report
weighted confusion totals and derived precision/recall/F1, AUC/AP, Brier/BCE,
per-seed results, stratum coverage and uncertainty. Respect stratified finite-
population sampling in variance estimation; never count tweets or seed copies as
independent users. Binary metrics on resolved labels are resolved-subset metrics;
report weighted Uncertain coverage and sensitivity bounds. Weights alone cannot
fix nonrandom annotation nonresponse. No evaluation or bootstrap is authorized now.

## Future target development: NOT selected

If needed, select a disjoint 300-account development set from the remaining users:
200 stratified random and 100 separately approved disagreement/coverage-enriched.
Freeze 200 development-training and 100 development-validation/calibration roles
before revealing labels. Preserve sampling-component identifiers. Never use the
600 audit identities for fitting or selection. Once audit results influence model
redesign, subsequent results on these same 600 are reused-audit results; reserve a
new independent confirmation sample for the revised model.
