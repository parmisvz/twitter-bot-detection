# Non-audit annotation pilot: coordinator workflow

The package assigns 18 previously unlabeled users to pilot-only annotation practice.
Keep private/ and selection_provenance.json coordinator-only. Provide each annotator
only their own folder. No model scores, predicted classes, pseudo-labels, sampling
groups or model uncertainty are included in their forms.

Three users are selected from each remaining nonempty follower-count by description
availability group using fixed-seed hash ranking. This deliberately varied exercise
is not a representative performance sample and makes no promise of including bots.
The two follower-missing users were already in the frozen audit and are excluded.
The earlier ten unassigned practice slots remain unchanged; use this new assigned
18-user package for the pilot, not both sets of templates.

## Review sequence

1. Before annotation, arrange authorized, dated evidence packets indexed by pilot
   annotation_id. Use the private mapping and snapshot date. No evidence has been
   collected by this preparation. Do not substitute current behavior for past evidence.
2. Each annotator reviews all 18 independently using the existing rubric and their
   own CSV. Preserve IDs and save a new submission. Do not reveal the other's labels.
3. Lock and preserve both original submissions before comparing them. Check missing
   evidence, dates, rationales and label spelling. A blank is not an Uncertain label.
4. Review every disagreement and every Uncertain case together with an adjudicator.
   Discuss whether the cause is missing evidence, temporal mismatch, mixed operation,
   or an unclear rubric. Use the separate adjudication form and retain original labels.
   Do not force Human/Bot agreement when evidence remains insufficient.
5. Summarize agreement counts and unresolved cases as annotation-process diagnostics,
   never model metrics. Clarify examples using pilot cases only. Record clarifications
   in a new dated rubric version before starting the audit. If a clarification changes
   the label definition rather than explaining it, seek approval first.
6. Proceed to the 600-user audit only after both annotators can apply the rubric
   consistently and the coordinator has explicitly authorized the start. If substantial
   disagreement persists, resolve the workflow first rather than practice on audit cases.

## Reserved use

Pilot labels are for annotator training only; do not merge them into training,
development, calibration or evaluation labels. Before any future M0/M2/M4 evaluation,
call `src.target_annotation_pilot.assert_not_pilot_evaluation(keys)` on all evaluated
identities. The pinned pilot registry fails closed if missing or altered. Existing
evaluation scripts are not automatically instrumented; add the guard before running
them. Continue enforcing the independent 600-user audit exclusion guard for fitting.

The frozen audit registry and all prior artifacts remain unchanged. Only identity
and split columns of the frozen split file were read to exclude historical test
identities; no historical-test labels, predictions or probabilities were read.
