# Annotation materials: coordinator instructions

This is a separate working-copy package. The frozen sample, original JSONL forms,
identity mapping and exclusion registry remain unchanged.

## Distribution

- Give Annotator A only `annotator_A/`; give Annotator B only `annotator_B/`.
- Give the adjudicator only `adjudicator/`, with A/B submissions later when review starts.
- Give trainees separate copies of `pilot/` when non-audit practice cases are approved.
- Keep this README, manifests, private identity mappings, sampling strata, weights
  and prior exposure information coordinator-only. Do not send the original sampling
  package or its protocol document to annotators as an evidence packet.
- Do not include scores, pseudo-labels, predicted labels, or the other annotator's
  decisions in independent annotation packets. This package contains no account lookup.

Use annotation_id to join future evidence packets privately. Supply the snapshot's
target period for each account and keep account identity separate from sampling
metadata. Do not expose user_key lookup tables. Evidence and account access details
needed for review can later be provided as authorized case packets.

## Working copies and returns

The CSVs preserve the frozen field names, blanks, IDs and row order. Each audit
form has 600 rows. Save submissions under new filenames, keeping these templates
immutable. Do not maintain competing CSV/XLSX versions of the same submission.
Check returned IDs for duplicates, omissions and additions before importing.
Join on annotation_id, never spreadsheet row number. Check label/access/confidence
values and evidence completeness; missing decisions must not become Human labels.

Keep A and B independent until both submissions are locked. Preserve every original
decision. Route disagreements to the adjudicator; keep unresolved cases Uncertain.
Never replace difficult, inaccessible or uncertain sampled users.

## Non-audit pilot

The ten PILOT rows have no assigned accounts, evidence or labels. Before assigning
any future practice account, check its canonical user_key with
`src.target_audit.assert_not_audit(keys)` and keep its mapping privately. Do not use
audit cases for training annotators. No pilot identity selection or evidence
collection has been performed by this preparation step. Pilot IDs alone cannot
prove a subsequently assigned account is outside the audit; check actual user_keys.

## Exclusions and next steps

Keep all 600 identities excluded from training, preprocessing fitting, representation
learning, threshold selection, calibration, pseudo-label training, model selection
and external-transfer training. Apply the existing fail-closed exclusion guard at
every future training/development identity boundary. These materials do not alter
the guard or grant permission to run models, collect evidence, or annotate accounts.

Later evaluation must use the frozen protocol. Annotation evidence must not be fed
back into model input features. This preparation performed no predictions, model
evaluation, training, pseudo-label generation or historical-test access.
