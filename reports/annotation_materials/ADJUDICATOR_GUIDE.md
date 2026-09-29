# Adjudication instructions

Use the same account rubric as the independent annotators. The adjudication CSV
contains blank rows for all 600 IDs; it does not mean every case has a disagreement.
The coordinator will later supply the review queue and original A/B decisions.

Review every disagreement, including disagreements involving Uncertain. Check
evidence completeness for Human/Human and Bot/Bot agreements before accepting them.
Uncertain/Uncertain remains unresolved unless adequate evidence supports resolution.
Do not force a binary label by majority vote. Preserve Uncertain where evidence
remains insufficient or temporally mismatched.

Record your identifier, reviewed evidence, final label/confidence and rationale.
Use `adjudication_rationale` to explain the competing interpretations and why the
evidence resolves them or does not. A blank adjudication row means not processed.
Never overwrite either independent annotator's original decisions. Keep an immutable
copy of those submissions and save your decisions separately.

Do not use model outputs or sampling information to settle an annotation decision.
