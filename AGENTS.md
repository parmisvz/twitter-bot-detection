# Research audit exclusion requirement

The target_audit_v1 sample is reserved exclusively for independent evaluation.
Before any future fitting, model/threshold selection, calibration, representation
learning, pseudo-label training or external-transfer training, call
`src.target_audit.assert_not_audit(keys)` on ALL participating user identities.
The guard must fail closed if its pinned registry is unavailable or altered.
Do not bypass it, resample the audit, use audit labels for development, or include
audit identities in unsupervised/domain-adaptation training. Existing legacy
scripts are not automatically protected: add the guard before executing them.
M4/P1 remains frozen. Historical-test evaluation, external downloads/training,
audit annotation, model predictions and evidence collection require separate
user authorization. Preserve the unrelated Notebook 06 working-tree edit.
