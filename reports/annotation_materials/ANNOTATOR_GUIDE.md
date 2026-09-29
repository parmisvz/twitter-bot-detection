# Account annotation guide

## Start here

Use your own form and work independently. Keep `annotation_id` and `rubric_version`
unchanged. Each row matches an evidence packet supplied by the coordinator.
If a packet or its time period is missing, ask the coordinator; do not guess the
account or assign a label. Do not view the other annotator's decisions.

Open the CSV as UTF-8, with columns treated as text. Keep the header names and
all rows. Save your completed working copy as UTF-8 CSV under a new filename.
Persian or English notes are acceptable. Do not edit the original blank template.
The CSV does not enforce dropdowns: use the exact label and status values below.

## Choose a label

| Label | Meaning |
| --- | --- |
| Human | Affirmative evidence supports predominantly human operation during the relevant period, such as sustained context-specific interaction and original personal responses. |
| Bot | Explicit automation evidence, or multiple corroborating indicators, supports substantially automated operation during the relevant period. Helpful or harmless automation still counts. |
| Uncertain | Evidence is insufficient, contradictory, inaccessible, from the wrong period, or shows unresolved mixed human/automated operation. |

A blank label means **not yet reviewed**. Uncertain means **reviewed but unresolved**.
Do not force a Human/Bot decision to complete the sheet. Confidence is `low`,
`medium`, or `high`, with a rationale; it is not a numerical probability.

## Evidence to record

Use dated evidence in the authorized packet: relevant posts and conversations,
archived profile information, explicit automation disclosures, and corroborated
posting patterns. Record sources, dates, a short factual summary, and both
automation and human-operation indicators. Explain conflicting observations.
Several indicators derived from the same repeated post are not independent evidence.

Popularity, political stance, anonymity, follower count, a default picture, high
activity, repetition, polished writing, or lack of posts alone cannot establish a
label. A personal-sounding message alone does not prove human operation. Do not
infer a label from a missing description or from account inaccessibility.

## Time and special cases

- **Temporal mismatch:** classify the period supported by the frozen snapshot,
  as specified by the coordinator. Record when you accessed evidence separately
  from when that evidence was created. Today's behavior does not establish past
  behavior. Describe a mismatch; use Uncertain if the target period is unsupported.
- **News, organization, or service accounts:** record this status separately.
  These accounts can be human-operated or automated. Their topic or institutional
  identity is not a Bot label. Mixed operation without sufficient evidence is Uncertain.
- **Inaccessible, deleted, or protected accounts:** record the access status.
  Adequate dated archived evidence may still support a label. Otherwise use Uncertain
  and explain the limitation. Do not substitute another account.

## Fill each reviewed row

| Field | What to enter |
| --- | --- |
| annotator_id | Your assigned reviewer identifier. |
| evidence_access_date | Date you reviewed evidence, preferably YYYY-MM-DD. |
| supported_period | Dates or interval supported by the evidence; say unknown if needed. |
| evidence_sources | Source references, packet references, or URLs supplied for the case. |
| evidence_types | For example archived profile, posts, conversation, automation disclosure. |
| evidence_summary | Brief observations with dates or source references. |
| automation_indicators | Observed automation evidence; write none observed when appropriate. |
| human_operation_indicators | Observed human-operation evidence; write none observed when appropriate. |
| limitations | Missing, conflicting, or weak evidence. |
| temporal_mismatch | yes / no / unknown, followed by an explanation when needed. |
| account_access_status | accessible / archived_only / inaccessible / deleted / protected / other. Explain other. |
| label | Human / Bot / Uncertain. |
| confidence | low / medium / high. |
| rationale | Why the evidence supports this label, including unresolved issues. |
| organization_news_service_status | Observed news/organization/service status, none observed, or unknown. |

Keep original independent decisions. An adjudicator resolves disagreements later;
you should not revise your judgment to match another reviewer.

## Practice template

Rows beginning `PILOT` are unassigned practice slots, not audit cases. Use separate
copies for each trainee after the coordinator supplies approved non-audit cases.
Practice labels never become audit labels. No accounts or evidence have been
assigned to this template.
