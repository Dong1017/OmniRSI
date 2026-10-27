# Isolated HTTP diagnostic fixture

This temporary fork-only branch and its draft PR provide a conversation target
for the owner-authorized HTTP diagnostic in [the diagnostic issue](https://github.com/Dong1017/OmniRSI/issues/3).
The issue records the frozen software provenance, request budget and limits.

The probe reuses the trusted publisher HTTP client and existing Actions token
permission baseline. It compares one comment creation and update on a diagnostic
Issue and on this diagnostic PR. It does not run a Skill Eval comparison or claim
an evaluation result. There is no execution authorization or model/GPU call.

The test uses an isolated branch push event. The upstream failing run used
`pull_request_target`; a successful fork probe cannot establish the upstream
root cause. Keep all test evidence and leave the fork's default branch unchanged.
