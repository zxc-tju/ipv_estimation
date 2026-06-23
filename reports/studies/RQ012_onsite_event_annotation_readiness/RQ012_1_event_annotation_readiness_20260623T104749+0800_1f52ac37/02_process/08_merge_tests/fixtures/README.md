# RQ012 W17b Merge Near-Duplicate Fixtures

All files in this directory are quarantined test fixtures.
They are not human annotations, not analysis labels, and not inputs
for agreement or event-IPV association. Simulated/fake values exist
only to prove the merge-validation gate rejects unsafe submissions.

The valid structural pair is blank by design and can pass only with
`structural_only=True` and `allow_test_fixtures=True`; it is still
not accepted as real human labels.
