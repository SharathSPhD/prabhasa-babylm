# Saṃsādhanī setup (Pāṇinian generator)

The Pāṇinian synthetic-Sanskrit stream is produced by **Saṃsādhanī**, the
computational-Sanskrit toolset from the Department of Sanskrit Studies,
University of Hyderabad (<https://sanskrit.uohyd.ac.in/scl/>). It is **not** a
pip package; it is provisioned on the DGX Spark and bound through
`psalm.infrastructure.generators.samsadhani.SamsadhaniGenerator`.

## Why an external adapter

The generator gives us, per sentence, the surface form **plus** the gold kāraka
parse and derivation — the H1 signal. The adapter parses that output into
`AnnotatedSentence(text, karaka_parse, derivation)`. Until the tool is
provisioned, the adapter refuses to run (`SamsadhaniNotConfiguredError`) rather
than fabricate data, per the program's integrity rules.

## Provisioning steps (on the Spark)

1. Obtain the Saṃsādhanī / SCL generator per the UoHyd licensing terms and place
   it under a host path, e.g. `~/tools/samsadhani`.
2. Verify it runs and emits sentences with kāraka annotations.
3. Point the adapter at it: `SamsadhaniGenerator(install_root="~/tools/samsadhani")`.
4. Record recorded-output fixtures under `tests/fixtures/samsadhani/` so the
   parsing adapter is unit-tested without the live tool.
5. Measure diversity/coverage (`psalm.domain.data.diversity`) on a sample before
   committing the pre-pretraining budget — this is the Phase 1 go/no-go.

## License note

Respect the UoHyd license for any redistributed generator output. The published
HF corpus must include a license manifest (Phase 1 contract artifact).
