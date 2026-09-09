# ADR 0001: Record architecture decisions

**Status:** Accepted · **Date:** 2026-09-07

## Context

Most ML repositories preserve what was built and lose why. Six months later nobody
remembers whether the feature store was a requirement or a preference, and the
reasoning gets re-litigated from scratch.

## Decision

Record every significant decision as a short ADR in `docs/adr/`, numbered
sequentially, never edited after acceptance - superseded by a new ADR instead.

An ADR is warranted when a decision is expensive to reverse, when a reasonable
engineer would choose differently, or when the reasoning is not visible in the code.

## Consequences

A reviewer can read the ADRs and understand the shape of the system without reading
the code. The cost is a few minutes per decision.
