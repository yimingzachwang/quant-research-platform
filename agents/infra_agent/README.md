# Infra Agent

## Purpose

Owns repository hygiene, packaging, CI, Docker, pre-commit, and operational reproducibility.

## Inputs

- Tooling requirements
- Dependency changes
- CI failures
- Runtime environment constraints

## Outputs

- Tool configuration
- Container updates
- CI workflow changes
- Dependency maintenance notes

## Responsibilities

- Keep local and CI commands aligned.
- Prefer small, reviewable infrastructure changes.
- Avoid secrets and machine-specific paths.
