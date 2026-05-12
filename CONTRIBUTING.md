# Contributing to Generative Logic

All contributions to Generative Logic require a signed Contributor License Agreement (CLA).

See [`legal/CONTRIBUTOR_LICENSE_AGREEMENT.md`](legal/CONTRIBUTOR_LICENSE_AGREEMENT.md) for the full text and signature instructions.

## Reporting issues

Open issues against the public repository: <https://github.com/Generative-Logic/GL/issues>

## Pull requests

Pull requests are reviewed by the maintainer. Ensure:

- The CLA is signed (see above) — the project cannot accept code from unsigned contributors.
- Tests pass: `python verifier.py` against the current processed proof graph reports `0 failures` across every tag category.
- Documentation in [`docs/AGENT_SwDD.md`](docs/AGENT_SwDD.md) is updated for any change that touches prover semantics, provenance recording, scope handling, the proof-graph contract, MPL grammar, or config schema.

## License

Contributions are accepted under the project's dual license (AGPLv3 + commercial) — see [`LICENSE`](LICENSE) and [`legal/COMMERCIAL_LICENSE.md`](legal/COMMERCIAL_LICENSE.md).
