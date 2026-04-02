# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public GitHub issue.

Instead, report it by emailing the maintainers via a [GitHub private vulnerability report](https://github.com/airupt/stubllm/security/advisories/new).

Include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

You will receive a response within 48 hours. We aim to release a patch within 7 days of confirmation.

## Notes

stubllm is a **test utility** designed to run locally or in CI. It should never be exposed to the public internet. The server binds to `127.0.0.1` by default for this reason.
