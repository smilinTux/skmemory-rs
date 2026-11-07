# Security Policy

`skmem` is the planned engine behind agent memory, so it will eventually handle
private data. At 0.1.0 it contains no engine and handles no data.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | Yes (bootstrap) |

## Reporting a vulnerability

Report privately. Do **not** open a public issue for a security report.

- Use GitHub private vulnerability reporting on `github.com/smilinTux/skmemory-rs`
  (Security, then Report a vulnerability).

Include the affected version, a reproduction, and the impact you observed. You will
get an acknowledgement, and reporters are credited unless they ask otherwise.

## Ground rules for contributors

- No secrets, keys, tokens, hostnames of private infrastructure or personal data in
  the tree, in fixtures, or in commit history. The secret-scan workflow scans the
  full history on every push.
- No real memory data in fixtures. Use synthetic records.
- Dependencies are kept minimal. A new dependency needs a stated reason in the PR.
- No `unsafe` without a comment stating the invariant it relies on.
