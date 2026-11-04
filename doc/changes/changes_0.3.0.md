# 0.3.0 - 2026-05-07

This release fixes vulnerabilities by updating dependencies:

| Dependency   | Affected | Vulnerabilities                                | Fixed in | Updated to |
|--------------|----------|------------------------------------------------|----------|------------|
| black        | 26.1.0   | CVE-2026-32274                                 | 26.3.1   | 26.3.1     |
| cryptography | 45.0.7   | CVE-2026-26007, CVE-2026-34073, CVE-2026-39892 | 46.0.7   | 48.0.0     |
| nltk         | 3.9.2    | CVE-2025-14009, CVE-2026-33230                 | 3.9.3    | 3.9.4      |
| pillow       | 12.0.0   | CVE-2026-25990                                 | 12.1.1   | (removed)  |
| pyasn1       | 0.6.2    | CVE-2026-30922                                 | 0.6.3    | 0.6.3      |
| requests     | 2.32.5   | CVE-2026-25645                                 | 2.33.0   | 2.33.0     |
| tornado      | 6.5.4    | CVE-2026-31958, GHSA-78cv-mqj4-43f7            | 6.5.5    | 6.5.5      |
| gitpython    | 3.1.46   | CVE-2026-42215, CVE-2026-42284                 | 3.1.47   | 3.1.49     |
| ptest        | 9.0.2    | CVE-2025-71176                                 | 9.0.3    | 9.0.3      |
| pip          | 26.0.0   | GHSA-jp4c-xjxw-mgf9                            | 26.1.0   | 26.1.1     |
| uv           | 0.11.5   | GHSA-pjjw-68hj-v9mw                            | 0.11.6   | 0.11.9     |

## Security Issues

* Fixed vulnerabilities by updating dependencies
* Fixed vulnerabilities by updating dependencies

## Refactorings

* Updated to PTB 6.0.0 & re-locked cryptography, filelock, nltk, pillow, pip, pyasn1, tornado, urllib3
* Used black & isort to automatically format files, re-locked pynacl & pyasn1, updated to PTB 6.1.0, & activated CI format checking
* Fixed circular import.
* Removed formatting overrides
* Enabled `ruff` formatting `F401`

## Dependency Updates

### `main`

* Updated dependency `exasol-advanced-analytics-framework:0.6.2` to `0.7.0`
* Updated dependency `exasol-python-extension-common:0.12.0` to `0.13.0`
* Updated dependency `pydantic:2.12.3` to `2.12.5`
* Updated dependency `transformers:4.57.1` to `4.57.6`

### `dev`

* Updated dependency `boto3:1.40.65` to `1.42.78`
* Updated dependency `exasol-toolbox:1.1.0` to `6.1.1`
* Updated dependency `pytest-exasol-backend:1.2.2` to `1.4.1`
* Updated dependency `pytest-exasol-extension:0.2.4` to `1.0.1`
* Updated dependency `pytest-exasol-slc:0.4.2` to `1.0.1`
* Added dependency `virtualenv:20.39.1`
