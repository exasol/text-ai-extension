# 0.4.0 - 2026-08-05

## Summary

Added support for Python 3.11 and 3.12, including the CI test matrix.
The Script Language Container (SLC) used to run
the extension's UDFs inside Exasol was also updated to run Python 3.12 internally.

## Security Issues

This release fixes vulnerabilities by updating dependencies:

| Dependency | Vulnerability | Affected | Fixed in |
|------------|---------------|----------|----------|
| click | PYSEC-2026-2132 | 8.2.1 | 8.3.3 |
| cryptography | PYSEC-2026-3553 | 48.0.0 | 49.0.0 |
| cryptography | PYSEC-2026-3554 | 48.0.0 | 49.0.0 |
| cryptography | GHSA-537c-gmf6-5ccf | 48.0.0 | 48.0.1 |
| gitpython | GHSA-mv93-w799-cj2w | 3.1.49 | 3.1.50 |
| gitpython | GHSA-2f96-g7mh-g2hx | 3.1.49 | 3.1.51 |
| gitpython | GHSA-956x-8gvw-wg5v | 3.1.49 | 3.1.51 |
| gitpython | GHSA-3rp5-jjmw-4wv2 | 3.1.49 | 3.1.53 |
| gitpython | GHSA-fjr4-x663-mwxc | 3.1.49 | 3.1.54 |
| gitpython | GHSA-6p8h-3wgx-97gf | 3.1.49 | 3.1.54 |
| gitpython | GHSA-r9mr-m37c-5fr3 | 3.1.49 | 3.1.54 |
| gitpython | GHSA-94p4-4cq8-9g67 | 3.1.49 | 3.1.55 |
| gitpython | GHSA-3f7w-8rr8-f37f | 3.1.49 | 3.1.57 |
| gitpython | GHSA-p538-c434-8v24 | 3.1.49 | 3.1.56 |
| idna | PYSEC-2026-215 | 3.11 | 3.15 |
| idna | PYSEC-2026-215 | 3.11 | 3.15 |
| msgpack | GHSA-6v7p-g79w-8964 | 1.1.2 | 1.2.1 |
| nltk | PYSEC-2026-3583 | 3.9.4 | 3.10.0 |
| nltk | PYSEC-2026-3582 | 3.9.4 | 3.10.0 |
| nltk | PYSEC-2026-3584 | 3.9.4 | 3.10.0 |
| pip | PYSEC-2026-196 | 26.1.1 | 26.1.2 |
| pip | PYSEC-2026-196 | 26.1.1 | 26.1.2 |
| pyasn1 | PYSEC-2026-3456 | 0.6.3 | 0.6.4 |
| pyasn1 | PYSEC-2026-3457 | 0.6.3 | 0.6.4 |
| pyasn1 | PYSEC-2026-3455 | 0.6.3 | 0.6.4 |
| pyasn1 | PYSEC-2026-3455 | 0.6.3 | 0.6.4 |
| setuptools | PYSEC-2026-3447 | 81.0.0 | 83.0.0 |
| setuptools | PYSEC-2026-3447 | 81.0.0 | 83.0.0 |
| soupsieve | PYSEC-2026-3072 | 2.8.3 | 2.8.4 |
| soupsieve | PYSEC-2026-3071 | 2.8.3 | 2.8.4 |
| torch | PYSEC-2025-194 | 2.11.0 | 2.13.0 |
| tornado | PYSEC-2026-3387 | 6.5.5 | 6.5.6 |
| tornado | PYSEC-2026-3388 | 6.5.5 | 6.5.6 |
| tornado | PYSEC-2026-3389 | 6.5.5 | 6.5.6 |
| tornado | GHSA-pw6j-qg29-8w7f | 6.5.5 | 6.5.7 |
| transformers | PYSEC-2026-2288 | 4.57.6 | 5.0.0 |
| transformers | PYSEC-2026-2289 | 4.57.6 | 5.3.0 |
| transformers | PYSEC-2026-2290 | 4.57.6 | 5.5.0 |
| urllib3 | PYSEC-2026-142 | 2.6.3 | 2.7.0 |
| urllib3 | PYSEC-2026-142 | 2.6.3 | 2.7.0 |
| urllib3 | PYSEC-2026-141 | 2.6.3 | 2.7.0 |
| uv | GHSA-4gg8-gxpx-9rph | 0.11.9 | 0.11.15 |

## Refactoring

* Updated exasol-toolbox to 8.2.0
* Re-enabled check-workflows in `checks.yml`, updated exasol-toolbox to 10.0.0, and updated dependencies

## Dependency Updates

### `main`

* Updated dependency `exasol-python-extension-common:0.13.0` to `0.16.0`
* Updated dependency `exasol-transformers-extension:3.0.0` to `5.0.0`
* Updated dependency `pydantic:2.12.5` to `2.13.4`
* Updated dependency `transformers:4.57.6` to `5.14.1`

### `dev`

* Updated dependency `boto3:1.42.78` to `1.43.61`
* Updated dependency `exasol-toolbox:6.1.1` to `10.0.0`
* Updated dependency `pytest-exasol-backend:1.4.1` to `1.5.1`
* Updated dependency `pytest-exasol-slc:1.0.1` to `1.1.1`