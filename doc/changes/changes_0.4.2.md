# 0.4.2 - 2026-10-19

## Summary

This release updates `virtualenv` and fixes vulnerabilities by updating dependencies.

## Security Issues

This release fixes vulnerabilities by updating dependencies:

| Dependency | Vulnerability | Affected | Fixed in |
|------------|---------------|----------|----------|
| gitpython | PYSEC-2026-3984 | 3.1.58 | 3.1.60 |
| gitpython | PYSEC-2026-3982 | 3.1.58 | 3.1.60 |
| gitpython | PYSEC-2026-3984 | 3.1.58 | 3.1.60 |
| gitpython | PYSEC-2026-3785 | 3.1.58 | 3.1.59 |
| gitpython | PYSEC-2026-3786 | 3.1.58 | 3.1.59 |
| gitpython | PYSEC-2026-3787 | 3.1.58 | 3.1.59 |
| gitpython | PYSEC-2026-3788 | 3.1.58 | 3.1.59 |
| gitpython | PYSEC-2026-3786 | 3.1.58 | 3.1.59 |
| gitpython | PYSEC-2026-3787 | 3.1.58 | 3.1.59 |
| gitpython | PYSEC-2026-3788 | 3.1.58 | 3.1.59 |
| gitpython | PYSEC-2026-3837 | 3.1.58 | 3.1.59 |
| gitpython | PYSEC-2026-3837 | 3.1.58 | 3.1.59 |
| nltk | PYSEC-2026-3736 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3739 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3738 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3735 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3737 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3741 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3733 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3749 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3748 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3751 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3749 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3752 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3741 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3739 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3738 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3790 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3790 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3733 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3748 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3737 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3735 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3736 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3752 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3751 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3868 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3871 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3867 | 3.10.2 | 3.10.3 |
| nltk | PYSEC-2026-3869 | 3.10.2 | 3.10.3 |
| tornado | PYSEC-2026-3928 | 6.5.7 | 6.5.8 |
| tornado | GHSA-chx6-46f5-w4vp | 6.5.7 | 6.5.9 |
| tornado | GHSA-c2m8-h5v5-343r | 6.5.7 | 6.5.9 |
| tornado | GHSA-3hv7-mjh2-fv65 | 6.5.7 | 6.5.9 |
| tornado | GHSA-wwv5-g3v4-889x | 6.5.7 | 6.5.8 |
| tornado | GHSA-8423-8fgw-73vq | 6.5.7 | 6.5.8 |
| urllib3 | PYSEC-2026-4177 | 2.7.0 | 2.8.0 |
| urllib3 | PYSEC-2026-4176 | 2.7.0 | 2.8.0 |
| urllib3 | PYSEC-2026-4175 | 2.7.0 | 2.8.0 |
| virtualenv | PYSEC-2026-4011 | 20.39.1 | 21.7.12 |
| virtualenv | PYSEC-2026-4012 | 20.39.1 | 21.7.11 |
| virtualenv | PYSEC-2026-4014 | 20.39.1 | 21.7.12 |
| virtualenv | PYSEC-2026-4013 | 20.39.1 | 21.7.13 |
| virtualenv | PYSEC-2026-4011 | 20.39.1 | 21.7.12 |
| virtualenv | PYSEC-2026-4012 | 20.39.1 | 21.7.11 |
| virtualenv | PYSEC-2026-4013 | 20.39.1 | 21.7.13 |
| virtualenv | PYSEC-2026-4014 | 20.39.1 | 21.7.12 |

## Refactorings

* Updated `virtualenv`

## Dependency Updates

### `main`

* Updated dependency `exasol-advanced-analytics-framework:0.7.0` to `0.9.0`
* Updated dependency `exasol-transformers-extension:5.0.0` to `5.2.1`
* Updated dependency `pydantic:2.13.4` to `2.14.1`
* Updated dependency `transformers:5.14.1` to `5.20.1`

### `dev`

* Updated dependency `boto3:1.43.64` to `1.43.114`
* Updated dependency `exasol-toolbox:10.4.0` to `11.0.0`
* Updated dependency `virtualenv:20.39.1` to `21.14.11`
