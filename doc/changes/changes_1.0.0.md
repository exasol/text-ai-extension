# 1.0.0 - 2026-11-04

## Summary

This release open sources the formally closed source Exasol Text AI Extension and drops support for Python 3.10.

The [changelog for the former releases](https://github.com/exasol/text-ai-extension/blob/main/doc/changes) is included, though.

## Features

* #1: Added initial repository content

## Bugfixes

* Fixed SLC upload to AWS S3

## Refactorings

* Updated exasol-toolbox to 12.0.0

## Dependency Updates

### `main`

* Added dependency `exasol-advanced-analytics-framework:0.9.0`
* Added dependency `exasol-keyphrase-vectorizers:1.0.0`
* Added dependency `exasol-python-extension-common:0.16.0`
* Added dependency `exasol-transformers-extension:5.2.1`
* Added dependency `keybert:0.8.5`
* Added dependency `numpy:1.26.4`
* Added dependency `pydantic:2.14.1`
* Added dependency `spacy:3.8.6`
* Added dependency `transformers:5.20.2`

### `dev`

* Added dependency `boto3:1.43.118`
* Added dependency `exasol-toolbox:12.0.0`
* Added dependency `exasol-udf-mock-python:0.5.0`
* Added dependency `pytest-exasol-backend:1.5.1`
* Added dependency `pytest-exasol-extension:1.0.1`
* Added dependency `pytest-exasol-slc:1.1.1`
* Added dependency `toml:0.10.2`
* Added dependency `virtualenv:21.14.13`
