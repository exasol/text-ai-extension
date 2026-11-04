# 0.1.0 - 2025-07-14

## Features

* Added initial implementation of the KeywordExtractor
* Made the algorithms stackable, i.e. enabled the user to create pipline
  with two or more algorithms chained together
* Handled the newlines and other space characters properly in the TokenizedTextSearch
* Excluded unneeded components from the spaCy pipline in the TokenizedTextSearch
* Added the default value for the `algorithm_name` parameter of all algorithms
* Defined PipelineStage Interface
* Provided an SQLStageQueryHandler for a UDFRunner
* Changed the interface of the UDFFactory, taking a UDFName instead of two separate strings
* Provided a way to link the output columns of one stage with the input columns of another stage
* Created the KeywordExtractorConverter
* Expanded the Converter interface by adding the context. Merged Converter and TypedConverter
* Added Recording the algorithm settings
* Changed the behaviour when an algorithm doesn't have keys
* Added named entity recognition algorithm
* Added topic identification algorithm
* Added BranchExtractor and related classes
* Added exclude_from_setup property to the algorithm interface
* Added definition of Extraction as pydantic class
* Added all remaining NLP extractors and converters
* Added empty implementation of `Extraction.run()` and `ExtractionRunner` protocol
* Added no-database integration tests for the Named Entity Recognition based on a Huggingface transformers model
* Added no-database integration tests for the Topic Identification based on a Huggingface transformers model
* Implemented class `NoOpPathLike` enabling to specify a temporary `BucketFsLocation` that does not actually access the BucketFS but allows `joinpath()` and other introspective methods
* Added on-database integration tests for the Named Entity Recognition and Topic Classification
* Added implementation to `PythonExtractionRunner`
* Optimised the output of the UDF Query Handler by adding the DISTRIBUTE BY and PARTITION BY specifiers.
* Normalised the output of the UDF Query Handler.
* Added initial implementation to `OrchestratorQueryHandler`
* Move stage graph handling to dedicated query handler `ExtractionQueryHandler`
* Simulated different number of iterations when testing the `OrchestratorQueryHandler`
* Added phrase_column_flags parameter to the TextSearch
* Added a simple column aggregation method for algorithms that do not support the error message.
* Exposed the batch_size to the user.
* Improved the normalisation performance in case some of the keys are normalisable.
* Added SLC Deployer with run function for deploying local SLC.
* Added an on-database integration test for the Source Table Extractor Query Handler.
* Added an on-database integration test for the UDFAlgoQueryHandler.
* Handled NULLs in the source table texts.
* Added mapping tables to the result returned by the Source Table Query Handler
* Took apart the integration conftest.py
* Added Source Table Graph Builder.
* Added SpacyTokenizer UDFAlgo
* Added parallelism_per_node option to the `NLPExtractor`.
* Added span predicate functions.
* Added audit information to the query handlers.
* Created a nox task for zipping and uploading the SLC and the project wheel.
* Disabled autocommit in `PythonExtractionRunner`
* Prepared end-2-end tests.
* Added another bunch of text scalar functions.
* Prepended "tmp_" to prefix for temporary database objects
* Created a graph builder for the tokenizer extractor.
* Added `AuditQueryHandler` to `PythonExtractionRunner`
* Added the source table views.
* Created the `TokenPositionQueryHandler`.
* Added a function creation query to the `TokenPositionQueryHandler`
* Added TextEnvelope class for packing any TextOutput.
* Added rollback in case of exceptions in `set_autocommit_off`
* Added the `name` field to the BaseExtractor
* Added Co-Occurrence Query Handler.
* Added another integration test for the Co-Occurrence view.
* Added the Filter and Splitter extractors.
* Added the extractors and graph builders for the views.
* Made `ListCollectorQueryHandler` creating a flattened list of inputs.
* Added the `OutputExtractor` with the `db_name` parameter.

## Documentation

* Updated documentation

## Refactorings

* Fixed circular import
* Cleaned up directory structure
* Moved tests to match package structure
* Split the extractors.py into separate extractors modules
* Refactored algorithms extracting and reusing common code
* Refactored validation of source extractors
* Moved package `extraction` to top-level
* Replaced DummyExtractor by NamedEntityExtractor
* Enabled the integration tests
* Used SqlExecutor in method run of ExtractionRunner
* Renamed "ConverterXXX" to "GraphBuilderXXX"
* Made the KeywordExtractor using a Huggingface model
* Renamed "TopicIdentification" to "TopicClassification"
* Moved `extraction_runners` into separate package `internal`
* Resolved circular dependency between `Extraction` and `ExtractionRunner`
* Enabled mypy.
* Renamed `db_output` parameter in the query handlers.
* Improved the construction of output table names.
* Moved internal packages to the folder `impl`.
* Added options to pytest to simplify debugging
* Refactored the consistency check of the algorithm assembly columns.
* Reversed ranking for the topic classifier.
* Made all column names uppercase.
* Added end-to-end-test for multiple sources in different pipelines
* Added end-to-end test for multiple sources fed into a single NLP algorithm
* Added end-to-end test for change data capture
* Updated to poetry 2.1.2 and exasol-toolbox 1.1.0
* Created input/output column interface for the algorithms.
* Added the Pipeline algorithm.
* Refactored UDFRunner.
* Refactored (deleted) the algorithm stage graph outputs.
* Refactored (deleted) the algorithm stage graph inputs.
* Refactored algorithm stage graph builders.
* Removed redundant code.
* Added end-to-end test for optimization parameters `batch_size` and `parallelism_per_node`
* Refactored the TokenPosition Query Handler.
* Split up SourceTableQueryHandler

## Bug Fixing

* Changed the imports in the python_extraction_runner.py avoiding potential cyclic references.
* Fixed uploaded zip for prereleases and invalidate CloudFront.
* Fixed incorrect handling of NULLs by the `UDFAlgoQueryHandler`.
* Added handling a special case of a KeyBERT result for a single document.
* Fixed incorrect handling of the no-tokens case by the `HftNamedEntityRecognition`.
* Fixed the setup comparison in the `UDFAlgoQueryHandler`.
* Fixed the setup comparison again and added an indempotency test.
* Fixed bug in AAF by updating dependencies
* Fixed the bug in the script_deployer and added the tests.
* Added handling of no-result cases by the algorithms.
* Added temporary fix for an invalid size of string error.
* Enclosed inputs for the views, Filter and Splitter in a ListInputOutput.
* Replaced `datasource` with `primary_view`
* Fix missing RUN_ID in audit log by adding log spans 
