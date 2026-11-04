from exasol.analytics.query_handler.query.select import LogSpan


def create_log_span_for_extractor(extractor):
    return LogSpan(name=f"{extractor.__class__.__name__} {extractor.name}")
