"""STRATA vision core.

A spatial version-control engine built on OpenCV 5. It registers two captures
of the same place into a shared canonical frame and reports an object-level
diff of what changed between them.

Modules
-------
config    : pipeline configuration objects
features  : pluggable local-feature backends (neural + classic)
geometry  : robust registration into a canonical frame
diff      : canonical change detection and change-region extraction
events    : change regions resolved into typed, explained events
coverage  : capture-coverage and uncertainty mapping for active perception
pipeline  : end-to-end orchestration
synth     : synthetic scene generator with ground truth, for tests
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
