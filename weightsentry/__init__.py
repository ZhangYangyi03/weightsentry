"""weightsentry - static supply-chain audit for local model artifacts.

Pure standard library on purpose: an audit tool must not import the very
framework (torch/pickle) whose loaders are the attack surface it inspects.
"""
__version__ = "0.1.0"
