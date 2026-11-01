"""Pretrained reaction tokens and protein residues, interacting before pooling.

This package is separate from the original graph and sequence experiments.
Feature extraction freezes both pretrained encoders; the trainable model uses
their full position-wise states, never a fabricated expansion of pooled caches.
"""
