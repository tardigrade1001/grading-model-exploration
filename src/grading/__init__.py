"""Reusable pieces for the grading-model exploration.

Everything that touches labels or fits statistics lives inside an sklearn
estimator so that cross-validation and the held-out split never share
information. See METHODOLOGY.md for the reasoning.
"""
