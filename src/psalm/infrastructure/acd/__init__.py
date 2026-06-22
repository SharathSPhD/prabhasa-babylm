"""Lane-C ACD (Active Circuit Discovery) infrastructure for PSALM/Prabhāsa.

Circuit-level activation capture and minimal-pair analysis for training-time
localization of weak paradigm circuits (NPI licensing, filler-gap, islands).

NOTE: ACD inference-time steering ≈ random; used as TRAINING-TIME LOCALIZER
(find circuits → later circuit-targeted fine-tuning/data-aug), NOT steering tool.
"""
