# Seismic Processing Pipeline

A lightweight Python pipeline for 2D marine/ground seismic data processing, from raw SEG-Y loading through NMO/stacking and post-stack Kirchhoff time migration.

## ⚠️ Key Assumption: Flat/Sub-Horizontal Layering

This pipeline assumes **near-flat geology (dip < ~20°)**. Under this assumption, the **CDP is treated as a proxy for the CMP** (i.e., reflection points are assumed to fall directly below the CDP location, with no dip-induced lateral shift).

This simplification underlies:
- Hyperbolic NMO correction (`nmo_correction_and_stacking.py`)
- Semblance-based velocity analysis (`velocity_field.py`)
- Post-stack (zero-offset) Kirchhoff migration (`migration.py`)

For steeper structural dips, this assumption breaks down and prestack time/depth migration with proper dip-moveout (DMO) handling would be required instead.

## Modules

| File | Purpose |
|---|---|
| `loading_and_qc.py` | SEG-Y ingestion (via `segyio`), header extraction, QC (dead/weak trace detection) |
| `signal_processing.py` | Bandpass filtering, despiking, spherical divergence correction, AGC, amplitude balancing |
| `static_correcion.py` | Water-column static correction, residual stack static correction (cross-correlation) |
| `velocity_field.py` | Interactive semblance velocity analysis/picking, velocity field interpolation |
| `nmo_correction_and_stacking.py` | NMO correction, CMP stacking, full-line velocity field build & stack |
| `wiener_deconvolution.py` | Auto-parametrized spiking (Wiener) deconvolution |
| `migration.py` | Coordinate/CDP spacing utilities, post-stack Kirchhoff time migration |

## Typical Workflow

1. Load & QC SEG-Y (`loading_and_qc.py`)
2. Signal conditioning: filtering, deconvolution, gain (`signal_processing.py`, `wiener_deconvolution.py`)
3. Statics correction (`static_correcion.py`)
4. Velocity analysis & NMO/stack (`velocity_field.py`, `nmo_correction_and_stacking.py`)
5. Residual stack statics (`static_correcion.py`)
6. Post-stack time migration (`migration.py`)

## Requirements

`numpy`, `scipy`, `segyio`, `ipywidgets`, `matplotlib`

## Disclaimer

Educational/research-oriented implementation. Not validated for production processing workflows.
