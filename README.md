# Bearing Fault Diagnostics — ML Pipeline & CWRU Validation

## Overview

This repository implements a machine learning pipeline for rolling-element bearing fault diagnosis, trained and cross-validated on the Case Western Reserve University (CWRU) bearing dataset. The pipeline is built with schema parity in mind so that ANSYS FEA-simulated data can be dropped into the same feature/label format for sim-to-real transfer experiments.

This README documents the **completed** CWRU + ML components. The ANSYS simulation branch (`src/features_extract/02_ansys_extraction.py`) is a stub and is not yet implemented — it is tracked as future work, not part of this document.

---

## Project Structure