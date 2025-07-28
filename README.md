# HomeWearable_ICG_PointDetection
Accurate BCX point detection in ICG signals using ECG-R anchoring and multi-stage denoising (wavelet + EEMD + LMS).
## ICG BCX Point Detection with ECG Anchoring and Advanced Denoising

This project implements a robust and physiologically grounded pipeline for identifying key characteristic points—B (aortic valve opening), C (peak ventricular contraction), and X (aortic valve closure)—in impedance cardiography (ICG) signals. Leveraging ECG-derived R-peak anchors, it segments cardiac cycles and applies a multi-stage denoising cascade (wavelet transform → EEMD → LMS adaptive filter) to enhance signal clarity in wearable or home environments.

Key features:
- ✅ ECG R-peak guided ICG segmentation
- ✅ Multi-stage denoising: db4/sym8 wavelets + EEMD + LMS filter
- ✅ BCX detection based on physiologically timed windows
- ✅ Accurate average beat reconstruction and visualization
- ✅ Suitable for PEP/LVET/SV/CO calculation

This tool supports research and prototyping of wearable cardiac monitoring systems, especially in noisy or ambulatory settings.
