"""A small CNN that gives a kick onset probability for each 10 ms frame.

Input: log-magnitude spectrograms at three window sizes (23, 46 and 93 ms),
in mel bands from 27.5 Hz to 16 kHz. A 2D convolution front end reads each
frame's spectrum, and a dilated temporal convolution network adds about
+/-420 ms of context (about one beat on each side at 160 BPM).
"""
