# Reproducibility notes

The 200 and 400 step results were collected using the toy GRU-based backbone.
The 200-step checkpoint weights are not included in Git (the local archive includes them).
The continuation script requires those checkpoint files under runs/validation200/.
A 2D random slice cannot establish global loss-landscape flatness.

Step 200: baseline validation 45%, latent validation 35%.
Step 400: baseline validation 35%, latent validation 45%.
Only 20 held-out examples and one seed were used.
