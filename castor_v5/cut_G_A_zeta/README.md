# Vertical cut on Re s = 1/2

Open G_A_zeta_cut_v2.html in a browser to view the self-contained
interactive chart. It includes the curves of G and A and the
complex-plane curve of their difference.

From this directory, run python3 vertical_cut_portable.py to regenerate
vertical_cut.json and vertical_cut_output.txt. The script imports cp_eval.py
from the parent directory and uses NumPy, SciPy, and mpmath. It writes its
outputs to the current directory, including replacing the supplied log.
The HTML is supplied as a finished file.

The recorded run reports 30 local minima of |G-A| on the grid with
step 0.01 for 0.5 <= Im s <= 100. After numerical refinement, 29
minima met the threshold |G-A| <= 1e-6. Their ordinates agreed with
the corresponding computed zeta ordinates within 6e-7.
