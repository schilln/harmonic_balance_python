# Harmonic balance

This is an implementation of the harmonic balance method (HBM) using exponential basis functions instead of trigonometric functions as is common in other implementations.
Head on over to [`theory/theory.pdf`](theory/theory.pdf) to learn more.
Note however that the implementation needs work.
It struggles to compute nonlinear frequency response (NLFR) curves, as demonstrated by the Duffing notebooks in the [`experiment`](experiment/) directory.
In addition, little time has been spent on code optimization.
(See the "Continuation" section of the theory document for more on NLFR curves.)
