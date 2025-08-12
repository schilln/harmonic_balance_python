"""

Assume periodic functions of the form

exp(1j * k * omega * t)

and a time-domain system dynamics equation of the form

Mx''(t) + Cx'(t) + Kx(t) + f_nl(x(t), x'(t)) = f(omega, t).

Notation
--------
k
    Frequency index
omega
    Fundamental frequency
x
    State in time domain
z
    State in frequency domain
M
    Mass matrix
C
    Damping matrix
K
    Stiffness matrix
f_nl
    Force function depending on state, potentially nonlinearly
f
    External force function
NH
    Assumed highest index of periodic basis functions / harmonics
    [1, exp(1j * 1 * omega * t), ..., exp(1j * NH * omega * t)]
n
    Number of degrees of freedom in the system
N
    Number of points to sample when computing alternating frequency/time
    Should have N >= 2 * NH + 1.
A
    Matrix describing linear dynamics in frequency domain
"""
