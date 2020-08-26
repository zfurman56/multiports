import numpy as np
from sympy import *

# Get port name from port ID (for debugging purposes)
def port_name(N, a):
    z = a%(N*5)
    return chr(ord('A')+(z//5)) + str((z%5)+1)

# Produces multiport transition matrix, given number of ports N and mirror
# phase phh_m
def transition_matrix(N, mirror_phases):
    U = np.zeros((N*5,N*5), dtype=complex)

    # Set amplitude to transition from basis state a_n to basis state b_m
    def set_amplitude(a, n, b, m, A):
        U[((b*5)+m)%(N*5), ((a*5)+n)%(N*5)] = A

    # Amplitudes for beam splitter transmission and reflectance
    t = 1/np.sqrt(2)
    r = 1j/np.sqrt(2)

    # Amplitudes for mirror reflections
    m = np.exp(1j*np.array(mirror_phases))

    for i in range(N):
        set_amplitude(i, 0, i+1, 2, r)
        set_amplitude(i, 0, i-1, 1, t)
        set_amplitude(i, 1, i,   4, r)
        set_amplitude(i, 1, i,   3, t*m[i])
        set_amplitude(i, 2, i,   3, r*m[i])
        set_amplitude(i, 2, i,   4, t)
        set_amplitude(i, 3, i+1, 2, t)
        set_amplitude(i, 3, i-1, 1, r)
        set_amplitude(i, 4, i,   4, 1)

    return U

# Produces multiport scattering matrix
def scattering_matrix(N, mirror_phases):
    U_trans = transition_matrix(N, mirror_phases)
    U_scat = np.around(np.linalg.matrix_power(U_trans, 1000), 4)
    return U_scat[4:(N*5):5, 0:(N*5):5]

def find_closed_form(N):
    U = zeros(N*5, N*5)

    # Set amplitude to transition from basis state a_n to basis state b_m
    def set_amplitude(a, n, b, m, A):
        U[((b*5)+m)%(N*5), ((a*5)+n)%(N*5)] = A

    # Amplitudes for beam splitter transmission and reflectance
    t = 1/sqrt(2)
    r = I/sqrt(2)

    # Amplitude for mirror reflection
    phi_m = symbols('phi_m')
    m = exp(I*phi_m)

    for i in range(3):
        set_amplitude(i, 0, i+1, 2, r)
        set_amplitude(i, 0, i-1, 1, t)
        set_amplitude(i, 1, i,   4, r)
        set_amplitude(i, 1, i,   3, t*m)
        set_amplitude(i, 2, i,   3, r*m)
        set_amplitude(i, 2, i,   4, t)
        set_amplitude(i, 3, i+1, 2, t)
        set_amplitude(i, 3, i-1, 1, r)
        set_amplitude(i, 4, i,   4, 1)

    return U

