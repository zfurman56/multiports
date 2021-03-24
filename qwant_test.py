# Tutorial 2.4.2. Closed systems
# ==============================
#
# Physics background
# ------------------
#  Fock-darwin spectrum of a quantum dot (energy spectrum in
#  as a function of a magnetic field)
#
# Kwant features highlighted
# --------------------------
#  - Use of `hamiltonian_submatrix` in order to obtain a Hamiltonian
#    matrix.

from cmath import exp
import numpy as np
import kwant

# For eigenvalue computation
import scipy.sparse.linalg as sla
from scipy.linalg import logm

# For plotting
from matplotlib import pyplot
import matplotlib.pyplot as plt


def make_system(a=1, t=1.0, r=10):
    # Start with an empty tight-binding system and a single square lattice.
    # `a` is the lattice constant (by default set to 1 for simplicity).

    lat = kwant.lattice.square(a, norbs=1)

    sys = kwant.Builder()

    for i in range(r):
        for j in range(r):
            # On-site Hamiltonian
            sys[lat(i, j)] = 4

    for i in range(r):
        for j in range(r):
            # Hopping in y-direction
            sys[lat(i, j), lat(i, (j - 1) % r)] = -1j

            # Hopping in x-direction
            sys[lat(i, j), lat((i - 1) % r, j)] = -1j

    # It's a closed system for a change, so no leads
    return sys

def plot_vec(vec, width, height):
    grid = np.reshape(vec, (width, height))
    plt.imshow(np.real(grid), cmap='gray', vmin=-0.5, vmax=0.5)
    plt.show()

def main():
    sys = make_system()

    # Check that the system looks as intended.
    kwant.plot(sys)

    # Finalize the system.
    sys = sys.finalized()

    # Plot an eigenmode of a circular dot. Here we create a larger system for
    # better spatial resolution.
    sys = make_system().finalized()

    # Calculate the wave functions in the system.
    ham_mat = sys.hamiltonian_submatrix()
    Tx = np.kron(np.roll(np.eye(10), 1, axis=1), np.eye(10))
    Ty = np.kron(np.eye(10), np.roll(np.eye(10), 1, axis=1))
    evals, evecs = np.linalg.eig(ham_mat+logm(Tx)+logm(Ty))

    for n in range(3):
        plot_vec(evecs[:, n], 10, 10)
    plot_vec(evecs[:, 78], 10, 10)


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()

