import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Convert complex grid to
def colorize(z):
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + pi)  / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    return c

def create_multiport_grid(U, width, height):
    grid = np.zeros((width, height, 4, 4), dtype=complex)
    grid[:, :] = U
    return grid

def create_multiport_barrier_grid(U1, U2, width, height):
    grid = np.zeros((width, height, 4, 4), dtype=complex)
    grid[:(width//2), :] = U1
    grid[(width//2):, :] = U2
    return grid

def create_vertical_multiport_barrier_grid(U1, U2, width, height):
    grid = np.zeros((width, height, 4, 4), dtype=complex)
    grid[:, :(height//2)] = U1
    grid[:, (height//2):] = U2
    return grid

def create_diagonal_multiport_barrier_grid(U1, U2, size):
    grid = np.zeros((size, size, 4, 4), dtype=complex)
    iu = np.triu_indices(size)
    grid[:, :] = U1
    grid[iu] = U2
    return grid

def create_multiport_barrier_grid_patch(U1, U2, width, height):
    grid = np.zeros((width, height, 4, 4), dtype=complex)
    grid[:, :] = U1
    grid[(width//4):width-(width//4), (height//4):height-(height//4)] = U2
    return grid

def periodic_boundary_time_evolution_from_multiport_grid(U_grid):
    width, height, _, _ = U_grid.shape
    U = np.zeros((width*height*4, width*height*4), dtype=complex)
    for i1 in range(width):
        for j1 in range(height):
            idx1 = 4*(i1*height + j1)
            for direction, (i2, j2) in enumerate([(i1, j1+1), (i1+1, j1), (i1, j1-1), (i1-1, j1)]):
                i2, j2 = (i2%width), (j2%height)
                idx2 = (4 * (i2*height + j2))
                U[idx1+direction, idx2:idx2+4] = U_grid[i2, j2, direction]
    return U

def closed_boundary_time_evolution_from_multiport_grid(U_grid):
    width, height, _, _ = U_grid.shape
    U = np.zeros((width*height*4, width*height*4), dtype=complex)
    for i1 in range(width):
        for j1 in range(height):
            idx1 = 4*(i1*height + j1)
            for direction, (i2, j2) in enumerate([(i1, j1+1), (i1+1, j1), (i1, j1-1), (i1-1, j1)]):
                if i2 < 0 or i2 >= width or j2 < 0 or j2 >= height:
                    pass
                else:
                    i2, j2 = (i2%width), (j2%height)
                    idx2 = (4 * (i2*height + j2))
                    U[idx1+direction, idx2:idx2+4] = U_grid[i2, j2, direction]
    return U

def create_momentum_vector(k, u, width, height):
    r = np.transpose([np.tile(np.arange(0, width), height), np.repeat(np.arange(0, height), width)])
    v_raw = np.kron(np.exp(1j*r.dot(k)), u).flatten()
    return v_raw / np.linalg.norm(v_raw)

def state_vector_to_state_grid(v, width, height):
   return np.reshape(v, (width, height, 4)) 

def reduce_state_grid(grid):
    return np.linalg.norm(np.real(grid), axis=2)

def plot_reduced_state_grid(grid):
    im = plt.imshow(grid, cmap='gray', vmin=0, vmax=0.1)
    return im

def plot_state_vector(v, width, height):
    return plot_reduced_state_grid(reduce_state_grid(state_vector_to_state_grid(v, width, height)))

def isolate_state_vector_region(v, width, height, region):
    v_grid = state_vector_to_state_grid(np.copy(v), width, height)
    (x_start, y_start), (x_end, y_end) = region
    v_grid[0:x_start, :, :] = 0
    v_grid[:, 0:y_start, :] = 0
    v_grid[x_end:width, :, :] = 0
    v_grid[:, y_end:height, :] = 0
    v_isolated = np.reshape(v_grid, (width*height*4))
    return v_isolated/np.linalg.norm(v_isolated)

class TimeEvolutionPlot:
    def __init__(self, U, v, width, height, max_time=20):
        self._U = U
        self._v = v
        self._width = width
        self._height = height

        plt.subplots_adjust(left=0.25, bottom=0.25)
        self._im = plot_state_vector(self._v, self._width, self._height)

        axtime = plt.axes([0.25, 0.15, 0.65, 0.03])
        time = Slider(axtime, 'Time', 0, max_time, valstep=1, valinit=0)
        time.on_changed(lambda t: self._update(t))
        
        plt.show()

    def _update(self, t):
        grid = reduce_state_grid(state_vector_to_state_grid(np.linalg.matrix_power(self._U, int(t)).dot(self._v), self._width, self._height))
        self._im.set_array(grid)
        plt.draw()

def is_edge_state(v, width, height, threshold=0.5):
    v_grid = state_vector_to_state_grid(v, width, height)
    has_mid_edge = np.linalg.norm(v_grid[width//2, height//2]) > threshold/(width*height)
    has_top = np.linalg.norm(v_grid[width//4, height//2]) > threshold/(width*height)
    has_bottom = np.linalg.norm(v_grid[width-(width//4), height//2]) > threshold/(width*height)
    return has_mid_edge, has_top, has_bottom

def is_vertical_edge_state(v, width, height, threshold=0.5):
    v_grid = state_vector_to_state_grid(v, width, height)
    has_mid_edge = np.linalg.norm(v_grid[width//2, height//2]) > threshold/(width*height)
    has_top = np.linalg.norm(v_grid[width//2, height//4]) > threshold/(width*height)
    has_bottom = np.linalg.norm(v_grid[width//2, height-(height//4)]) > threshold/(width*height)
    return has_mid_edge, has_top, has_bottom

def is_patch_edge_state(v, width, height, threshold=0.5):
    v_grid = state_vector_to_state_grid(v, width, height)
    has_edge = np.linalg.norm(v_grid[width//4, height//2]) > threshold/(width*height)
    has_exterior = np.linalg.norm(v_grid[0, height//2]) > threshold/(width*height)
    has_interior = np.linalg.norm(v_grid[width//2, height//2]) > threshold/(width*height)
    return has_edge, has_exterior, has_interior

def find_edge_state_indices(eig, width, height, threshold=0.5, edge_state_finder=is_edge_state):
    indices = []
    for i in range(width*height*4):
        if edge_state_finder(eig[1][:, i], width, height, threshold) == (True, False, False):
            indices.append(i)
    return indices

def plot_spectrum_transition(U1, U2, res=10):
    pairs = [(t, np.real(-1j*np.log(e))) for t in np.linspace(0, 1, res) for e in np.linalg.eigvals(U1*(1-t) + U2*t)]
    T = [p[0] for p in pairs]
    energies = [p[1] for p in pairs]
    plt.scatter(T, energies)
    plt.show()


