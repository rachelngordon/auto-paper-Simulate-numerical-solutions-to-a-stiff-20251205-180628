# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

# Robertson stiff reaction system
def f(y):
    y1, y2, y3 = y
    dy1 = -0.04 * y1 + 1e4 * y2 * y3
    dy2 = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2 ** 2
    dy3 = 3e7 * y2 ** 2
    return np.array([dy1, dy2, dy3])

def jacobian(y):
    y1, y2, y3 = y
    J = np.zeros((3, 3))
    J[0, 0] = -0.04
    J[0, 1] = 1e4 * y3
    J[0, 2] = 1e4 * y2
    J[1, 0] = 0.04
    J[1, 1] = -1e4 * y3 - 6e7 * y2
    J[1, 2] = -1e4 * y2
    J[2, 1] = 6e7 * y2
    J[2, 2] = 0.0
    return J

def rk4_step(y, h):
    k1 = f(y)
    k2 = f(y + 0.5 * h * k1)
    k3 = f(y + 0.5 * h * k2)
    k4 = f(y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def backward_euler_step(y, h, tol=1e-10, max_iter=20):
    # Solve y_new = y + h * f(y_new) using Newton-Raphson
    y_new = y.copy()
    for _ in range(max_iter):
        g = y_new - y - h * f(y_new)
        if np.linalg.norm(g, ord=np.inf) < tol:
            break
        J = np.eye(3) - h * jacobian(y_new)
        delta = np.linalg.solve(J, -g)
        y_new += delta
    return y_new

def integrate(method_step, y0, h, t_end):
    steps = int(np.ceil(t_end / h))
    y = np.empty((steps + 1, 3))
    t = np.empty(steps + 1)
    y[0] = y0
    t[0] = 0.0
    for i in range(steps):
        y[i + 1] = method_step(y[i], h)
        t[i + 1] = t[i] + h
    return t, y

def experiment1():
    y0 = np.array([1.0, 0.0, 0.0])
    h = 0.1
    t_end = 10.0
    t_rk, y_rk = integrate(rk4_step, y0, h, t_end)
    t_be, y_be = integrate(backward_euler_step, y0, h, t_end)
    plt.figure(figsize=(8, 5))
    for i, label in enumerate(['y1', 'y2', 'y3']):
        plt.plot(t_rk, y_rk[:, i], '--', label=f'{label} (RK4)')
        plt.plot(t_be, y_be[:, i], '-', label=f'{label} (BE)')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('Explicit RK4 vs Implicit Backward Euler on Robertson Problem')
    plt.legend()
    plt.tight_layout()
    plt.savefig('robertson_explicit_vs_implicit.png')
    plt.close()

def experiment2():
    y0 = np.array([1.0, 0.0, 0.0])
    t_end = 1.0
    # Reference solution with very small step
    h_ref = 1e-5
    t_ref, y_ref = integrate(backward_euler_step, y0, h_ref, t_end)
    y_ref_final = y_ref[-1]
    step_sizes = [1.0, 0.5, 0.25, 0.125]
    max_errors = []
    for h in step_sizes:
        t_coarse, y_coarse = integrate(backward_euler_step, y0, h, t_end)
        # Compute error at each coarse time point using reference solution
        errors = np.abs(y_coarse - y_ref[(t_coarse / h_ref).astype(int)])
        max_err = errors.max(axis=0)  # max over time for each component
        max_errors.append(max_err)
    max_errors = np.array(max_errors)  # shape (len(step_sizes), 3)
    plt.figure(figsize=(8, 5))
    for i, label in enumerate(['y1', 'y2', 'y3']):
        plt.loglog(step_sizes, max_errors[:, i], 'o-', label=label)
    plt.xlabel('Step size (h)')
    plt.ylabel('Maximum absolute error')
    plt.title('Error vs Step Size for Backward Euler on Robertson Problem')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig('error_vs_step_size.png')
    plt.close()
    return y_ref_final[2]

if __name__ == '__main__':
    experiment1()
    final_y3 = experiment2()
    print('Answer:', final_y3)
