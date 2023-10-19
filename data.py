import numpy as np
import torch

#Numerical Method to Solve the 2D KS Equation
def nonlinear_term(u, L):
    N = u.shape[0]
    k = np.fft.fftfreq(N) * N * 2 * np.pi / L
    u_fft = np.fft.fft2(u)
    uy = np.real(np.fft.ifft2(1j * k[:, np.newaxis] * u_fft))
    ux = np.real(np.fft.ifft2(1j * u_fft * k))
    f = -(ux**2 + uy**2) / 2
    return f

def implicit_step(u, dt, L):
    N = u.shape[0]
    k = np.fft.fftfreq(N) * N * 2 * np.pi / L
    k_sq = k ** 2 + (k[:, np.newaxis]) ** 2
    u_fft = np.fft.fft2(u)
    u_next = np.real(np.fft.ifft2(u_fft / (1 - dt * k_sq + dt * k_sq ** 2)))
    return u_next

def operator_splitting_step(u, dt, L):
    f = nonlinear_term(u, L)
    u_next = implicit_step(u + f * dt, dt, L)
    return u_next

N = 256
L = 100
dt = 0.1
steps = 10000

gp = np.linspace(0, 2 * np.pi, N, endpoint=False)
x, y = np.meshgrid(gp, gp)

#Initial conditions -> sum of cosines plus sines and random noise
u = np.cos(x) + np.cos(y)
u = 5 * np.random.rand(*u.shape)

#Data generation helper functions
def nonlinear_term(u, L):
    N = u.shape[0]
    k = np.fft.fftfreq(N) * N * 2 * np.pi / L
    u_fft = np.fft.fft2(u)
    uy = np.real(np.fft.ifft2(1j * k[:, np.newaxis] * u_fft))
    ux = np.real(np.fft.ifft2(1j * u_fft * k))
    f = -(ux**2 + uy**2) / 2
    return f

def implicit_step(u, dt, L):
    N = u.shape[0]
    k = np.fft.fftfreq(N) * N * 2 * np.pi / L
    k_sq = k ** 2 + (k[:, np.newaxis]) ** 2
    u_fft = np.fft.fft2(u)
    u_next = np.real(np.fft.ifft2(u_fft / (1 - dt * k_sq + dt * k_sq ** 2)))
    return u_next

def operator_splitting_step(u, dt, L):
    f = nonlinear_term(u, L)
    u_next = implicit_step(u + f * dt, dt, L)
    return u_next


gp = np.linspace(0, 2 * np.pi, N, endpoint=False)
x, y = np.meshgrid(gp, gp)

#Initial conditions -> sum of cosines plus sines and random noise
u = np.cos(x) + np.cos(y)
u = 5 * np.random.rand(*u.shape)

def generate_synthetic_data(N=256, L=100, dt=0.25, steps=2000):
    gp = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x, y = np.meshgrid(gp, gp)
    u = np.cos(x) + np.cos(y)
    u = 5 * np.random.rand(*u.shape)
    u_tensor_list = []

    for _ in range(steps):
        u = operator_splitting_step(u, dt, L)
        u -= np.mean(u)
        u_tensor = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape -> (1, 1, N, N)
        u_tensor_list.append(u_tensor)

    return torch.cat(u_tensor_list, dim=0)  # shape -> (steps, 1, N, N)