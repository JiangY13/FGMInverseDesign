import argparse
import torch
import h5py
from tqdm import tqdm
import json

def JCAL(JCALP_vector, d=0.01, layers=1, freq_start=500, freq_end=6000, frequency_num=550):
    # JCALP parameters ['Por', 'alpha_inf', 'alpha_0v', 'alpha_0th', 'K0', 'theta_0', 'Lambda_th', 'Lambda_v']
    # JCALP_vector : [batch_num, 8]
    # d: Cell size [m]
    ## Air properties at 20°C
    rho_0 = 1.2  # kg/m ** 2 Density
    P_0 = 101320  # Pa Atmosphere pressure
    eta = 1.82e-5  # N*s/m **2  Dynamic viscosity
    gamma = 1.4  # Specific heat ratio
    # Pr = 0.71
    kappa = 2.6e-2  #W/(m*K) Thermal conductivity
    Cp = 1006  # J/(kg *K) Specific heat capability
    c0 = 343  # m/s speed of sound


    por = JCALP_vector[..., 0].unsqueeze(-1)
    alpha_inf = JCALP_vector[..., 1].unsqueeze(-1)
    # alpha_0v = JCALP_vector[..., 2].unsqueeze(-1)
    # alpha_0th = JCALP_vector[..., 3].unsqueeze(-1)
    K_0 = JCALP_vector[..., 4].unsqueeze(-1)
    theta_0 = JCALP_vector[..., 5].unsqueeze(-1)
    Lambda_th = JCALP_vector[..., 6].unsqueeze(-1)
    Lambda_v = JCALP_vector[..., 7].unsqueeze(-1)

    eps = 1e-12
    j = torch.complex(torch.tensor(0.0), torch.tensor(1.0))  
    f = torch.linspace(freq_start, freq_end, frequency_num).to(JCALP_vector.device).unsqueeze(0)
    omega = 2 * torch.pi * f
    omega_c = omega.to(torch.complex64)

    # JCALP model
    F = torch.sqrt(1 + (j * 4 * omega_c * rho_0 * (K_0 * alpha_inf) ** 2 / (eta * (por * Lambda_v) ** 2 + eps)) + eps)
    G = torch.sqrt(1 + (j * 4 * theta_0 ** 2 * Cp * rho_0 * omega_c / (kappa * (Lambda_th * por) ** 2  + eps)) + eps)

    rho_eq = rho_0 * alpha_inf / por * (1 + (eta * por / (j * omega_c * rho_0 * K_0 * alpha_inf + eps)) * F)
    K_eq = gamma * P_0 / por / (
                gamma - (gamma - 1) / (1 - (j * por * kappa / (theta_0 * Cp * rho_0 * omega_c + eps) * G) + eps))

    Zc = torch.sqrt(rho_eq * K_eq + eps)
    k = omega_c / torch.sqrt(K_eq / rho_eq + eps)

    # Transfer matrix for single layer
    cos_kd = torch.cos(k * d)
    sin_kd = torch.sin(k * d)
    T11 = cos_kd
    T12 = j * Zc * sin_kd
    T21 = j * sin_kd / (Zc + eps)
    T22 = cos_kd

    # multiple layers
    T = T11 + T22  
    Delta = T11 * T22 - T12 * T21 
    delta = torch.sqrt(T * T - 4 * Delta)  # delta = sqrt(T^2 - 4*Delta)
    lambda1 = (T + delta) / 2  
    lambda2 = (T - delta) / 2
    lam1_n = lambda1.pow(layers)  
    lam2_n = lambda2.pow(layers)

    denom = lambda1 - lambda2 + eps 

    A11 = (lam1_n * (T11 - lambda2) - lam2_n * (T11 - lambda1)) / denom
    A12 = T12 * (lam1_n - lam2_n) / denom
    A21 = T21 * (lam1_n - lam2_n) / denom
    A22 = (lam1_n * (T22 - lambda2) - lam2_n * (T22 - lambda1)) / denom

    # Absorption coefficient
    Zs = A11 / (A21 + eps)
    # Zs = T11 / (T21 + eps)  one layer
    R = (Zs - rho_0 * c0) / (Zs + rho_0 * c0  + eps)
    alpha = 1 - (R.real**2 + R.imag**2)  #torch.abs(R) ** 2  # Absorption coefficient

    return alpha
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file")
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument(
        "--frequency",
        nargs=2,
        type=float,
        default=[500, 6000],
        metavar=("FMIN", "FMAX")
    )
    args = parser.parse_args()
    
    filename_h5 = args.source_file
    layers = args.layers
    fmin, fmax = args.frequency
    
    cell_size = 0.003 #[m]
    property_list = ['Por', 'alpha_inf', 'alpha_0v', 'alpha_0th', 'K0', 'theta_0', 'Lambda_th', 'Lambda_v']
    absorption_log = []
    filename_log = []
    with h5py.File(filename_h5, "r") as fh5:
        filenames = list(fh5.keys())
        number_sample = len(filenames)
        print("Number of samples:", number_sample)
        for id in tqdm(range(number_sample)):
            filename = filenames[id]
            # Sound absorption coefficient calculation ################
            if ('properties' in fh5[filename]):
                absor_vec = torch.tensor([fh5[filename]['properties'][property_name][()] for property_name in property_list])
                absorption = JCAL(torch.abs(absor_vec), d=cell_size, layers=layers, freq_start=fmin, freq_end=fmax)
                absorption_avg = torch.mean(absorption, dim=-1)
                filename_log.append(filename)
                absorption_log.append(round(absorption_avg.item(), 4))
    
    absorption_dict = dict(zip(filename_log, absorption_log))
    with open("absorption.json", "w") as f:
        json.dump(absorption_dict, f, indent=4)
    
