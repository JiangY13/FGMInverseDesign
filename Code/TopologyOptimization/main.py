import glob
from pathlib import Path
import argparse
import resource
from lib.models.decoder import *  
from lib.models.NFfield import *
from lib.models.DANN import *
from lib.workspace import *
from lib.utils import * 
from FEA.FEA import FE
from FEA.BC import *
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

class Optimizer:
    # -----------------------------#
    def __init__(self, experiment_directory, morphology_learning_source, initial_case, mesh, matProp, bc, desiredVolumeFraction, densityProjection, graded_design=True, compatible_design=True, overrideGPU=True):
        os.makedirs(experiment_directory, exist_ok=True)
        self.graded = graded_design
        self.compatible_design = compatible_design
        self.setDevice(overrideGPU)
        self.dtype = torch.float32
        self.mesh = mesh
        c = self.generate_elements()
        c_array, d_array = self.elem2array(c)
        self.FE = FE(self.mesh, matProp, bc, device=self.device)
        self.desiredVolumeFraction = desiredVolumeFraction
        self.local_vol_max = 2 * desiredVolumeFraction ** (2/3)
        self.density = self.desiredVolumeFraction * np.ones((self.FE.mesh.numElems))
        self.densityProjection = densityProjection
        self.ks = 5  #penalty parameter for ks_mean
        self.max_ks = 25
        if self.compatible_design:
            self.grad_gap = 0.25
        else:
            self.grad_gap = 0
        self.initial_case = initial_case
        self.vm_0 = 1
        self.objective = 0.0

        xy = torch.tensor(c_array, dtype=self.dtype)  
        self.xy = xy.clone().detach().float().view(-1, 3).to(self.device) 
        dis_e = torch.tensor(d_array, dtype=self.dtype).detach().float().view(-1, 3).to(self.device)
        self.xy_e = self.xy - dis_e
        self.print_xy_e = self.xy_e / (1/2-1/self.mesh['resolution']/2) * 1/2
        self.elem_x, self.elem_y, self.elem_z = int(self.mesh['len'][0]), int(self.mesh['len'][1]), int(self.mesh['len'][2])
        ##################
        II, JJ, SS = self.generate_filterM()
        II = torch.tensor(II)
        II = II.int()
        JJ = torch.tensor(JJ)
        JJ = JJ.int()
        SS = torch.tensor(SS)
        SS = torch.reshape(SS, (-1,))
        ind = torch.cat((torch.reshape(II, (1, -1)), torch.reshape(JJ, (1, -1))), dim=0)
        self.LL = torch.sparse_coo_tensor(indices=ind, values=SS, size=(self.FE.mesh.numElems, self.FE.mesh.numElems), dtype=self.dtype, device=self.device)

        ########## Load MorphL Model ########
        specs_filename = os.path.join(morphology_learning_source, "specs.json")
        specs = json.load(open(specs_filename))
        latent_size = specs["CodeLength"]
        decoder_source = DeepSDF(latent_size, **specs["NetworkSpecs"])
        decoder_source = torch.nn.DataParallel(decoder_source)
        saved_model_state = torch.load(os.path.join(morphology_learning_source, model_params_subdir, "latest-best.pth"))
        decoder_source.load_state_dict(saved_model_state["model_state_dict"])
        self.decoder = decoder_source.module.to(self.device)
        self.decoder.eval()
        latent_filename = os.path.join(morphology_learning_source, latent_codes_subdir, "latest-best.pth")
        self.latent = torch.load(latent_filename)["latent_codes"]["weight"].to(self.device).to(self.dtype)
        
        s = morphology_learning_source.name.lower()
        if "truss" in s:
            data_type = "truss"
        elif "mix" in s:
            data_type = "mix"
        elif "shell" in s:
            data_type = "shell"
        else:
            raise ValueError(f"Invalid source file name '{morphology_learning_source}'.")

        latent_D = self.latent.shape[1]
        ########## NF Model ########
        tau_quantile = 0.9
        nf_cfg = NF_Config(
            latent_dim=latent_D,
            noise_ratio={'shell': 0.025, 'truss': 0.025, 'mix': 0.02}.get(data_type, 0.025),
            q=1-tau_quantile, 
        )

        flow_path = os.path.join(experiment_directory, model_params_subdir, "latest.pth")

        if os.path.isfile(flow_path):
            saved = torch.load(flow_path, map_location=self.device)
            flow = RealNVP(dim=nf_cfg.latent_dim, n_layers=nf_cfg.n_layers, hidden=nf_cfg.hidden_dim)
            flow.load_state_dict(saved["model_state_dict"])
            flow = flow.to(self.device)
            print("NF module loaded.")

        else:
            print("No saved NF module, training new one")

            flow = train_flow(self.latent,
                            nf_cfg,
                            save_path=experiment_directory,
                            device=self.device).eval()
        self.nf_field = NFField(flow, self.latent, q=nf_cfg.q) 
        ########## DA ####################
        da_path = os.path.join(experiment_directory, operator_params_subdir, "latest.pth")
        if os.path.isfile(da_path):
            saved = torch.load(da_path, map_location=self.device)
            da_cfg_loaded = saved["config"]
            da_cfg = DA_Config()
            da_cfg.__dict__.update(da_cfg_loaded)
            self.DA_net = DANet(latent_dim=da_cfg.latent_dim, hidden_dim=da_cfg.hidden_dim, depth=da_cfg.depth, 
                            dropout=da_cfg.dropout)
            self.DA_net.load_state_dict(saved["operator_state_dict"])
            self.DA_net = self.DA_net.to(self.device)
            print("DA operator loaded.")

        else:
            print("No saved DA operator, training new one")
            ########## Train Dataset ############
            sampled_samples=3000
            save_samples_path = (
                experiment_directory /
                f"Num_{sampled_samples}_latent_bank_aug.pt"
            )
            if save_samples_path.exists():
                print("Loading latent_bank_aug...")
                latent_bank_aug = torch.load(save_samples_path).to(self.device)
            else:
                latent_bank_aug, stats = build_latent_bank_aug(
                    latent_bank=self.latent,   # [N,D]
                    nf_model=self.nf_field,
                    num_aug=sampled_samples,  #number of training samples
                    device=self.device,
                )
                torch.save(latent_bank_aug, save_samples_path)

            print(f"[DANN] latent bank shape: {latent_bank_aug.shape}")

            bank_info = make_bins(
                latent_bank=latent_bank_aug,
                nf_model=self.nf_field,
                device=self.device,
            )

            da_cfg = DA_Config(
                latent_dim=latent_D,
                batch_size=256,  
                train_samples=sampled_samples,
                steps=500,
                lr=2e-4,
                hidden_dim=256,
                depth=3,
                lambda_nf=1.0,
                lambda_smooth=5e1,
                lambda_regular=1.0,
                amp=True,
            )

            self.DA_net = train_DA_net(
                latent_info=bank_info,
                nf_model = self.nf_field,
                cfg=da_cfg,
                save_path=experiment_directory,
                device=self.device,
                ).eval()
            
        self.da_trans = DA_Trans(self.DA_net, self.nf_field)

    def barrier_loss(self, z):
        """
        z: (B, d) design latent
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        logp = self.nf_field.s0(z)
        gap = - logp
        gap_relu = torch.relu(gap)
        return gap_relu.mean()  
    # -----------------------------#
    def setDevice(self, overrideGPU):
        if (torch.cuda.is_available() and (overrideGPU == True)):
            import subprocess
            def get_free_gpu():
                result = subprocess.check_output(
                        "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader",
                        shell=True,
                )
                mem = [int(x) for x in result.decode().split()]
                return mem.index(max(mem))

            gpu = get_free_gpu()
            self.device = torch.device(f"cuda:{gpu}")
            print("GPU enabled, using GPU", gpu)
        else:
            self.device = torch.device("cpu")
            print("Running on CPU")

    def generate_elements(self,):
        res = self.mesh['resolution']
        nelx, nely, nelz = res, res, res
        step = 1.0 / res  #res_rescale
        v = np.zeros((nelx * nely * nelz, 3))
        cen = np.array([-1 / 2 + step / 2, 1 / 2 - step / 2, -1 / 2 + step / 2])
        k = 0
        for z in range(nelz):
            for x in range(nelx):
                for y in range(nely):
                    v[k, :] = cen + np.array([x, -(y), z]) * step
                    k = k + 1
        return v

    def generate_points(self,):
        len_ = self.mesh['len']
        res = self.mesh['resolution']
        extra_board = self.mesh['extra_board']
        nelx, nely, nelz = self.mesh['nelx'], self.mesh['nely'], self.mesh['nelz']
        step = 1.0 / res  
        v = np.zeros((nelx * nely * nelz, 3))
        cen = np.array([-len_[0] / 2 + step / 2 - extra_board[0,0]*step, len_[1] / 2 - step / 2 + extra_board[1,1]*step, -len_[2] / 2 + step / 2 - extra_board[2,0]*step])
        k = 0
        for z in range(nelz):
            for x in range(nelx):
                for y in range(nely):
                    v[k, :] = cen + np.array([x, -(y), z]) * step
                    k = k + 1
        return v

    def generate_grids(self,):
        len_ = self.mesh['len']
        res = self.mesh['resolution']
        extra_board = self.mesh['extra_board']
        nelx, nely, nelz = self.mesh['nelx'], self.mesh['nely'], self.mesh['nelz']
        step = 1.0 / res  
        v = np.zeros(((nelx+1) * (nely+1) * (nelz+1), 3))
        cen = np.array([-len_[0] / 2 - extra_board[0,0]*step, len_[1] / 2 + extra_board[1,1]*step, -len_[2] / 2 - extra_board[2,0]*step])
        k = 0
        for z in range(nelz+1):
            for x in range(nelx+1):
                for y in range(nely+1):
                    v[k, :] = cen + np.array([x, -(y), z]) * step
                    k = k + 1
        return v

    def generate_filterM(self,):
        p = self.FE.mesh.elemCenters
        kk = 64
        nbrs = NearestNeighbors(n_neighbors=kk, algorithm='ball_tree').fit(p)
        dis, ind = nbrs.kneighbors(p)
        dis1 = np.exp(-2 * dis)
        dd = np.sum(dis1, 1)
        dis1 = dis1 / np.reshape(dd, (-1, 1))
        a = np.reshape(np.arange(0, p.shape[0]), (-1, 1))
        b = np.tile(a, (1, kk))
        II = b.flatten()
        JJ = ind.flatten()
        SS = dis1.flatten()

        return II, JJ, SS
    # -----------------------------#
    def elem2array(self, c):
        res = self.mesh['resolution']
        c = c.reshape(res,res,res,3)
        X, Y, Z = map(int, self.mesh['len'])  
        pad_x, pad_y, pad_z = int(res * (X-1)), int(res * (Y-1)), int(res * (Z-1))
        c_array = np.pad(c, pad_width=((0, pad_z), (0, pad_x), (0, pad_y), (0,0)), mode='wrap')

        xy_real = c_array.copy() + 0.5
        xy_real[..., 0] = xy_real[..., 0] + np.repeat(np.arange(X), res)[None, :, None]
        xy_real[..., 1] = xy_real[..., 1] + np.repeat(np.arange(Y), res)[None, None, :]
        xy_real[..., 2] = xy_real[..., 2] + np.repeat(np.arange(Z), res)[:, None, None]
        d_array = xy_real - c_array

        return xy_real.reshape(-1, 3), d_array.reshape(-1, 3)

    def projectDensity(self, x):
        if (self.densityProjection['isOn']):
            b = self.densityProjection['sharpness']
            xp = (torch.exp(2*x) - 1)*(torch.exp(-2*x) + 1)
            xp = ((np.tanh(0.0 * b) + torch.tanh(b * (xp - 0.0))) / (np.tanh(0.0 * b) + np.tanh(b*(1-0.0))) + 1) / 2
        return xp

    def densityFilter(self, x):
        xt = torch.matmul(self.LL, torch.reshape(x, (-1, 1)))
        xt = torch.reshape(xt, (-1, ))
        return xt

    def ks_mean(self, x):  
        # x: [..., N] stress（eg. sigma_vm/sigma_allow）
        xmax, _ = x.max(dim=-1, keepdim=True)
        y = xmax + torch.log(torch.mean(torch.exp(self.ks * (x - xmax)), dim=-1, keepdim=True) + 1e-20) / self.ks
        return y.squeeze(-1)

    def adjust_learning_rate(self, lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def adjust_opt_parameters(self, e):
        alphaMax = 500
        alphaIncrement = 1
        beitaMax = 6e-2
        beitaIncrement = 2e-4
        self.alpha = min(alphaMax, self.alpha + alphaIncrement)
        self.beita = min(beitaMax, self.beita + beitaIncrement)

        if ((e + 1) % 30 == 0) and self.densityProjection[
            'sharpness'] < 16:  
            if self.densityProjection['sharpness'] < 3:
                self.densityProjection['sharpness'] = self.densityProjection['sharpness'] * 2
            else:
                self.densityProjection['sharpness'] = self.densityProjection['sharpness'] + 1

        if ((e + 1) % 100 == 0) and self.ks < self.max_ks:  
            self.ks += 5

    def structuralopt(self, maxepoch):
        lr = 5e-3 
        self.alpha = 10
        self.beita = 1e-3 
        Obj_vol = np.zeros((maxepoch, 5))  
        loss_record = []
        converge_log = []

        # Initializing Design variables
        x = self.latent[self.initial_case].clone()
        Lcode = np.zeros((maxepoch, x.reshape(-1).shape[0]))
        x.requires_grad = True
        # Initializing Optimizer
        self.optimizer = torch.optim.Adam([x], lr)
        try:
            start_epoch, design_var = load_optimizer(experiment_directory, "latest.pth", self.optimizer)
            Lcode[:start_epoch+1] = np.loadtxt(experiment_directory / 'optimized_latent_code.txt')
            Obj_vol[:start_epoch+1] = np.loadtxt(experiment_directory / 'Obj_vol.txt')
            with torch.no_grad():
                x.copy_(design_var.to(x.device).to(x.dtype))
                self.vm_0 = 0.05 * Obj_vol[0, -1]
                for ep in range(start_epoch + 1):
                    self.adjust_opt_parameters(ep)
            print("optimizer loaded, epoch =", start_epoch)
        except:
            start_epoch = -1
            print("new optimizer initialized")

        print("Starting optimization:")
        # Starting Optimization
        for e in tqdm(range(maxepoch)):
            if e <= start_epoch:
                continue
            else:
                self.optimizer.zero_grad()
                nf_constraint = self.barrier_loss(x)
                if self.graded:
                    x_expand = self.da_trans(x, self.xy[:,2], w=self.grad_gap)
                else:
                    x_expand = self.da_trans(x.expand(self.mesh["len"][2], -1), self.xy[:,2], w=self.grad_gap)
                
                sdf_elem = self.decoder(x_expand["out"], self.xy_e).to(self.device)  
                sdf_e = 4 / self.FE.mesh.nelx * torch.ones(self.FE.mesh.numElems).to(self.device)
                sdf_e[self.FE.mesh.elemStructure] = torch.flatten(sdf_elem)

                rhoFilter = sdf_e.clone()  
                rhoElem = self.projectDensity(self.FE.mesh.nelx / 10 * rhoFilter)
                rhoElem[self.FE.mesh.solid_elems] = 1
                rhoElem[self.FE.mesh.void_elems] = 0
                rhoElem = torch.clamp(rhoElem, min=0, max=1)
                self.density = rhoElem.detach().cpu().numpy()
                vm_, Compliance = self.FE.solve_stress(rhoElem)

                if (e == 0):
                    self.vm_0 = 0.05 * max(vm_).detach()

                self.objective = self.ks_mean(vm_/self.vm_0)

                volConstraint = ((torch.sum(self.FE.mesh.elemArea.to(self.device) * rhoElem) - self.FE.mesh.solidArea) / \
                                ((self.FE.mesh.netArea - self.FE.mesh.solidArea - self.FE.mesh.voidArea) * self.desiredVolumeFraction)) - 1.0
                currentVolumeFraction = (torch.sum(self.FE.mesh.elemArea.to(self.device) * rhoElem) - self.FE.mesh.solidArea) / \
                                (self.FE.mesh.netArea - self.FE.mesh.solidArea - self.FE.mesh.voidArea)
                currentVolumeFraction = currentVolumeFraction.cpu().detach().numpy()
                vol_term = torch.pow(volConstraint, 2)

                rhoElem3D = rhoElem[self.FE.mesh.elemStructure].reshape(self.mesh['len'][2],self.mesh['resolution'],self.mesh['len'][0],
                                                                        self.mesh['resolution'],self.mesh['len'][1],self.mesh['resolution'])
                vol_zmin = rhoElem3D[:, 0, :, :, :, :].sum(dim=(2, 4))
                vol_zmax = rhoElem3D[:, -1, :, :, :, :].sum(dim=(2, 4))

                vol_xmin = rhoElem3D[:, :, :, 0, :, :].sum(dim=(1, 4))
                vol_xmax = rhoElem3D[:, :, :, -1, :, :].sum(dim=(1, 4))

                vol_ymin = rhoElem3D[:, :, :, :, :, 0].sum(dim=(1, 3))
                vol_ymax = rhoElem3D[:, :, :, :, :, -1].sum(dim=(1, 3))

                face_volumes = torch.stack([vol_zmin, vol_zmax, vol_xmin, vol_xmax, vol_ymin, vol_ymax],dim=-1)
                local_vol = face_volumes.reshape(-1) / self.mesh['resolution'] ** 2
                local_volConstraint = torch.relu(local_vol/self.local_vol_max - 1.0).mean()

                loss = self.objective + self.alpha * (vol_term + local_volConstraint) + self.beita * nf_constraint
                loss.backward(retain_graph=True)
                with torch.no_grad():
                    Obj_vol[e, 0] = self.objective.item()
                    Obj_vol[e, 1] = currentVolumeFraction
                    Obj_vol[e, 2] = local_volConstraint.item()
                    Obj_vol[e, 3] = nf_constraint.item()
                    Obj_vol[e, 4] = max(vm_).item()  #max stress

                    Lcode[e] = x.reshape(-1).detach().cpu().numpy()
                    loss_record.append(loss.item())
                self.adjust_opt_parameters(e)
                self.optimized_latent_code = x

                titleStr = "{:d} \t Obj. {:.4F} \t Vol. {:.4F} \t Local_vol. {:.4F} \t nf. {:.4F}". \
                    format(e, self.objective.item(), currentVolumeFraction, local_volConstraint.item(), nf_constraint.item())
                print(titleStr)

                if e >= 5:
                    obj_avg = np.sum(Obj_vol[e - 4:e + 1, 0]) / 5
                    obj_converge = np.abs(Obj_vol[e, 0] - obj_avg) / obj_avg
                else:
                    obj_converge = np.ones(1)
                converge_log.append(obj_converge.item())

                self.optimizer.step()

                save_optimizer(experiment_directory, "latest.pth", self.optimizer, e, x)
                np.savetxt(experiment_directory/'optimized_latent_code.txt', Lcode[:e+1])
                np.savetxt(experiment_directory/'Obj_vol.txt', Obj_vol[:e+1])
                np.savetxt(experiment_directory/'convergence.txt', converge_log)

        print('======>Loop: {:d}, Avg. time: {:.4f}=== Obj. ini: {:.4f}, opt:{:.4f}=== Vol. ini: {:.4f}, opt:{:.4f}<======'.format(
            (e+1), avg_time, Obj_vol[0, 0], Obj_vol[e, 0], Obj_vol[0, 1], Obj_vol[e, 1]))




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_type", default="mix")
    parser.add_argument("--compatible_design", type=lambda x: x.lower()=="true", default=True)
    parser.add_argument("--graded_design", type=lambda x: x.lower()=="true", default=True)
    parser.add_argument("--maxEpochs", type=int, default=1500)
    
    args = parser.parse_args()
    
    maxEpochs = args.maxEpochs

    data_type = args.data_type
    compatible_design = args.compatible_design
    graded_design = args.graded_design

    matProp = {'E': 1.55e9, 'nu': 0.36, 'penal': 3.} 
    desiredVolumeFraction = 0.30
    len_ = np.array([1,1,4]).astype(int)  #must be integral cell number
    elemSize = 0.003 * np.array([1.0, 1.0, 1.0])
    extra_board = np.array([[1, 0], [0, 0], [0, 0]]).astype(int)  
    res = 40 
    #=============================================================
    mesh, bc = BC_bending(res, len_, elemSize, extra_board)

    densityProjection = {'isOn': True, 'sharpness': 2}  

    if graded_design:
        results_name = 'graded'
    else:
        results_name = 'uniform'

    experiment_directory = Path("Outputs") / f"{results_name}_{data_type}"
    matches = list(
        Path("./Code/MorphologyLearning/experiments").glob(
            f"cellculture_period_{data_type}"
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"No experiment directory found for data_type='{data_type}'"
        )
    
    morphology_learning_source = matches[0]

    initial_case = 2 * np.ones(len_[2]) 
    if not graded_design:
        initial_case = int(initial_case[0])

    topOpt = Optimizer(experiment_directory, morphology_learning_source, initial_case, mesh, matProp, bc, desiredVolumeFraction, densityProjection, graded_design=graded_design, compatible_design=compatible_design, overrideGPU=False)  #[experiment_directory_init, experiment_directory]
    topOpt.structuralopt(maxEpochs)
