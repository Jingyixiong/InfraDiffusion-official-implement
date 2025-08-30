import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
import torch.utils.data as data

from datasets import get_dataset, data_transform, inverse_data_transform
import torchvision.utils as tvu

def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level

def MeanUpsample(x, scale):
    # downsampled the image size 
    # h, w = image size//scale, image size//scale
    # this A^{\dagger} operator satisfies the
    # [1/n^2, 1/n^2, ..., 1/n^2] * A^{\dagger}^T = I 
    # it sum up the downsampled pixels and then upsampled
    # to the original size by averaging
    n, c, h, w = x.shape    
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)    

def get_beta_schedule(beta_schedule, *, 
                      beta_start, beta_end, num_diffusion_timesteps):
    """
    This function generates a beta schedule for the diffusion process.
    Args:
        beta_start: the starting value of beta
        beta_end: the ending value of beta
        num_diffusion_timesteps: the number of diffusion timesteps
    """
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

# Code form RePaint   
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    # if T_sampling = 100, travel_length = 1, travel_repeat = 1
    # output: [99, 98, ..., 1, -1]
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


class InfraDiffusion():
    def __init__(
            self, 
            config,
            device=None, 
            model=None,  
            dataset=None, 
            test_dataset=None
            ):
        """InfraDiffusion extended to accept externally created model & datasets.

        Args:
            model: (optional) preloaded diffusion model. If None, internal loader used.
            cls_fn: (optional) classifier guidance function.
            dataset/test_dataset: Optional pre-instantiated datasets to reuse.
        """
        self.config = config
        self.data_setting = self.config.data_setting
        self.cfg_ir = self.config.image_restore

        # Override loaded model if provided
        # (1) Load model
        if model is not None:
            self.model = model
        else:
            raise ValueError("Please provide a preloaded model.")

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        # (2) Load data
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.use_boundary_constraint = config.general.use_boundary_constraint

    def _ddpm_scheduler(self):
        '''
        This scheduler is only applicable for DDPM and DDIM.
        '''
        self.model_var_type = self.config.model.ddpm.var_type
        # p(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
        # alpha_t = 1 - beta_t
        # p(x_t | x_0) = N(x_t; sqrt(\bar{alpha}_t) * x_0, (1 - \bar{alpha}_t) * I)
        cfg_scheduler = self.config.scheduler
        betas = get_beta_schedule(
            beta_schedule=cfg_scheduler.beta_schedule,
            beta_start=cfg_scheduler.beta_start,
            beta_end=cfg_scheduler.beta_end,
            num_diffusion_timesteps=cfg_scheduler.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(
            betas
            ).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        # accumulated product: [alpha_0, \alpha_0 \alpha_1, \alpha_0 \alpha_1 \alpha_2, ...]
        alphas_cumprod = alphas.cumprod(dim=0) 
        # alphas_cumprod[:-1] contains \alpha_0 \alpha_1 \alpha_2 ... \alpha_{t-1}, without \alpha_t
        # alphas_cumprod_prev = [1, \alpha_0, \alpha_0 \alpha_1, ...]
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), 
             alphas_cumprod[:-1]], 
             dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        # diffusion variance
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def _get_output_paths(
            self, patch_path: str
            ):
        """Helper function to construct all output paths for a given patch."""
        rendered_dir = os.path.dirname(
            patch_path
            )
        instance_dir = os.path.dirname(
            rendered_dir
            )
        size_str = "_".join(
            map(str, self.config.general.patch_size)
            )
        
        inpainting_dir = os.path.join(
            instance_dir, 
            f"inpainting_{size_str}"
            )
        debug_dir = os.path.join(
            inpainting_dir, "debug")
        base_filename = os.path.basename(
            patch_path
            ).replace('patch_', '')
        
        output_path_npz = os.path.join(
            inpainting_dir, 
            f"inpainted_{base_filename}"
            )
        
        return output_path_npz, inpainting_dir, debug_dir
    
    def _perform_ddim_ir(self, batch):
        """
        Runs the core DDNM sampling loop on a single batch of data.
        This contains the logic from your `simplified_ddnm_plus` method.
        """
        norm_depth_map, _, mask_inpainting, mask_boundary, _, patch_paths = batch

        # Prepare tensors
        x_orig = norm_depth_map.to(
                self.device
                ).permute(0, 3, 1, 2).float()

        mask = mask_inpainting.to(
            self.device).unsqueeze(axis=-1).permute(0, 3, 1, 2).float() 
        mask_boundary = mask_boundary.to(
            self.device).unsqueeze(axis=-1).permute(0, 3, 1, 2).float() 

        x_orig = data_transform(
            self.config, 
            x_orig
            )
        
        # get IR parameters
        A, Ap = self.get_degradation_operator(
            self.cfg_ir,
            mask_tensor=mask
            )
        # print('x_orig data type:', x_orig.dtype)
        # print('mask data type:', mask.dtype)
        # print('mask_boundary data type:', mask_boundary.dtype)

        y = A(x_orig)
        # y = x_orig.detach().clone().to(self.device)
        Apy = Ap(y)

        x = torch.randn(
            y.shape[0],
            self.data_setting.channels,
            self.data_setting.image_size,
            self.data_setting.image_size,
            device=self.device,
        )

        with torch.no_grad():
            n = x.size(0)
            x0_preds = []
            xs = [x]
            times = get_schedule_jump(
                self.time_travel.T_sampling,
                self.time_travel.travel_length,
                self.time_travel.travel_repeat
                )
            time_pairs = list(zip(times[:-1], times[1:]))
            skip = self.config.scheduler.num_diffusion_timesteps // \
                self.time_travel.T_sampling

            for i, j in tqdm.tqdm(time_pairs):
                i, j = i * skip, j * skip
                if j < 0:
                    j = -1

                if j < i:
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = compute_alpha(self.betas, t.long())
                    at_next = compute_alpha(self.betas, next_t.long())
                    sigma_t = (1 - at_next**2).sqrt()
                    xt = xs[-1].to(self.device)

                    et = self.model(xt, t)
                    # et = model(xt, t)*mask_boundary
                    if et.size(1) == 6:
                        et = et[:, :3]

                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                    
                    # original code assume batch size is 1
                    # however, it creates problems in if sentence
                    # if sigma_t >= at_next * sigma_y:
                    #     lambda_t = 1.0
                    #     gamma_t = (sigma_t**2 - (at_next * sigma_y)**2).sqrt()
                    # else:
                    #     lambda_t = sigma_t / (at_next * sigma_y)
                    #     gamma_t = 0.0
                    
                    # updated code should be compatible with batch size > 1
                    # by using torch.where
                    condition = sigma_t >= at_next * self.sigma_y_scaled
                    while len(condition.shape) < len(x0_t.shape):
                        condition = condition.unsqueeze(-1)
                        
                    # Use torch.where for vectorized conditional logic
                    lambda_t = torch.where(
                        condition,
                        torch.ones_like(condition, device=self.device),
                        sigma_t / (at_next * self.sigma_y_scaled)
                    )
                    gamma_t = torch.where(
                        condition,
                        torch.sqrt(
                            torch.clamp(
                                sigma_t**2 - (at_next * self.sigma_y_scaled)**2, min=0
                            )),
                        torch.zeros_like(condition, device=self.device)
                    )

                    x0_t_hat = x0_t - lambda_t * Ap(A(x0_t) - y)
                    if self.use_boundary_constraint:
                        x0_t_hat *= mask_boundary

                    eta = self.cfg_ir.eta
                    c1 = (1 - at_next).sqrt() * eta
                    c2 = (1 - at_next).sqrt() * ((1 - eta**2) ** 0.5)
                    xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)
                    # xt_next = xt_next

                    x0_preds.append(x0_t.to('cpu'))
                    xs.append(xt_next.to('cpu'))
                else:
                    next_t = (torch.ones(n) * j).to(x.device)
                    at_next = compute_alpha(self.betas, next_t.long())
                    x0_t = x0_preds[-1].to(self.device)
                    xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()
                    xs.append(xt_next.to('cpu'))

            output = xs[-1]

        return output, batch, x_orig, Apy
    
    def sample_ddim(self):
        """Run the sampling process."""
        # Get scheduler for DDPM/DDIM
        self._ddpm_scheduler() 

        # the time travel paramter only activated using DDPM/DDIM
        self.time_travel = self.config.scheduler.time_travel
            
        print('Run Simplified DDNM, without SVD.',
                f'{self.time_travel.T_sampling} sampling steps.',
                f'travel_length = {self.time_travel.travel_length},',
                f'travel_repeat = {self.time_travel.travel_repeat}.',
                f'Task: {self.cfg_ir.deg}.'
                )
        self.simplified_ddnm_plus() 

    def simplified_ddnm_plus(self,):
        print(len(self.test_dataset), "samples in the dataset")
        if self.config.general.subset_start >= 0 \
            and self.config.general.subset_end > 0:
            self.test_dataset = torch.utils.data.Subset(
                self.test_dataset, 
                range(
                    self.config.general.subset_start, 
                    self.config.general.subset_end
                    )
                )
        else:
            self.config.general.subset_start = 0
            self.config.general.subset_end = len(self.test_dataset)

        def seed_worker(worker_id):
            worker_seed = self.config.general.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(self.config.general.seed)

        val_loader = data.DataLoader(
            self.test_dataset,
            batch_size=self.data_setting.batch_size,
            shuffle=False,
            num_workers=self.data_setting.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        self.sigma_y_scaled = self.cfg_ir.sigma_y
        pbar = tqdm.tqdm(val_loader)

        for batch in pbar:
            # --- check if the output already exists ---
            # patch_path = batch[-1][0] # Get the patch_path from the batch
            # output_path_npz, _, _ = self._get_output_paths(
            #     patch_path
            #     )
            # if os.path.exists(output_path_npz) \
            #     and not self.active_config.recompute:
            #     pbar.set_description(fd"Skipping existing: {os.path.basename(output_path_npz)}")
            #     continue
            # --- end the check ---
            
            inpainted_batch, original_batch,\
            x_orig_batch, Apy_batch\
                = self._perform_ddim_ir(batch)
            
            if self.config.general.save_results:
                self._save_and_visualize_results(
                    original_batch, inpainted_batch,
                    x_orig_batch, Apy_batch,
                    recompute=self.config.general.recompute
                )

    def _save_and_visualize_results(
            self, 
            original_batch, inpainted_batch, 
            x_orig_batch, Apy_batch,
            recompute
            ):
        """
        Saves the inpainted images, debug images, and visualizations.
        """
        norm_depth_map, _, mask_inpainting, \
            mask_boundary, _, patch_paths = original_batch
        
        for i in range(len(patch_paths)):
            patch_path = patch_paths[i]
            
            rendered_dir = os.path.dirname(patch_path)
            instance_dir = os.path.dirname(rendered_dir)
            size_str = "_".join(
                map(str, self.config.general.patch_size)
                )
            # adjustment needs to made to accommodate for 
            # different parameters
            if self.sigma_y_scaled == 0.0:    # no noise in source img
                inpainting_folder_name = f"inpainting_{size_str}"
            else:
                inpainting_folder_name = f"inpainting_{size_str}_{self.sigma_y_scaled}"

            if not self.use_boundary_constraint:
                # Add suffix for the vanilla DDNM
                inpainting_folder_name += "_naive" 

            inpainting_dir = os.path.join(
                instance_dir, 
                inpainting_folder_name
                )
            print('Save into:', inpainting_dir)
            os.makedirs(inpainting_dir, exist_ok=True)
            debug_dir = os.path.join(inpainting_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            base_filename = os.path.basename(patch_path).replace('patch_', '')
            
            # Save Debug Images
            orig_path = os.path.join(
                debug_dir, 
                f"orig_{base_filename.replace('.npz', '.png')}"
                )
            apy_path = os.path.join(
                debug_dir, 
                f"Apy_{base_filename.replace('.npz', '.png')}"
                )

            if not os.path.exists(orig_path)\
                or recompute:
                orig_img_np = inverse_data_transform(
                    self.config, x_orig_batch[i]
                    ).permute(1, 2, 0).cpu().numpy()
                plt.imsave(
                    orig_path, 
                    np.clip(orig_img_np, 0, 1)
                    )

            if not os.path.exists(apy_path) or recompute:
                apy_img_np = inverse_data_transform(
                    self.config, Apy_batch[i]
                    ).permute(1, 2, 0).cpu().numpy()
                plt.imsave(apy_path, np.clip(apy_img_np, 0, 1))
            
            # Save Final Inpainting Result
            output_path_npz = os.path.join(
                inpainting_dir, f"inpainted_{base_filename}"
                )
            if os.path.exists(output_path_npz) and not recompute:
                continue

            inpainted_image_np = inverse_data_transform(
                self.config, inpainted_batch[i]
                ).permute(1, 2, 0).cpu().numpy()
            np.savez(
                output_path_npz, 
                inpainted_depth=inpainted_image_np
                )
            print(f"  Saved inpainting result to {output_path_npz}")
            
            # Save Visualization Summary
            output_path_vis = os.path.join(
                inpainting_dir, 
                f"inpainted_{base_filename.replace('.npz', '_visualization.png')}"
                )
            self._visualize_single(
                norm_depth_map[i], 
                inpainted_batch[i], 
                mask_inpainting[i], 
                mask_boundary[i], 
                output_path_vis
                )

    def _visualize_single(
            self, 
            norm_depth_map, 
            x_img, 
            mask_inpainting, mask_boundary, 
            save_path
            ):
        """Generates and saves a 2x2 visualization for a single item."""
        norm_depth = np.mean(
            norm_depth_map.cpu().numpy(), 
            axis=2
            )
        inpainted = np.mean(
            inverse_data_transform(
                self.config, x_img
                ).permute(1, 2, 0).cpu().numpy(), 
                axis=2
                )
        mask_inp = mask_inpainting.cpu().numpy()
        mask_bnd = mask_boundary.cpu().numpy()

        mask_inp_rgb = np.zeros((*mask_inp.shape, 3), dtype=np.uint8)
        mask_inp_rgb[mask_inp != 0] = [0, 255, 0]
        mask_bnd_rgb = np.zeros((*mask_bnd.shape, 3), dtype=np.uint8)
        mask_bnd_rgb[mask_bnd != 0] = [0, 255, 0]

        _, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs[0, 0].imshow(
            norm_depth, cmap='gray'
            ); axs[0, 0].set_title("Normalized Depth")
        axs[0, 1].imshow(
            inpainted, cmap='gray'
            ); axs[0, 1].set_title("Inpainted Output")
        axs[1, 0].imshow(
            mask_inp_rgb
            ); axs[1, 0].set_title("Mask: Inpainting")
        axs[1, 1].imshow(
            mask_bnd_rgb
            ); axs[1, 1].set_title("Mask: Boundary")
        for ax in axs.ravel(): ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def get_degradation_operator(
            self, 
            config_ir, 
            mask_tensor=None
            ):
        if config_ir.deg == 'colorization':
            A = lambda z: color2gray(z)
            Ap = lambda z: gray2color(z)

        elif config_ir.deg == 'denoising':
            A = lambda z: z
            Ap = A

        elif config_ir.deg == 'sr_averagepooling':
            scale = round(config_ir.deg_scale)
            A = torch.nn.AdaptiveAvgPool2d(
                (256 // scale, 
                 256 // scale)
                 )
            Ap = lambda z: MeanUpsample(z, scale)

        elif config_ir.deg == 'inpainting':
            if mask_tensor is None:
                raise ValueError("Mask tensor must be provided for inpainting.")
            A = lambda z: z * mask_tensor
            Ap = A

        elif config_ir.deg == 'mask_sr':
            if mask_tensor is None:
                raise ValueError("Mask tensor must be provided for mask_sr.")
            A1 = lambda z: z * mask_tensor
            A1p = A1
            scale = round(config_ir.deg_scale)
            A2 = torch.nn.AdaptiveAvgPool2d((256 // scale, 256 // scale))
            A2p = lambda z: MeanUpsample(z, scale)

            A = lambda z: A2(A1(z))
            Ap = lambda z: A1p(A2p(z))

        elif config_ir.deg == 'mask_color_sr':
            if mask_tensor is None:
                raise ValueError("Mask tensor must be provided for mask_color_sr.")
            A1 = lambda z: z * mask_tensor
            A1p = A1
            A2 = lambda z: color2gray(z)
            A2p = lambda z: gray2color(z)
            scale = round(config_ir.deg_scale)
            A3 = torch.nn.AdaptiveAvgPool2d((256 // scale, 256 // scale))
            A3p = lambda z: MeanUpsample(z, scale)

            A = lambda z: A3(A2(A1(z)))
            Ap = lambda z: A1p(A2p(A3p(z)))

        elif config_ir.deg == 'diy':
            if mask_tensor is None:
                raise ValueError("Mask tensor must be provided for diy.")
            A1 = lambda z: z * mask_tensor
            A1p = A1
            A2 = lambda z: color2gray(z)
            A2p = lambda z: gray2color(z)
            A = lambda z: A2(A1(z))
            Ap = lambda z: A1p(A2p(z))

            # scale = args.deg_scale
            # A3 = torch.nn.AdaptiveAvgPool2d((256 // scale, 256 // scale))
            # A3p = lambda z: MeanUpsample(z, scale)

            # A = lambda z: A3(A2(A1(z)))
            # Ap = lambda z: A1p(A2p(A3p(z)))

        else:
            raise NotImplementedError(
                f"Degradation type {config_ir.deg} not supported."
                )

        return A, Ap


