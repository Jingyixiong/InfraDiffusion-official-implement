import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate

from diffusion_models.models.ddpm.ddpm import DDPMModelLoader
from diffusion_models.infradiffusion import InfraDiffusion

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    # 1. Instantiate dataset (using Hydra _target_ in data config)
    dataset = instantiate(cfg.data)
    print(f"Dataset loaded: {len(dataset)} samples")

    # 2. Instantiate DDPM model using loader
    device = "cuda"  # or "cpu"
    ddpm_loader = DDPMModelLoader(
        cfg, 
        device=device, 
        data_parallel=False
        )
    
    model, _ = ddpm_loader.load()
    print("DDPM model loaded.")

    # 3. Initialize InfraDiffusion with config, model, and dataset
    runner = InfraDiffusion(
        cfg, 
        device=device,
        model=model, 
        dataset=dataset, 
        test_dataset=dataset
    )
    print("InfraDiffusion initialized.")
    runner.sample_ddim()
    
if __name__ == "__main__":
    main()