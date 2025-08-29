import hydra
from hydra.utils import instantiate

@hydra.main(config_path="configs", config_name="sam_config", version_base=None)
def main(cfg):
    # Select infra group
    infra_cfg = getattr(cfg, cfg.general.infra_group)
    dataset = instantiate(infra_cfg)
    print(f"Loaded EvaluationDataset: {len(dataset)} samples")

    # Instantiate SAMPredictor
    sam_predictor = instantiate(cfg.sam)
    print("SAMPredictor initialized.")

    # Run evaluation
    sam_predictor.run_evaluation(
        dataset=dataset,
        patch_indices=None,
        use_negative_prompts=cfg.general.use_negative_prompts
    )

# --- Main Execution Block ---
if __name__ == '__main__':
    main()
    # config_name = "infra.yml"
    # with open(os.path.join("configs", config_name), "r") as f:
    #     config = yaml.safe_load(f)
    # config = dict2namespace(config)
    
    # # Load the infra dataset
    # sigma_y = config.sam.sigma_y
    # dataset = get_postprocessing_dataset(
    #     config,
    #     sigma_y=sigma_y
    # )
    # print(len(dataset))
    # # run_sam_evaluation(
    # #     dataset=dataset,
    # #     config=config,
    # # )
    # sam_evaluate_results = SAMPredictor(
    #     sam_model_type=config.sam.model_type,
    #     sam_checkpoint=config.sam.root,
    #     debug=False
    # )
    
    # sam_evaluate_results.run_evaluation(
    #     dataset=dataset,
    #     patch_indices=None,
    #     use_negative_prompts=True
    # )
    # # sam_evaluate_results.plot_results()
