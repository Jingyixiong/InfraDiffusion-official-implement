import hydra

from sam.vis import plot_sam_results
from datasets.eval import SAMResultsDataset

@hydra.main(config_path="configs", config_name="sam_config", version_base=None)
def sam_vis(cfg):
    infra_group = cfg.general.infra_group
    infra_cfg = getattr(cfg, infra_group)

    # Instantiate datasets (EvaluationDataset + SAMResultsDataset)
    eval_dataset = hydra.utils.instantiate(infra_cfg)
    results_dataset = SAMResultsDataset(
        root=infra_cfg.root,
        patch_size=infra_cfg.patch_size,
        infra_name=infra_cfg.infra_name,
        sigma_y=infra_cfg.sigma_y,
        use_negative_prompts=getattr(
            cfg.general, 
            'use_negative_prompts', 
            False
            )
    )

    # Example figure names list (can be overridden externally)
    fig_ns = [
        'sample_patch'
    ]

    plot_sam_results(
        cfg=cfg,
        results_dataset=results_dataset,
        eval_dataset=eval_dataset,
        fig_n=fig_ns[0],
        patch_index=cfg.general.patch_index,
        iou_threshold=cfg.general.iou_threshold
    )
    

if __name__ == '__main__':
    sam_vis()