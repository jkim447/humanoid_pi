def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # --- NEW: EgoDex branch ---
    if repo_id == "egodex":
        # Lazy import to avoid hard dependency when not used
        from .egodex_dataset import EgoDexSeqDataset

        # Resolve root (field on DataConfig OR env var fallback)
        egodex_root = getattr(data_config, "egodex_root", None) or os.getenv("EGODEX_ROOT")
        if egodex_root is None:
            raise ValueError(
                "EgoDex: set data_config.egodex_root or the EGODEX_ROOT env var to the dataset root."
            )

        return EgoDexSeqDataset(
            root_dir=str(egodex_root),
            action_horizon=action_horizon,                               # use model’s horizon
            image_size=getattr(data_config, "image_size", (224, 224)),
            state_format=getattr(data_config, "state_format", "pi0"),    # or "ego"
            window_stride=getattr(data_config, "window_stride", 1),
            traj_per_task=getattr(data_config, "traj_per_task", None),
            max_episodes=getattr(data_config, "max_episodes", None),
        )

    # --- existing LeRobot branch (unchanged) ---
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    return dataset