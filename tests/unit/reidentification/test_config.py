"""Tests for re-identification configuration module."""

from pathlib import Path

from jaguars.reidentification.config import (
    BackboneConfig,
    DatasetConfig,
    EvaluationConfig,
    ModelConfig,
    ReidentificationConfig,
    TrainingConfig,
    WandbConfig,
    get_default_config,
    load_config_from_dict,
)


def test_backbone_config_defaults() -> None:
    """Test BackboneConfig with default values."""
    config = BackboneConfig()
    assert config.name == "vit_large_patch14_dinov2.lvd142m"
    assert config.pretrained is True
    assert config.input_size == 384
    assert config.batch_size == 32
    assert config.embedding_dim == 1536


def test_model_config_defaults() -> None:
    """Test ModelConfig with default values."""
    config = ModelConfig()
    assert config.hidden_dim == 512
    assert config.embedding_dim == 256
    assert config.dropout == 0.3
    assert config.arcface_margin == 0.5
    assert config.arcface_scale == 64.0


def test_dataset_config_defaults() -> None:
    """Test DatasetConfig with default values."""
    config = DatasetConfig()
    assert config.source == "fiftyone"
    assert config.fo_dataset_name == "JID_Master_Dataset"
    assert config.num_workers == 0
    assert config.pin_memory is False


def test_training_config_defaults() -> None:
    """Test TrainingConfig with default values."""
    config = TrainingConfig()
    assert config.learning_rate == 1e-4
    assert config.batch_size == 32
    assert config.num_epochs == 50
    assert config.early_stopping_patience == 10
    assert config.save_dir == Path("data/models/reidentification")


def test_evaluation_config_defaults() -> None:
    """Test EvaluationConfig with default values."""
    config = EvaluationConfig()
    assert config.compute_map is True
    assert config.compute_cmc is True
    assert config.cmc_top_k == [1, 5, 10, 20]
    assert config.add_to_fiftyone is False


def test_wandb_config_defaults() -> None:
    """Test WandbConfig with default values."""
    config = WandbConfig()
    assert config.enabled is True
    assert config.project == "camerate-trap-reidentificationentification"
    assert config.log_frequency == 10


def test_reidentification_config_defaults() -> None:
    """Test complete ReidentificationConfig."""
    config = ReidentificationConfig()
    assert isinstance(config.backbone, BackboneConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.dataset, DatasetConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.evaluation, EvaluationConfig)
    assert isinstance(config.wandb, WandbConfig)
    assert config.seed == 42
    assert config.verbose is False


def test_get_default_config() -> None:
    """Test get_default_config function."""
    config = get_default_config()
    assert isinstance(config, ReidentificationConfig)
    assert config.backbone.name == "vit_large_patch14_dinov2.lvd142m"


def test_load_config_from_dict() -> None:
    """Test loading config from dictionary."""
    config_dict = {
        "backbone": {"name": "custom_backbone", "embedding_dim": 2048},
        "model": {"hidden_dim": 1024, "embedding_dim": 512},
        "training": {"learning_rate": 0.001, "num_epochs": 100},
        "seed": 123,
    }

    config = load_config_from_dict(config_dict)
    assert config.backbone.name == "custom_backbone"
    assert config.backbone.embedding_dim == 2048
    assert config.model.hidden_dim == 1024
    assert config.model.embedding_dim == 512
    assert config.training.learning_rate == 0.001
    assert config.training.num_epochs == 100
    assert config.seed == 123


def test_config_post_init() -> None:
    """Test configuration post-initialization path creation."""
    config = ReidentificationConfig()
    # Directories should be created
    assert config.training.save_dir.exists()
    assert config.evaluation.output_dir.exists()


def test_config_custom_paths() -> None:
    """Test configuration with custom paths."""
    config = ReidentificationConfig()
    config.training.save_dir = Path("custom/models")
    config.evaluation.output_dir = Path("custom/results")

    # Should be Path objects
    assert isinstance(config.training.save_dir, Path)
    assert isinstance(config.evaluation.output_dir, Path)
