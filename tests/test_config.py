"""Tests for pc_rai.config module."""

import pytest
from pathlib import Path


def test_default_config():
    """Test RAIConfig instantiates with correct defaults."""
    from pc_rai.config import RAIConfig

    config = RAIConfig()

    # Roughness radius parameters (Markus et al. 2023)
    assert config.radius_small == 0.175
    assert config.radius_large == 0.425

    # K-NN parameters
    assert config.k_small == 30
    assert config.k_large == 100

    # Classification thresholds
    assert config.thresh_talus_slope == 42.0
    assert config.thresh_overhang == 90.0
    assert config.thresh_cantilever == 150.0
    assert config.thresh_r_small_low == 6.0
    assert config.thresh_r_small_mid == 11.0
    assert config.thresh_r_small_high == 18.0
    assert config.thresh_r_large == 12.0

    # Methods (default is knn only for performance)
    assert config.methods == ["knn"]


def test_rai_class_names():
    """Test RAI class name dictionary."""
    from pc_rai.config import RAI_CLASS_NAMES

    assert len(RAI_CLASS_NAMES) == 8
    assert RAI_CLASS_NAMES[0] == "Unclassified"
    assert RAI_CLASS_NAMES[1] == "Talus"
    assert RAI_CLASS_NAMES[2] == "Intact"
    assert RAI_CLASS_NAMES[3] == "Fragmented Discontinuous"
    assert RAI_CLASS_NAMES[4] == "Closely Spaced Discontinuous"
    assert RAI_CLASS_NAMES[5] == "Widely Spaced Discontinuous"
    assert RAI_CLASS_NAMES[6] == "Shallow Overhang"
    assert RAI_CLASS_NAMES[7] == "Cantilevered Overhang"


def test_rai_class_abbrev():
    """Test RAI class abbreviation dictionary."""
    from pc_rai.config import RAI_CLASS_ABBREV

    assert len(RAI_CLASS_ABBREV) == 8
    assert RAI_CLASS_ABBREV[0] == "U"
    assert RAI_CLASS_ABBREV[1] == "T"
    assert RAI_CLASS_ABBREV[2] == "I"
    assert RAI_CLASS_ABBREV[3] == "Df"
    assert RAI_CLASS_ABBREV[4] == "Dc"
    assert RAI_CLASS_ABBREV[5] == "Dw"
    assert RAI_CLASS_ABBREV[6] == "Os"
    assert RAI_CLASS_ABBREV[7] == "Oc"


def test_rai_class_colors():
    """Test RAI class color dictionary."""
    from pc_rai.config import RAI_CLASS_COLORS

    assert len(RAI_CLASS_COLORS) == 8
    # All should be valid hex colors
    for code, color in RAI_CLASS_COLORS.items():
        assert color.startswith("#")
        assert len(color) == 7


def test_custom_config():
    """Test RAIConfig with custom values."""
    from pc_rai.config import RAIConfig

    config = RAIConfig(
        radius_small=0.2,
        radius_large=0.5,
        k_small=20,
        k_large=80,
        thresh_talus_slope=35.0,
        methods=["radius"],
    )

    assert config.radius_small == 0.2
    assert config.radius_large == 0.5
    assert config.k_small == 20
    assert config.k_large == 80
    assert config.thresh_talus_slope == 35.0
    assert config.methods == ["radius"]


def test_config_output_dir_default():
    """Test default output directory."""
    from pc_rai.config import RAIConfig

    config = RAIConfig()
    assert config.output_dir == Path("./output")


def test_config_up_vector_default():
    """Test default up vector is +Z."""
    from pc_rai.config import RAIConfig

    config = RAIConfig()
    assert config.up_vector == (0.0, 0.0, 1.0)


def test_save_and_load_config(tmp_path):
    """Test saving and loading configuration from YAML."""
    from pc_rai.config import RAIConfig, save_config, load_config

    # Create config with custom values
    original = RAIConfig(
        radius_small=0.2,
        thresh_talus_slope=40.0,
        methods=["knn"],
    )

    # Save to file
    yaml_path = tmp_path / "config.yaml"
    save_config(original, yaml_path)

    # Verify file was created
    assert yaml_path.exists()

    # Load and verify values match
    loaded = load_config(yaml_path)
    assert loaded.radius_small == original.radius_small
    assert loaded.thresh_talus_slope == original.thresh_talus_slope
    assert loaded.methods == original.methods


def test_load_config_file_not_found():
    """Test loading non-existent config raises error."""
    from pc_rai.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))


def test_load_config_empty_file(tmp_path):
    """Test loading empty config returns defaults."""
    from pc_rai.config import RAIConfig, load_config

    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("")

    config = load_config(yaml_path)
    default = RAIConfig()

    assert config.radius_small == default.radius_small
    assert config.thresh_talus_slope == default.thresh_talus_slope
