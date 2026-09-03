"""共享测试夹具：合法最小 config 样板（schema 字段必填集的最短满足形式）。"""

import copy
import json
from pathlib import Path

import pytest

from cynosure.config import DEFAULT_CROSS_MODAL_PAIRS

# 12 个有序 src→tgt 对（脑 MRI 四序列，每序列作 anchor、其余三序列为目标）
CROSS_MODAL_PAIRS = [[src, tgt] for src, tgt in DEFAULT_CROSS_MODAL_PAIRS]

# 组1、生产尺寸的最小合法 config（必填字段全部显式给出）
MINIMAL_CONFIG_DICT: dict = {
    "experiment": {"group": "modal-label"},
    "latent_shape": [4, 64, 64, 32],
    "artifacts": {
        "unet_ckpt": "ckpts/unet.pt",
        "vae_ckpt": "ckpts/vae.pt",
        "net_config_json": "configs/net.json",
        "modality_mapping_json": "configs/modality_mapping.json",
    },
    "reward": {
        "disc_batch_size_k": 4,
        "replay_buffer_capacity": 64,
        "real_pool_manifest": "artifacts/real_pool.json",
        "heldout_real_manifest": "artifacts/heldout_real.json",
        "channel_stats_json": "artifacts/channel_stats.json",
    },
    "schedule": {"seed": 0},
}


@pytest.fixture
def valid_config_dict() -> dict:
    return copy.deepcopy(MINIMAL_CONFIG_DICT)


@pytest.fixture
def valid_config_json(tmp_path: Path, valid_config_dict: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(valid_config_dict), encoding="utf-8")
    return path
