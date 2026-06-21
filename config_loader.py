import json
import os

# 加载实验配置文件
def load_experiment_config(experiment_dir):
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
