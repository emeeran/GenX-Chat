import yaml

def load_config():
    with open("config.yaml", "r") as config_file:
        return yaml.safe_load(config_file)