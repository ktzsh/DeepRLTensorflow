import yaml
import json

def get_config(config_file='cfg/atari.yaml'):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as e:
            raise Exception(str(e))
    return config
