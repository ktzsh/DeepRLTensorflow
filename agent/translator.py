import json


def export_config(orig_config, implementation):
    config = {}
    for key, value in orig_config.items():
        if implementation == 'OpenAI':
            if key == 'MODEL':
                network_type = value['TYPE']
                if (network_type == 'mlp'):
                    config['MODEL'] = {
                        'TYPE': network_type,
                        'ARGS': {
                            'num_layers': value['MLP']['NUM_LAYERS'],
                            'num_hidden': value['MLP']['NUM_HIDDEN']
                        }
                    }
                elif (network_type == 'conv_only'):
                    config['MODEL'] = {
                        'TYPE': network_type,
                        'ARGS': {
                            'convs': [tuple(x) for x in value['CONVS']]
                        }
                    }
                elif (network_type == 'cnn_small'):
                    config['MODEL'] = {
                        'TYPE': network_type,
                        'ARGS': {}
                    }
                elif (network_type == 'cnn'):
                    config['MODEL'] = {
                        'TYPE': network_type,
                        'ARGS': {}
                    }
                elif (network_type == 'lstm'):
                    config['MODEL'] = {
                        'TYPE': network_type,
                        'ARGS': {
                            'nlstm': value['LSTM_UNITS']
                        }
                    }
                elif (network_type == 'cnn_lstm'):
                    config['MODEL'] = {
                        'TYPE': network_type,
                        'ARGS': {

                            'nlstm': value['LSTM_UNITS']
                        }
                    }
                elif (network_type == 'custom'):
                    config['MODEL'] = {
                        'TYPE': network_type,
                        'ARGS': {
                            'nlstm': value['LSTM_UNITS'],
                            'convs': [tuple(x) for x in value['CONVS']]
                        }
                    }
                else:
                    raise Exception('Unsupported Network Type String')
            elif key == 'LOAD_PATH_PREFIX':
                if value and orig_config['LOAD_FROM_CHECKPOINT']:
                    config['LOAD_PATH'] = {'load_path': value + config['ENV_NAME'] + '.pkl'}
                else:
                    config['LOAD_PATH'] = {}
            else:
                config[key] = value
        else:
            raise Exception('Agent implementation %s not supported yet' % implementation)
    config['AGENT_IMPLEMENTATION'] = implementation
    print(json.dumps(config, indent=4))
    return config
