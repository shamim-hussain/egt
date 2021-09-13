import sys
from lib.training.training_base import read_config_from_file
from lib.training.importer import import_scheme

if __name__ == '__main__':
    config = read_config_from_file(sys.argv[1])
    SCHEME = import_scheme(config['scheme'])
    
    training = SCHEME(config)
    training.execute_training()
