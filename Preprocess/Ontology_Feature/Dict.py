
try:
	import configparser as ConfigParser
except ImportError:
	import ConfigParser


patterns = {}
fpath = "./Data/Dictionary.ini"



def load_lists(fpath):
    config = ConfigParser.ConfigParser()
    with open(fpath, encoding='utf-8') as f:
        config.readfp(f)
    for ind_type in config.sections():
        try:
            ind_pattern = config.get(ind_type, 'pattern')
        except:
            continue
        if ind_pattern:
            patterns[ind_type] = ind_pattern
    return patterns




