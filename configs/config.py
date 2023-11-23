import os

from easydict import EasyDict

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = "/workspace/"  # TODO: change this
# CONF.PATH.BASE = "/Users/irisawa/Documents/competition/sony2023"  # TODO: change this
CONF.PATH.BASE = "/home/user/Documents/competition/sony2023"  # TODO: change this
CONF.PATH.DATASET = os.path.join(CONF.PATH.BASE, "datasets")
CONF.PATH.CONFIG = os.path.join(CONF.PATH.BASE, "configs")
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")
CONF.PATH.SOURCE = os.path.join(CONF.PATH.BASE, "src")
CONF.PATH.ANALYSIS = os.path.join(CONF.PATH.BASE, "analysis")
