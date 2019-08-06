import os
import sys

import pkg_resources
dir_ = pkg_resources.resource_filename(__name__, 'bert')

os.listdir(dir_)
sys.path.insert(0, dir_)

from BertLib.models.BertFEModel import BertFEModel
from BertLib.models.BertFTModel import BertFTModel