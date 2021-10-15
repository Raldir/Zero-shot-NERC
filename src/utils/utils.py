
import sys, os
import random
import pickle
import yaml
import math
import torch.utils.data

class RequiredParam(object):
  @staticmethod
  def check(argdict, caller=None):
    for k, v in argdict.items():
      if v is RequiredParam:
        raise Exception(f''''{k:s}' is a required parameter but not provided for '{caller:s}'.''')
