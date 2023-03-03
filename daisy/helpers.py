import numpy as np

def query_kwargs(key, default, **kwargs):

    if key in kwargs.keys():
        return kwargs[key]
    else:
        return default
