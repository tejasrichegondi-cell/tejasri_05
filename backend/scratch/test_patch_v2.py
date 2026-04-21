import os
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense

class PatchedDTypePolicy:
    @classmethod
    def from_config(cls, config):
        return config.get('name', 'float32')

class PatchedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        kwargs.pop('optional', None)
        # Handle dtype if it is still a dict
        dtype = kwargs.get('dtype')
        if isinstance(dtype, dict) and dtype.get('class_name') == 'DTypePolicy':
            kwargs['dtype'] = dtype.get('config', {}).get('name', 'float32')
        super().__init__(*args, **kwargs)

class PatchedDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        # Handle dtype if it is still a dict
        dtype = kwargs.get('dtype')
        if isinstance(dtype, dict) and dtype.get('class_name') == 'DTypePolicy':
            kwargs['dtype'] = dtype.get('config', {}).get('name', 'float32')
        super().__init__(*args, **kwargs)

# The complex config for dtype
dtype_config = {
    "module": "keras",
    "class_name": "DTypePolicy",
    "config": {
        "name": "float32"
      },
    "registered_name": None
}

config = {
    'batch_shape': [None, 1, 300], 
    'dtype': dtype_config, 
    'sparse': False, 
    'ragged': False, 
    'name': 'input_layer', 
    'optional': False
}

print("Attempting to instantiate PatchedInputLayer with dtype config...")
try:
    layer = PatchedInputLayer(**config)
    print(f"Success! Layer dtype: {layer.dtype}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Failed: {e}")

print("\nTesting PatchedDTypePolicy.from_config...")
try:
    res = PatchedDTypePolicy.from_config(dtype_config['config'])
    print(f"Result: {res}")
except Exception as e:
    print(f"Failed: {e}")
