import h5py
import json

h5_path = "final_lstm.h5"
try:
    with h5py.File(h5_path, 'r') as f:
        if 'model_config' in f.attrs:
            config_str = f.attrs['model_config']
            if isinstance(config_str, bytes):
                config_str = config_str.decode('utf-8')
            config = json.loads(config_str)
            print(json.dumps(config, indent=2))
        else:
            print("No model_config attribute found.")
except Exception as e:
    print(f"Error reading h5 file: {e}")
