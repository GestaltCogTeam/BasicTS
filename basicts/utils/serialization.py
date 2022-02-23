import pickle

def load_pkl(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data     = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data     = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def dump_pkl(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
