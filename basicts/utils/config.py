from typing import Dict

def get_dataset_name(cfg: Dict) -> str:
    """
    Extract the dataset name from the configuration dictionary.

    Args:
        cfg (Dict): Configuration dictionary.

    Returns:
        str: The resolved dataset name or 'Unknown Dataset' if not found.
    """

    # Check if 'DATASET' key exists and is a dictionary
    dataset_cfg = cfg.get('DATASET')
    if dataset_cfg:
        assert isinstance(dataset_cfg, dict), "'DATASET' must be a dictionary."
        return dataset_cfg.get('NAME', 'Unknown Dataset')

    # Collect possible dataset names from 'TRAIN', 'VAL', and 'TEST' sections
    dataset_names = [
        cfg.get(section, {}).get('DATA', {}).get('DATASET', {}).get('NAME')
        for section in ['TRAIN', 'VAL', 'TEST']
    ]

    # Filter out any None values
    dataset_names = [name for name in dataset_names if name]
    dataset_names = list(set(dataset_names))

    # Return concatenated dataset names or 'Unknown Dataset' if none are found
    return '_'.join(dataset_names) if dataset_names else 'Unknown Dataset'
