from data.cooking_dataloader import CookingDataset, CookingDataLoader, get_cooking_consistency
from data.openpi_dataloader import OpenPIDataset, OpenPIDataLoader, get_openpi_consistency
from data.trip_dataloader import TRIPDataset, TRIPDataLoader, get_trip_consistency


DATASET_REGISTRY = {
    'recipes': {
        'dataset': CookingDataset,
        'dataloader': CookingDataLoader,
    },
    'openpi': {
        'dataset': OpenPIDataset,
        'dataloader': OpenPIDataLoader,
    },
    'trip': {
        'dataset': TRIPDataset,
        'dataloader': TRIPDataLoader,
    }
}

CONSISTENCY_FN_REGISTRY = {
    'recipes': get_cooking_consistency,
    'openpi': get_openpi_consistency,
    'trip': get_trip_consistency,
}