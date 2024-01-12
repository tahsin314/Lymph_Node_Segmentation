from DataModule import LNDataModule
from CloudDataset import CloudDataset
from LNDataset import LNDataset
from augmentation import Augmentation

def data_module_creation(train_set, val_set, test_set, dimension, num_class, batch_size):
	aug = Augmentation()
	train_ds = LNDataset(train_set, dim=dimension, num_class=num_class, transforms=aug)
	valid_ds = LNDataset(val_set, dim=dimension, num_class=num_class)
	test_ds = LNDataset(test_set, dim=dimension, num_class=num_class)
	print(len(train_ds), len(valid_ds), len(test_ds))
	data_module = LNDataModule(train_ds, valid_ds, test_ds, batch_size=batch_size)
	return train_ds, valid_ds, test_ds, data_module