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

def load_data(save_path, split='train'):
	save_path = os.path.join(save_path, split)
	pos_samples = pickle.load(open(os.path.join(save_path, 'lymph_node.pkl'), 'rb'))
	neg_samples = pickle.load(open(os.path.join(save_path, 'no_lymph_node.pkl'), 'rb'))
	samples = []
	samples.extend(pos_samples)
	
	# Balanced Sampling for Training
	# taking all the pos samples --> 9k pos samples with lhymph nodes 
	# neg samples --> 9k neg samples from 14k neg samples ---> 9k neg sample

	# 1. smaller objects --> imbalance in each image

	
	# 2. no lymoh node images >>>>> lymph node images ---> imbalance in the dataset
	# by equal sampling --> tried to remove 2nd level of imbalance

	if split=='train':
		random.shuffle(neg_samples)
		print(len(pos_samples), len(neg_samples))
		# random.shuffle(neg_samples)
		neg_samples = neg_samples[:len(pos_samples)]
	samples.extend(neg_samples)
	if split != 'test':
		random.shuffle(samples)

    return samples
	