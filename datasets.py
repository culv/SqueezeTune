import os
import sys
import random

import numpy as np

import torch
from torch.utils.data.dataset import Subset

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib as mpl
mpl.use('TkAgg') # use TkAgg backend to avoid segmentation fault
import matplotlib.pyplot as plt

from tqdm import tqdm


def eth80(transform=transforms.ToTensor()):
	"""Load the ETH-80 dataset into PyTorch. There are 8 unique classes (apple, car, cow, cup,
	dog, horse, pear, tomato) with 410 samples each. Also note that each class has 10 subclasses
	with 41 samples per subclass, corresponding to different orientations

	Args:
		transform = Torchvision transform to use on data
	
	Returns:
		dataset = The dataset
	"""

	# Directories
	script_dir = os.path.dirname(os.path.realpath(__file__))
	data_dir = os.path.join(script_dir, 'data', 'eth80-cropped256')

	# Load dataset in Pytorch
	dataset = datasets.ImageFolder(data_dir, transform=transform)

	# Remove any 'map' images from ETH80
	# First, make a reference copy of 'dataset.imgs' (since items will be removed from 'dataset.imgs')
	orig_img_list = list(dataset.imgs)
	# Iterate through the dataset
	for img in orig_img_list:
		# Get the path
		img_path = os.path.abspath(img[0])
		# Split into filename and path to its directory
		head, _ = os.path.split(img_path)
		# Get the directory name only
		head_dir = os.path.split(head)[1]
		# If it is 'maps', remove this element from the dataset
		if head_dir == 'maps':
			dataset.imgs.remove(img)

	# Combine subclasses into classes (i.e. combine 'apple1' through 'apple10' into 'apple')
	# Since Pytorch sorts classes before constructing the 'class_to_idx' lookup, and ETH80 has
	# 10 subclasses per class, this is as simple as converting each label to floor(label/10)
	dataset.imgs = [(img[0], int(img[1]/10)) for img in dataset.imgs]
	dataset.samples = dataset.imgs

	# New 'class' list and 'class_to_idx' lookup dictionary
	dataset.classes = ['apple', 'car', 'cow', 'cup', 'dog', 'horse', 'pear', 'tomato']
	dataset.class_to_idx = {dataset.classes[i]: i for i in range(len(dataset.classes))}

	# Add a reverse lookup, 'idx_to_class' to go from numeric label to class name
	dataset.idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

	return dataset



def ar_faces(transform=transforms.ToTensor()):
	"""Load the AR faces dataset. Classes are the 100 unique identities, with 26 samples
	per class (corresponding to different lighting conditions, glasses, scarf, date of picture, etc.)

	Args:
		tranform = Torchvision transform to use on images

	Returns:
		dataset = The dataset
	"""

	# Directories
	script_dir = os.path.dirname(os.path.realpath(__file__))
	data_dir = os.path.join(script_dir, 'data', 'AR_database_cropped')

	# Load dataset in Pytorch
	dataset = datasets.ImageFolder(data_dir, transform=transform)

	# 'AR_database_cropped' has a '__MACOSX/test2' and 'test2' directory
	# Data in '_MACOSX/test2' is a duplicate of 'test2' and is not valid, so remove it from the dataset
	# First, make a reference copy of 'dataset.imgs' (since items will be removed from 'dataset.imgs')
	orig_img_list = list(dataset.imgs)
	# Iterate through the dataset
	for img in orig_img_list:
		# Get the path
		img_path = os.path.abspath(img[0])
		# Split into filename and head, head will have the form 'head_path/test2'
		head, _ = os.path.split(img_path)
		# Get the directory that 'test2' is in (will either be '__MACOSX' or 'AR_database_cropped')
		test2_dir = os.path.split(os.path.split(head)[0])[1]
		# If it is '__MACOSX', remove this element from the dataset
		if test2_dir == '__MACOSX':
			dataset.imgs.remove(img)

	# Now, every remaining image in the dataset is labeled as '1' corresponding to 'test2' (since the two
	# subdirectories of 'AR_database_cropped' were '__MACOSX' and 'test2')

	# Change the labels of the images based on the person who is the subject of the image (each image file is
	# 'M-xxx-yy.bmp' or 'W-xxx-yy.bmp' where 'xxx' is the person's number and 'M'/'F' specifies male/female)
	
	# 'xxx' is 001-050 for males and females. To ensure unique identities, map 'M-xxx'->'xxx'-1 and 'W_xxx'->49+'xxx'
	# meaning there will be 100 classes total (one for each identity) and 26 samples per class

	# Iterate through the data and change labels based on the filename mapping described above
	new_imgs = []
	for img in dataset.imgs:
		img_path = os.path.abspath(img[0])
		_, fname = os.path.split(img_path)

		fname_parts = fname.split('-')

		label = int(fname_parts[1])-1
		if fname_parts[0] == 'W':
			label += 50

		new_imgs.append((img[0], label))

	dataset.imgs = new_imgs
	dataset.samples = new_imgs

	# Create new 'classes' list and 'class_to_idx' lookup
	dataset.classes = [i for i in range(100)]
	dataset.class_to_idx = {i: i for i in range(100)}

	return dataset


def cifar10(transform=transforms.ToTensor()):
	"""Load the CIFAR-10 dataset

	Args:
		transform = Torchvision transform for images

	Returns:
		train = Training set
		val = Validation set
	"""

	# Directories
	script_dir = os.path.dirname(os.path.realpath(__file__))
	data_dir = os.path.join(script_dir, 'data', 'cifar10')

	# Load training set, downloading if necessary
	dataset = datasets.CIFAR10(data_dir, train=True, transform=transform, download=True)

	# Split into train, val, and test sets

	# Add 'classes' list and 'class_to_idx' lookup dictionary to both sets
	classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 
		'horse', 'ship', 'truck']
	class_to_idx = {classes[i]: i for i in range(len(classes))}

	dataset.classes = classes
	dataset.class_to_idx = class_to_idx

	return dataset


def data_split(dataset, val_ratio=0.1, test_ratio=0.1, seed=1234):
	"""Splits a dataset into separate training, validation, and testing sets, where the size
	of the val and test sets are (roughly) given by val_ratio and test_ratio. Splitting is
	done so that class priors of training and validation sets match the original data (this
	is also known as 'stratified sampling').

	Note: size of val/test set may be bigger than ratio*original_data_size since this value is
	rounded up to an integer

	Note: still need to update this function to be more efficient (i.e. remove for-loops)
	Note: still need to add checks to make sure val and test sets don't exceed data size for
		extreme case where val_ratio+test_ratio=1

	Args:
		dataset = The dataset to split
		val_ratio = The amount of the given dataset to use for the validation set, must be 0-1
			(default 0.1)
		test_ratio = The amount of the given data set to use for the test set, must be 0-1
			(default 0.1)

			* Also require: val_ratio+test_ratio<=1 (if =1, then there will be no training set!)

		seed = Seed for random shuffling of dataset (used for reproducibility) (default 1234)

	Returns:
		train = The training dataset
		val = The validation dataset
		test = The test dataset
	"""

	# How you grab the labels will depend on what type of Pytorch Dataset object 'dataset' is
	# (i.e. ImageFolder/DatasetFolder or not)

	# For fun, check the method resolution order (MRO) of 'dataset'
	print('Dataset object\'s inheritance: ', type(dataset).__mro__)

	# Determine what kind of Dataset object it is, then grab labels
	# Warning: currently this will break for anything other than an ImageFolder or CIFAR10 train set
	if isinstance(dataset, datasets.CIFAR10):
		labels = dataset.train_labels
	elif isinstance(dataset, datasets.ImageFolder):
		labels = [img[1] for img in dataset.imgs]
	else:
		error('Dataset not supported yet')

	# Calculate class priors, (number in class)/(size of dataset)
	idcs = [i for i in range(len(dataset))]
	samples_per_class = np.bincount(np.array(labels))
	priors = samples_per_class/len(labels)

	# Number of samples in each class for val and test set 
	val_per_class = np.ceil(samples_per_class*val_ratio).astype(np.int)
	test_per_class = np.ceil(samples_per_class*test_ratio).astype(np.int)

	# Copy and shuffle the labels and corresponding indices to randomize before splitting
	shuffled_labels = list(labels)
	shuffled_idcs = list(idcs)
	random.Random(seed).shuffle(shuffled_labels)
	random.Random(seed).shuffle(shuffled_idcs)

	# Iterate through, grabbing indices for each class to place in validation set
	# until the desired number is reached
	val_idcs = []
	val_counts = np.zeros(val_per_class.shape)

	for i, l in zip(shuffled_idcs, shuffled_labels):
		# Check if validation set quota has been reached yet for this class
		if val_counts[l] < val_per_class[l]:
			val_idcs.append(i)
			val_counts[l] += 1

		# Check if stopping point is reached
		if (val_counts == val_per_class).all():
			break

	# Repeat for test set
	test_idcs = []
	test_counts = np.zeros(test_per_class.shape)
	for i, l in zip(shuffled_idcs, shuffled_labels):
		# Check if this index is already in val set
		if i in val_idcs:
			continue

		# Check if test set quota has been reached yet for this class
		if test_counts[l] < test_per_class[l]:
			test_idcs.append(i)
			test_counts[l] += 1

		# Check if stopping point is reached
		if (test_counts == test_per_class).all():
			break

	# Get train indices too (all the remaining samples not in val or test)
	train_idcs = [j for j in idcs if j not in val_idcs+test_idcs]

	# Split the data
	train = Subset(dataset, train_idcs)
	val = Subset(dataset, val_idcs)
	test = Subset(dataset, test_idcs)

	return train, val, test

def channel_mean_std(dataset, batch_size=64):
	"""Calculate the mean and standard deviation for each color channel of
	an image dataset

	Args:
		dataset = The dataset
		batch_size = Batch size for iterating through entire dataset (default 64)

	Returns:
		mean = A triplet of the RGB means, or a float for grayscale
		std = A triplet of the RGB standard deviations, or a float for grayscale
	"""

	# Make a loader for the data
	loader = DataLoader(dataset, batch_size=batch_size)

	# Number of color channels and size of dataset
	ch = next(iter(loader))[0].shape[1]
	n = len(dataset)

	# Iterate through data, computing mean and variance of each batch and summing
	# to running mean and running variance (this utilizes the fact that the mean is
	# equal to the mean of the batch means, and similarly for the variance)
	mean = torch.zeros(3)
	var = torch.zeros(3)

	for batch, _ in tqdm(loader):
		# Ratio of batch size to entire dataset (for weighting purposes)
		r = float(batch.shape[0]/n)
		# Reshape batch so that it is just a set of vectors, one per channel, whose entries
		# are a concatenation of all that channel's values in all the batch images
		ch_vectors = torchvision.utils.make_grid(batch, padding=0).view(ch, -1)

		# Add to running mean and variance
		mean.add_(torch.mean(ch_vectors, 1)*r)
		var.add_(torch.var(ch_vectors, 1)*r)

	return mean, var.sqrt()

def main():
	# Check dataset loading
	# data = ar_faces()
	data = eth80(transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]))
	# data = cifar10()

	# Check splitting (sizes and that sets are disjoint)
	val_ratio = 0.2
	test_ratio = 0.2

	tr, v, t = data_split(data, seed=4, val_ratio=val_ratio, test_ratio=test_ratio)	

	# Check sizes
	print(len(tr))
	print(len(v))
	print(len(t))

	# Check if sets are disjoint
	print(set(tr.indices).isdisjoint(v.indices))
	print(set(tr.indices).isdisjoint(t.indices))
	print(set(t.indices).isdisjoint(v.indices))

	# Check label representation
	try:
		tr_priors = np.bincount([tr.dataset.imgs[i][1] for i in tr.indices])/len(tr.indices)
		t_priors = np.bincount([t.dataset.imgs[i][1] for i in t.indices])/len(t.indices)
		v_priors = np.bincount([v.dataset.imgs[i][1] for i in v.indices])/len(v.indices)
	except:
		tr_priors = np.bincount([tr.dataset.train_labels[i] for i in tr.indices])/len(tr.indices)
		t_priors = np.bincount([t.dataset.train_labels[i] for i in t.indices])/len(t.indices)
		v_priors = np.bincount([v.dataset.train_labels[i] for i in v.indices])/len(v.indices)

	print(tr_priors)
	print(t_priors)
	print(v_priors)

	# Check reproducibility
	tr2, v2, t2 = data_split(data, seed=4, val_ratio=val_ratio, test_ratio=test_ratio)
	print(tr2.indices==tr.indices)
	print(v2.indices==v.indices)
	print(t2.indices==t.indices)

	sys.exit()

	# Check channel mean and std finder
	_, cifar_test = cifar10()
	mean, std = channel_mean_std(eth80())#ar_faces())#cifar_test)
	print(mean)
	print(std)

if __name__ == '__main__':
	main()