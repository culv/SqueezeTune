import os
import sys

import torch
import torch.nn as nn

import numpy as np

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import matplotlib as mpl
mpl.use('TkAgg') # use TkAgg backend to avoid segmentation fault
import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm

from datasets import eth80, ar_faces, cifar10, data_split

import argparse

# Command line arguments
parser = argparse.ArgumentParser(description='Evaluate trained model on test dataset')

parser.add_argument('--data', type=str, required=True,
	help='dataset to use, either "eth80", "ar_faces", or "cifar10"')


def load_model(model, fname='model.pt'):
	"""Load Pytorch model from fname"""
	script_dir = os.path.dirname(os.path.realpath(__file__))
	load_from = os.path.join(script_dir, fname)
	model.load_state_dict(torch.load(load_from))
	return model

def get_predictions(model, loader):
	"""Gets the model's prediction for each sample in the dataset"""

	# Ensure model is in 'eval' mode
	model.eval()

	# Iterate over data
	# Start by preallocating memory
	labels = np.zeros(len(loader.dataset)).astype(np.int)
	predictions = np.zeros(len(loader.dataset)).astype(np.int)

	i = 0
	for batch_images, batch_labels in tqdm(loader):
		bs = batch_images.shape[0]

		outputs = model(batch_images)
		_, batch_predictions = torch.max(outputs, 1)

		labels[i:i+bs] = batch_labels.numpy()
		predictions[i:i+bs] += batch_predictions.numpy()

		i += bs

	return labels, predictions


def confusion_matrix(ax, labels, predictions):
	"""Generate and plot a confusion matrix for the given predictions and
	ground-truth labels"""

	# Number of classes, and samples per class
	samples_per_class = np.bincount(labels)
	num_classes = samples_per_class.shape[0]

	# Create blank confusion matrix (ground-truth goes left-to-right, predictions
	# go top-to-bottom)
	confusion = np.zeros([num_classes, num_classes])

	# Iterate through labels and predictions to construct confusion matrix
	for l, p in zip(labels, predictions):
		confusion[p, l] += float(1/samples_per_class[l])

	ax = sns.heatmap(confusion)#ax.matshow(confusion, cmap=plt.get_cmap('Wistia'))

	ax.set_ylabel('Predicted')
	ax.set_xlabel('Ground-truth')

def train_val_acc(ax, train_acc, val_acc):
	"""Plot the training and validation accuracy over time (epochs)"""

	epochs = np.arange(0, len(train_acc), 1)

	ax.plot(epochs, train_acc, 'r')
	ax.plot(epochs, val_acc, 'b')

	ax.set_xlabel('Epochs')
	ax.set_ylabel('Accuracy (%)')
	ax.legend(['Average during training', 'Validation'])

	ax.set_ylim([0,100])
	ax.set_xlim([0, len(epochs)-1])

	ax.set_xticks(np.arange(0, len(epochs), 1))
	ax.set_yticks(np.arange(0, 101, 10))

def precision_recall(labels, predictions):
	"""Visualize the precision and recall of each class"""

	# Number of classes, and samples per class
	samples_per_class = np.bincount(labels)
	num_classes = samples_per_class.shape[0]

	# Create blank confusion matrix (ground-truth goes left-to-right, predictions
	# go top-to-bottom)
	confusion = np.zeros([num_classes, num_classes])

	# Iterate through labels and predictions to construct confusion matrix
	for l, p in zip(labels, predictions):
		confusion[p, l] += 1


	# True-positives (TP), false-positives (FP) and false-negatives (FN) for each class
	TP = np.diag(confusion)
	FP = np.sum(confusion, 1) - TP
	FN = np.sum(confusion, 0) - TP

	# Compute recall and precision by class
	recall_by_class = TP/(TP+FN)
	precision_by_class = TP/(TP+FP)


def main():
	# Get command line args
	args = parser.parse_args()
	print(args)

	# Batch size
	batch_size = 64

	seed = 18

	data = args.data

	# Since model was pretrained on ImageNet, normalize using ImageNet statistics
	imagenet_mean = (0.485,0.456,0.406)
	imagenet_std = (0.229,0.224,0.225)

	if data=='eth80':
		val_ratio = 0.1
		test_ratio = 0.1

		_, _, test_set = data_split(eth80(), seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

		val_xfm = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(imagenet_mean, imagenet_std)
			])

		test_set.dataset.transform = val_xfm

		model_path = 'squeezenet_eth80_9.pt'

		train_acc = [73.74, 90.7, 92.38, 94.55, 95.5, 96.26, 96.65, 97.26, 97.64, 97.41]
		val_acc = [88.11, 93.9, 95.12, 96.64, 97.86, 96.95, 97.87, 97.87, 97.87, 98.78]

		print_data = 'ETH-80'

	elif data == 'ar_faces':
		val_ratio = 0.1
		test_ratio = 0.1

		xfm = transforms.Compose([
			transforms.Resize((224, 162)),
			transforms.Pad((31, 0)),
			transforms.ToTensor(),
			transforms.Normalize(imagenet_mean, imagenet_std)
			])

		_, _, test_set = data_split(ar_faces(), seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

		test_set.dataset.transform = xfm

		model_path = 'squeezenet_ar_faces_19.pt'

		train_acc = [5.45, 26.9, 41.65, 49.7, 58.4, 66.5, 72.4, 75.55, 79.5, 83.4, 86.7, 90,
			90.85, 91.5, 93.55, 94.05, 94.5, 95.85, 96.35, 96.7]
		val_acc = [18.67, 28.67, 36, 38.33, 45.67, 51, 56.33, 59.67, 66.67, 70, 71, 75, 75.67,
			78.67, 78.33, 80.33, 82, 84.33, 85.33, 87]

		print_data = 'AR Faces'

	elif data == 'cifar10':
		val_ratio = 0.2
		test_ratio = 0.2

		val_xfm = transforms.Compose([
			transforms.Resize((224,224)),
			transforms.ToTensor(),
			transforms.Normalize(imagenet_mean, imagenet_std)
			])

		_, _, test_set = data_split(cifar10(), seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

		test_set.dataset.transform = val_xfm

		model_path = 'squeezenet_cifar10_4.pt'

		train_acc = [68.53, 78.42, 80.75, 82.01, 82.71]
		val_acc = [77.08, 80.24, 81.57, 82.45, 82.54]

		print_data = 'CIFAR-10'

	# Number of classes
	num_classes = len(test_set.dataset.classes)

	# Loader for dataset
	test_loader = DataLoader(test_set, batch_size=batch_size)

	# Load model
	model = models.squeezenet1_1(pretrained=True)

	# Reshape classification layer to have 'num_classes' outputs
	model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
	model.num_classes = num_classes

	# Replace dropout layer in classifier with batch normalization
	model.classifier[0] = nn.BatchNorm2d(512)

	# Load parameters
	model = load_model(model, model_path)

	print(model.features)
	print(model.classifier)

	# Get labels and predictions
	l, p = get_predictions(model, test_loader)

	
	# Save labels and predictions temporarily
	np.save('_temp_labels.npy', l)
	np.save('_temp_predictions.npy', p)


	l = np.load('_temp_labels.npy')
	p = np.load('_temp_predictions.npy')
	
	sns.set()

	fig, ax = plt.subplots()
	confusion_matrix(ax, l, p)
	ax.set_title('Confusion Matrix for {}'.format(print_data))
	plt.savefig('{}_confusion.png'.format(data), bbox_inches='tight')

	fig2, ax2 = plt.subplots()
	train_val_acc(ax2, train_acc, val_acc)
	ax2.set_title('Accuracy During Training for {}'.format(print_data))
	plt.savefig('{}_accuracy.png'.format(data), bbox_inches='tight')

	# precision_recall(l, p)

	plt.show()

	print('Test acc: ', np.mean(l==p))

if __name__ == '__main__':
	main()