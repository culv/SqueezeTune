import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader

import matplotlib as mpl
mpl.use('TkAgg') # use TkAgg backend to avoid segmentation fault
import matplotlib.pyplot as plt

from tqdm import tqdm

from datasets import eth80, ar_faces, cifar10, data_split, channel_mean_std


def save_model(model, fname='model.pt'):
	"""Save Pytorch model to fname in script directory"""
	script_dir = os.path.dirname(os.path.realpath(__file__))
	save_to = os.path.join(script_dir, fname)
	torch.save(model.state_dict(), save_to)

def load_model(model, fname='model.pt'):
	"""Load Pytorch model from fname"""
	script_dir = os.path.dirname(os.path.realpath(__file__))
	load_from = os.path.join(script_dir, fname)
	model.load_state_dict(torch.load(load_from))
	return model

def train_model(model, loader, criterion, optimizer):
	"""Finetune the model"""

	# Keep track of accuracy and average loss
	acc = 0
	avg_loss = 0

	# Ensure model is set to train mode
	model.train()

	# Run through the training set
	i=0
	for images, labels in tqdm(loader):

		# Forward pass of network
		outputs = model(images)
		_, predictions = torch.max(outputs, 1)
		loss = criterion(outputs, labels)

		# Backprop
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Get batch accuracy and loss every 10 iterations
		if (i+1)%10==0:
			batch_acc = float(torch.sum(predictions == labels))/images.shape[0]
			batch_loss = float(loss)

			print('Batch training loss: {}'.format(batch_loss))
			print('Batch training acc: {}%'.format(batch_acc*100))		
		i+=1

		# Update running accuracy and loss
		acc += float(torch.sum(predictions == labels))
		avg_loss += float(loss)*images.shape[0]

	# Stats on the final batch
	batch_acc = float(torch.sum(predictions == labels))/images.shape[0]
	batch_loss = float(loss)

	print('Batch training loss: {}'.format(batch_loss))
	print('Batch training acc: {}%'.format(batch_acc*100))		

	# Stats on the entire epoch
	acc /= len(loader.dataset)
	avg_loss /= len(loader.dataset)

	print('Avg training loss: {}'.format(avg_loss))
	print('Avg training acc: {}%'.format(acc*100))

	return model, acc, avg_loss

def eval_model(model, loader):
	"""Evalute the model on data from a given DataLoader"""

	# Keep track of accuracy
	acc = 0

	# Ensure model is in 'eval' mode
	model.eval()

	# Iterate over data
	i=0
	for images, labels in tqdm(loader):

		outputs = model(images)
		_, predictions = torch.max(outputs, 1)

		acc += float(torch.sum(predictions == labels))

		# Show batch accuracy every 10 iterations
		if (i+1)%10==0:
			print('Batch acc: {}'.format(100*float(torch.sum(predictions == labels))/images.shape[0]))
		i+=1

	# Show accuracy for last batch
	print('Batch acc: {}'.format(100*float(torch.sum(predictions == labels))/images.shape[0]))

	# Show accuracy for whole set
	acc /= len(loader.dataset)
	print('Eval accuracy: {}%'.format(100*acc))

	return acc

def main():
	# Batch size
	batch_size = 64
	epochs = 5

	seed = 18

	data = 'ar_faces'

	# Since model was pretrained on ImageNet, normalize using ImageNet statistics
	imagenet_mean = (0.485,0.456,0.406)
	imagenet_std = (0.229,0.224,0.225)

	# Means and standard deviations for other datasets
	cifar10_mean = (0.4914,0.4822,0.4465)
	cifar10_std = (0.247,0.243,0.261)

	if data=='eth80':
		val_ratio = 0.1
		test_ratio = 0.1

		train_set, val_set, test_set = data_split(eth80(), seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

		train_xfm = transforms.Compose([
			transforms.Resize((224,224)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(imagenet_mean, imagenet_std)
			])

		val_xfm = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(imagenet_mean, imagenet_std)
			])

		train_set.dataset.tranform = train_xfm
		val_set.dataset.transform = val_xfm
		test_set.dataset.transform = val_xfm

	elif data == 'ar_faces':
		val_ratio = 0.1
		test_ratio = 0.1

		xfm = transforms.Compose([
			transforms.Resize((224, 162)),
			transforms.Pad((31, 0)),
			transforms.ToTensor(),
			transforms.Normalize(imagenet_mean, imagenet_std)
			])

		train_set, val_set, test_set = data_split(ar_faces(), seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

		train_set.dataset.transform = xfm
		val_set.dataset.transform = xfm
		test_set.dataset.transform = xfm

	elif data == 'cifar10':
		val_ratio = 0.2
		test_ratio = 0.2

		train_xfm = transforms.Compose([
			# transforms.Resize((224,224)),
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(imagenet_mean, imagenet_std)
			])

		val_xfm = transforms.Compose([
			transforms.Resize((224,224)),
			transforms.ToTensor(),
			transforms.Normalize(imagenet_mean, imagenet_std)
			])

		train_set, val_set, test_set = data_split(cifar10(), seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

		train_set.dataset.tranform = train_xfm
		val_set.dataset.transform = val_xfm
		test_set.dataset.transform = val_xfm

	# Number of classes
	num_classes = len(train_set.dataset.classes)

	# Loaders for each dataset
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_set, batch_size=batch_size)
	test_loader = DataLoader(test_set, batch_size=batch_size)

	# # Get a batch to visualize
	# imgs, labels = next(iter(train_loader))

	# # Make into grid
	# imgs_grid = torchvision.utils.make_grid(imgs).permute(1,2,0).numpy()

	# # Undo normalize
	# mean = np.array(imagenet_mean)[None,None,:]
	# std = np.array(imagenet_std)[None,None,:]
	# imgs_grid = imgs_grid*std+mean

	# # Print labels and filepaths
	# print(labels)
	# # print([test_loader.dataset.dataset.imgs[i][0] for i in test_loader.dataset.indices[0:batch_size]])

	# plt.imshow(imgs_grid)
	# plt.show()
	# sys.exit()

	# Load pretrained model
	model = models.squeezenet1_1(pretrained=True)

	# Reshape classification layer to have 'num_classes' outputs
	model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
	model.num_classes = num_classes

	# Replace dropout layer in classifier with batch normalization
	model.classifier[0] = nn.BatchNorm2d(512)

	print(model.features)
	print(model.classifier)

	# # Resuming training
	# model = load_model(model, 'squeezenet_ar_faces_9.pt')

	# Freeze all parameters
	for p in model.parameters():
		p.requires_grad = False

	# Make classifier and the last 2 layers of the feature extractor trainable
	# for p in model.features[-1].parameters():
	# 	p.requires_grad = True

	# for p in model.features[-2].parameters():
	# 	p.requires_grad = True

	for p in model.classifier.parameters():
		p.requires_grad = True

	trainable_params = [p for p in model.parameters() if p.requires_grad==True]

	# Cross entropy loss function
	criterion = nn.CrossEntropyLoss()

	# Adam optimizer
	optimizer = optim.Adam(trainable_params, lr=10e-4)

	# For each epoch, train model on train set and evaluate on eval set
	for epoch in range(epochs):

		# # Resuming training
		# epoch += 10

		print('Epoch: {}'.format(epoch))

		# Train
		model, _, _ = train_model(model, train_loader, criterion, optimizer)

		# Validate
		val_acc = eval_model(model, val_loader)

		# Save
		save_model(model, fname='squeezenet_{}_{}.pt'.format(data, epoch))

	# Test
	test_acc = eval_model(model, test_loader)

if __name__ == '__main__':
	main()