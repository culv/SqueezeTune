# SqueezeTune

Finetuning SqueezeNet to achieve high accuracies without a GPU!

### The Problem

Most deep learning is done on powerful GPUs, but what if you don't have a GPU? Even worse, what if your computer's CPU isn't all that powerful either?

### Finetuning

If you're working on a computer vision problem and find yourself lacking computing power, **finetuning** (a.k.a. **transfer learning**) is an excellent method for training up a convolutional neural network (CNN) on your data without having to start from scratch. *In fact, this should probably be a go-to method regardless of your computing situation as it helps with many other issues, like having a small dataset!*

Finetuning simply means **re-using** the weights of a **pre-trained** network (VGGNet, ResNet, Inception, or some other popular architecture) that has been trained on a large computer vision dataset (usually ImageNet), and retraining the last layer (or last several layers) on your own data. Re-using weights lets you take advantage of the rich set of features the model has learned over the course of days (maybe even weeks) of training on the powerful GPU clusters of well-funded researchers or companies. All you have to do is:

* Load the model and reshape the classification layer of the network
    
    *This is as simple as changing the number of outputs of the network to the number of classes in your data (this may require looking at the architecture beforehand to see what kind of layer, e.g. Conv2d or Linear, is used):*
    
```python
# First, you'll need access to torchvision's pre-trained models and PyTorch's neural network
# modules (i.e. your "building blocks")
from torchvision import models
import torch.nn as nn

# Specify the number of classes you're reshaping to
num_classes = 10

# Load your choice of pre-trained network and reshape the classification layer. For most
# architectures, like ResNet and VGGNet, the classification layer is fully-connected (nn.Linear)
# which can be reshaped by changing the 'out_features' attribute of the nn.Linear module or
# replacing it entirely
network = models.resnet18(pretrained=True)
network.fc.out_features = num_classes

# Or alternatively...
in_feats = network.fc.in_features
network.fc = nn.Linear(in, num_classes)

# For SqueezeNet or other all-convolutional networks, the classification layer will be
# convolutional (nn.Conv2d). To change the output features you can update the 'out_channels'
# attribute of the nn.Conv2d module, or replace it
network = models.squeezenet1_1(pretrained=True)
network.classifier[1].out_channels = num_classes

# Or alternatively...
in_chans = network.classifier[1].in_channels
k = network.classifier[1].kernel_size
s = network.classifier[1].stride
network.classifier[1] = nn.Conv2d(in_chans, num_classes, kernel_size=k, stride=s)

# With the line below you can print the architecture of your network. This can give you
# insight on the types of layers and what you need to change
print(network.modules)
```
* Decide which parameters you want to tune
    
    *Usually you'll just tune the final layer, although you may want to also tune the last couple layers of the feature generator (i.e. convolutional layers). For example, for SqueezeNet 1.1:*
    
```python
# Freeze all parameters in the network
for p in network.parameters():
    p.requires_grad = False
    
# Make classification layer trainable
for p in network.classifier.parameters():
    p.requires_grad = True
    
# If you want to make last layer of feature generator trainable. Similarly for the last two,
# or three layers you would use the slice [-2:-1] or [-3:-1] 
for p in network.features[-1].parameters():
    p.requires_grad = True
```
Now that your model is ready to go, just train it on your data! Also note that you can swap out any other parts of the network you want to (for example, I replaced SqueezeNet's dropout layer with a batch normalization layer to improve regularization and reduce training time).

So clearly finetuning can save loads of time and computation. But there's still a problem: **what if you can't even *load* these networks into memory due to their large size?** For example, VGG-19 (the 19-layer variant of VGGNet) takes ~6GB to load into memory. We need a smaller network, but we don't want to sacrifice accuracy...

### SqueezeNet

SqueezeNet<sup>1</sup> is the answer to our problem. Its creators saw the same problem: high-accuracy networks are too large which makes them infeasible for many applications where memory is limited. They employed several strategies to reduce model size and achieve high accuracies. First, they replaced 3x3 conv filters with 1x1 conv filters as much as possible and minimized the number of input channels to these filters to cut down on parameters. Second, they waited until late in the network to downsample which has been shown to improve accuracy<sup>2</sup>.

This led to the development of the **Fire module**, a building block that uses 1x1 filters to reduce input channels followed by a mixture of 1x1 and 3x3 filters.

<p align="center">
<img src="https://github.com/culv/SqueezeTune/blob/master/images/squeezenet_fire_module.PNG" title="Fire module")
</p>

SqueezeNet stacks these blocks together to create a model with **AlexNet level accuracy** while maintaining **~0.5MB model size**. This makes it perfect for our needs!

They combined these ideas into a re-usable block called a **Fire module** that uses 1x1 filters to reduce the number of input channels

<sup>1</sup>[*SqueezeNet: AlexNet-level accuracy with 50X fewer parameters and <0.5MB model size*, Iandola et al.](https://arxiv.org/abs/1602.07360)

<sup>2</sup>[*Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*, He et al.](https://arxiv.org/abs/1502.01852)

### Results

SqueezeNet was finetuned on three datasets: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [ETH-80](http://people.csail.mit.edu/jjl/libpmk/samples/eth.html) and [AR faces](http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html). Each datasets was split into a train, validation, and test set using *stratified* sampling, which ensures each subset has the same class ratios as the original dataset. The train set was used to update model parameters, validation set was used to monitor training and adjust hyperparameters, and test set was used to evaluate the fully-trained model. In addition to test accuracy, a confusion matrix was used visualize *precision* and *recall*.

##### CIFAR-10

This dataset contains 10 classes: airplanes, automobiles, birds, cats, deers, dogs, frogs, horses, ships, and trucks. Data augmentation, including a random zoom/crop and random horizontal flip.

The model was finetuned for 5 epochs and achieved a test accuracy of **82.59%**. From the confusion matrix below we can also see the model had excellent precision and recall.

<p align="center">
<img src="https://github.com/culv/SqueezeTune/blob/master/images/cifar10_accuracy.png" width="46%" title="Training (CIFAR-10)"> <img src="https://github.com/culv/SqueezeTune/blob/master/images/cifar10_confusion.png" width="42.5%" title="Confusion matrix (CIFAR-10)">
</p>

##### ETH-80

This dataset contains 8 classes: apples, pears, tomatoes, cows, dogs, horses, cups, and cars. Each class has 10 sub-classes. The only data augmentation used for this dataset was random horizontal flips.

The model was finetuned for 10 epochs and achieved a test accuracy of **99.08%**. The confusion matrix indicates excellent precision and recall.

<p align="center">
<img src="https://github.com/culv/SqueezeTune/blob/master/images/eth80_accuracy.png" width="46%" title="Training (ETH-80)"> <img src="https://github.com/culv/SqueezeTune/blob/master/images/eth80_confusion.png" width="42.5%" title="Confusion matrix (ETH-80)">
</p>

##### AR Faces

This dataset contains 100 classes corresponding to 100 different individuals' faces. Random horizontal flips were used to augment this data.

The model was finetuned for 20 epochs and achieved a test accuracy of **79.03%**. The confusion matrix indicates good precision and recall, however we can also see which identities were consistently mistaken for another.

<p align="center">
<img src="https://github.com/culv/SqueezeTune/blob/master/images/ar_faces_accuracy.png" width="46%" title="Training (AR faces)"> <img src="https://github.com/culv/SqueezeTune/blob/master/images/ar_faces_confusion.png" width="42.5%" title="Confusion matrix (AR faces)">
</p>









