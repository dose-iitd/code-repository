import random
random.seed (42)

import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

from queue import PriorityQueue

from DenseNCLoss import DenseNCLoss

'''
buffer_size : int = maximum number of images to store (eg 20)
replay_period : int = entire buffer is replayed after these many images are replaced since it was full or last replayed (eg 5) 
update_period : int = number of images after which the buffer should be updated (eg 10) #*5
num_update : int = number of images that will be added/updated in the buffer per buffer update (eg 1)


Usage:

import buffer

# define hyperparameters
buffer_size = 20
replay_period = buffer_size//2
update_period = 5
num_update = 1 if random else update_period

# define priority function (for example, priority by ncloss is given below)

softmax = nn.Softmax(dim=1)

def priority_fn (image, label, output):
	images = torch.unsqueeze (image, 0)
	output = torch.unsqueeze (output, 0)
	denormalized_image = denormalizeimage(images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
	croppings = torch.ones ((images.shape[0], images.shape[2], images.shape[3]), dtype=torch.float32)
	probs = softmax(output)

	with torch.no_grad():
		densencloss = self.densenclosslayer(denormalized_image,probs,croppings)

	return -float(densencloss.item())


# initialize buffer
 if (self.args.buffer=='random'):
	buffer = buffer.BufferManager (buffer_size, replay_period, update_period,num_update=1, buffer='random')
elif (self.args.buffer=='priority'):
	buffer = buffer.BufferManager (buffer_size, replay_period, update_period,num_update=update_period, buffer='priority')
elif (self.args.buffer=='priority_nc'):
	buffer = buffer.BufferManager (buffer_size, replay_period, update_period,num_update=update_period, buffer='priority',priority_fn=priority_fn)


# update buffer inside loop of dataloader

with torch.no_grad():
	output = self.model (image.cuda()).cpu()

image = image.cpu()
target = target.cpu()
replay_dataset = buffer.update (image, target, output)
  
if(replay_dataset is None):
	continue
buffer_dl = torch.utils.data.DataLoader(replay_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=False, num_workers=self.args.workers)
for i, image_label_pairs in enumerate (buffer_dl):
	images, target = image_label_pairs
	images = images.cuda()
	...
	...

'''

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    entropy = x.cpu().softmax(1) * x.cpu().log_softmax(1)
    return -entropy.sum()/(x.shape[1]*x.shape[2])


def priority_fn (image, label, output):
    '''
    Takes a [3, H, W] shaped single image (pytorch tensor),
    corresponding [H, W] label pytorch tensor, and [19, H, W] 
    shaped model's output tensor to calculate the priority of 
    a DataPoint. Higher the floating point number returned by 
    this function, higher the priority of that datapoint in the buffer.
    
    Return: float: priority of a DataPoint
    '''
    
    return softmax_entropy (output)


class DataPoint:
    def __init__(self, image, label, output, priority):
        '''
        image: individual image: pytorch.tensor of shape [3, H, W]
        label: semantic ground truth masks of above individual image: pytorch.tensor of shape [H, W]
        output: output of the model on the image: pytorch.tensor of shape [19, H, W]
		priority: float as returned by the priority_fn
        '''
        self.image = image
        self.label = label
        self.output = output
        self.priority = priority if (priority is not None) else 0
        # print ('priority = {}'.format(self.priority))
        self.present = False	# if it was present in the buffer
    
    def __lt__(self,other):
        return self.priority < other.priority
    def __le__(self,other):
        return self.priority <= other.priority
    def __eq__(self,other):
        return self.priority == other.priority
    def __ne__(self,other):
        return self.priority != other.priority
    def __ge__(self,other):
        return self.priority >= other.priority
    def __gt__(self,other):
        return self.priority > other.priority
    
    
# The dataset which is returned to the caller of BufferManager.update
class BufferedDataset(Dataset):
	def __init__(self, datapoints):
		self.datapoints = datapoints

	def __len__(self):
		return len(self.datapoints)

	def __getitem__(self, idx):
		image = self.datapoints[idx].image
		label = self.datapoints[idx].label
		return image, label

	def __str__(self):
		return '[' + ','.join ([str(i.priority) for i in self.datapoints]) + ']'

class Buffer(ABC): 
	def __init__ (self, buffer_size: int = 20, num_update: int = 1):
		self._max_size = buffer_size
		self._num_update = num_update
		self._buffer = []

	@abstractmethod
	def update (self, datapoints):
		'''Given the minibatch (type: list of class DataPoint Objects), update the buffer according to the policy.
  		Return: int = number of datapoints updated in the buffer
     	'''
		...

	def get_buffer (self):	# may override
		'''Return the entire buffer in the form of list of class DataPoint Objects in the order they should be replayed to the model'''
		return self._buffer

	def is_full (self):	# may override
		'''Return boolean: if the buffer is full or not.'''
		return len(self._buffer) >= self._max_size

class RandomBuffer(Buffer):
	def __init__ (self, buffer_size: int = 20, num_update: int = 1):
		super().__init__(buffer_size, num_update)
		
	def update (self, datapoints):	# datapoints:= the new minibatch
		# sample random images out of minibatch
		random_indices = random.sample(range(0, len(datapoints)), self._num_update)
		datapoints = list(map(datapoints.__getitem__, random_indices))	# only keep those random datapoints
		# print (f'initial number of images in buffer = {len(self._buffer)}')

		if (len (self._buffer) <= self._max_size - len (datapoints)):    # append all new datapoints into the buffer
			self._buffer.extend (datapoints)

		elif (len (self._buffer) < self._max_size):	# a few datapoints will be appended, and the rest replaced
			orig_buffer_size = len (self._buffer)
			num_extend = self._max_size - orig_buffer_size
			self._buffer.extend (datapoints[0:num_extend])

			# randomly replace datapoints from other than those appended just now
			replace_indices = random.sample(range(0, orig_buffer_size), self._num_update - num_extend)
			for i,index in enumerate(replace_indices):
				self._buffer[index] = datapoints[i]
			
		else:   # buffer full => replace self.__num_update datapoints
			replace_indices = random.sample(range(0, self._max_size), self._num_update)
			for i,index in enumerate(replace_indices):
				self._buffer[index] = datapoints[i]
		# print (f'finally number of datapoints in buffer = {len(self._buffer)}')
		return self._num_update

class PriorityBuffer(Buffer):
    def __init__ (self, buffer_size: int = 20, num_update: int = 1):
        super().__init__(buffer_size, num_update)
        self._buffer = []
    
    def update (self, datapoints):	
        # foreign datapoints have present=False and datapoints already present have present=True
        self._buffer.extend (datapoints)	
        self._buffer = sorted (self._buffer,reverse=True)[0:self._max_size]	# sort according to highest priority and select top self._max_size datapoints
        new_added = 0
        for dp in self._buffer:
            if (not dp.present):
                new_added += 1
                dp.present = True
        return new_added
        

# num_update doesn't make sense in priority buffer	(should be equal to update_period)
class BufferManager: 
	def __init__ (self, buffer_size: int = 20, replay_period: int = 5, update_period: int = 10,num_update: int = 1, buffer: str = 'random',priority_fn=priority_fn):		
		self.size = buffer_size
		self.replay_period = replay_period
		self.update_period = update_period
		self.num_update = num_update
		self.type = buffer
		self.priority_fn = priority_fn
  
		if (self.update_period < self.num_update):
			raise ValueError ("update_period must be greater than or equal to num_update")
		
		if (self.type == 'random'):
			self.buffer = RandomBuffer (self.size, self.num_update)
			
		elif (self.type == 'priority'):
			if (self.num_update is not None and self.update_period != self.num_update):
				print ('num_update != update_period does not make sense for priority buffer. Making num_update = update_period')
				self.num_update = self.update_period
			if (priority_fn is None):
				raise ValueError ('priority_fn must be given in argument to constructor of BufferManager')
			
			self.buffer = PriorityBuffer (self.size, self.num_update)

		else:
			raise NotImplementedError

		self.buffered_datapoints = []
		self.num_replaced = 0	# check if buffer should be replayed or not

	def update (self, images, labels, outputs):
		'''Takes the minibatches (type: tensor, as coming from the dataloader) and 
  		returns the (images,target) Dataset which should be replayed to the model, if any, 
    	otherwise returns None.'''

    	# convert minibatch tensor to list of individual image tensors (ref. https://stackoverflow.com/q/70964264/17800557)
		new_images, new_labels, new_outputs = list (images), list (labels), list (outputs)	
		
		
		if (self.type == 'random'):
			new_datapoints = [DataPoint (image, label, output, None) for (image, label, output) in zip(new_images, new_labels, new_outputs)]
			# for i in range (images.shape[0]):
			# 	self.buffered_datapoints.append (DataPoint (images[i], labels[i], None))
		elif (self.type == 'priority'):
			# for i in range (images.shape[0]):
			# 	self.buffered_datapoints.append (DataPoint (images[i], labels[i], self.priority_fn(images[i], labels[i], outputpriorities[i]))
			new_datapoints = [DataPoint (image, label, output, self.priority_fn(image, label, output)) for (image, label, output) in zip(new_images, new_labels, new_outputs)]
		else:
			raise NotImplementedError
		
		self.buffered_datapoints.extend (new_datapoints)

		final_datapoints = new_datapoints.copy()

		while (len(self.buffered_datapoints) >= self.update_period):
			self.num_replaced += self.buffer.update (self.buffered_datapoints[0:self.update_period])
			self.buffered_datapoints = self.buffered_datapoints[self.update_period:]

			if (self.num_replaced >= self.replay_period and self.buffer.is_full()):
				self.num_replaced = 0
				final_datapoints.extend (self.buffer.get_buffer())
				break
		
		return BufferedDataset (final_datapoints)
		
		    
     
if __name__ == "__main__":
	print ('Running now')
	bs = 5
	h = 5
	w = 10

	buffer = BufferManager (buffer_size=20, replay_period=5, update_period=5, num_update=1, buffer='priority')
	# buffer = BufferManager (buffer_size=20, replay_period=5, update_period=5, num_update=1, buffer='random')

	c = 0
	for i in range (100):
		imgs_from_dl, labels_from_dl, outputs_of_model = torch.rand ((bs, 3, h, w)), torch.rand ((bs, h, w)), torch.rand ((bs, 19, h, w)) 
		c += bs
		dataset = buffer.update (imgs_from_dl, labels_from_dl, outputs_of_model)
		if dataset is not None:
			print ('replay after {} images'.format(c))
			buffer_dl = DataLoader(dataset, batch_size=5)
			for x, img_lbl_pairs in enumerate (buffer_dl):
				images, labels = img_lbl_pairs
				# print (x, len(images), len(labels))
			# print (f'i = {i} | replay_images = \n', )
        
