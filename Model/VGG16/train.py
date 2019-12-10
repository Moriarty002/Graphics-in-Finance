import torch
import torch.nn as nn
import numpy as np
from vgg16 import VGG16
from dataset import IMAGE_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
#import torchvision.models as models
from torchvision import transforms
from pathlib import Path
import copy
import math
from torch.utils.data.sampler import SubsetRandomSampler
#import horovod.torch as hvd
##REPRODUCIBILITY
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#args = parse_args()
#CUDA_DEVICES = args.cuda_devices
#DATASET_ROOT = args.path
CUDA_DEVICES = 0
#DATASET_ROOT = './seg_train'
DATASET_ROOT1 = 'D:/Git/AI-Barista/images/50'
PATH_TO_WEIGHTS = './model_label_classification.pth'

def train(i,train_acc,train_loss):
    
	data_transform = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	
        
#	data_transform = transforms.Compose([
#		transforms.CenterCrop((224, 224)),
#		transforms.ToTensor(),
#		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#	])
	#print(DATASET_ROOT)
	all_data_set = IMAGE_Dataset(Path(DATASET_ROOT1), data_transform)
	
	#print('set:',len(train_set))
	indices = list(range(len(all_data_set)))
	#print('old',indices)
	np.random.seed(3)
	np.random.shuffle(indices)
	#print('new',indices)
	split = 40
	train_idx, valid_idx = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_idx)
	test_sampler = SubsetRandomSampler(valid_idx)
	#print('test')
	#print(test_sampler)
        #train_set, test_set = torch.utils.data.random_split(train_set, [400, 115])
	print('train_set:',len(train_sampler),'test_set:',len(test_sampler))

	train_data_loader = DataLoader(dataset=all_data_set, batch_size=8, shuffle=False, num_workers=0,sampler=train_sampler)
	test_data_loader=DataLoader(dataset=all_data_set,batch_size=8,shuffle=False,num_workers=0,sampler=test_sampler)
	#print(train_set.num_classes)
	'''
	if i==1:
	    model = models.resnet101(pretrained=True)
	    fc_features=model.fc.in_features
	    model.fc=nn.Linear(fc_features,5)
	if i!=1:
	    model=torch.load(PATH_TO_WEIGHTS)'''
	if i==1:
		model=VGG16(num_classes=all_data_set.num_classes)
	elif i!=1:
		model=torch.load(PATH_TO_WEIGHTS)
	model = model.cuda(CUDA_DEVICES)
	model = nn.DataParallel(model, device_ids = [0])
	model.train()           #train

	best_model_params = copy.deepcopy(model.state_dict())    #複製參數
	best_acc = 0.0
	num_epochs = 20
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(params=model.parameters(), lr=0.005, momentum=0.9)

	for epoch in range(num_epochs):
		print(f'Epoch: {epoch + 1}/{num_epochs}')
		print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

		training_loss = 0.0
		training_corrects = 0

		for i, (inputs, labels) in enumerate(train_data_loader):
			inputs = Variable(inputs.cuda(CUDA_DEVICES))
			labels = Variable(labels.cuda(CUDA_DEVICES))			
			print("-----labels-----",labels)

			optimizer.zero_grad()
			
			outputs = model(inputs)
			_ , preds = torch.max(outputs.data, 1)
			#print("outputs\n",outputs)
			print("------preds-----\n",preds)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			training_loss += loss.item() * inputs.size(0)
			#revise loss.data[0]-->loss.item()
			training_corrects += torch.sum(preds == labels.data)
			#print(f'training_corrects: {training_corrects}')
			#if(not(i%10)):
			#	print(f'iteration done :{i}')

		training_loss = training_loss / len(train_sampler)							#train loss
		training_acc =training_corrects.double() /len(train_sampler)				#tarin acc
		#print(training_acc.type())
		#print(f'training_corrects: {training_corrects}\tlen(train_set):{len(train_set)}\n')
		print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')
		train_acc.append(training_acc)		#save each 10 epochs accuracy
		train_loss.append(training_loss)

		if training_acc > best_acc:
			best_acc = training_acc
			best_model_params = copy.deepcopy(model.state_dict())

	model.load_state_dict(best_model_params)	#model load new best parms								#model載入參數
	torch.save(model, f'model_label_classification.pth')	#save model			#存整個model
	return(train_acc,train_loss,test_data_loader)

#if __name__ == '__main__':
	#train()
