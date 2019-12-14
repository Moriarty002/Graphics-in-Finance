import os
import copy
import math
import numpy as np

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"   # "%d" % CUDA_DEVICES

import torch
import torch.nn as nn
# from vgg16 import VGG16
from dataset import IMAGE_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
from pathlib import Path
from torch.utils.data.sampler import SubsetRandomSampler
#import horovod.torch as hvd
##REPRODUCIBILITY
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#args = parse_args()
#CUDA_DEVICES = args.cuda_devices
#DATASET_ROOT = args.path
REG_OUTPUT = 1
BATCH_SIZE = 8  
EPOCH_SIZE = 1
#DATASET_ROOT = './seg_train'
DATASET_ROOT1 = "../images/mix"
PATH_TO_WEIGHTS = './Model_all_best.pth'
RB_PATH_TO_WEIGHTS = './Model_recently_best.pth'

def train(i,train_acc,train_loss, CUDA_DEVICES = 0, DEVICE_IDS = [0]):
    data_transform = transforms.Compose([
        #transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    #print(DATASET_ROOT)
    all_data_set = IMAGE_Dataset(Path(DATASET_ROOT1), data_transform)
    
    #print('set:',len(train_set))
    indices = list(range(len(all_data_set)))
    #print('old',indices)
    np.random.seed(1)
    np.random.shuffle(indices)
    #print('new',indices)
    split = math.ceil(len(all_data_set)*0.1)  # extract 10% dataset as test-set
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(valid_idx)
    #print('test')
    #print(test_sampler)
    #train_set, test_set = torch.utils.data.random_split(train_set, [400, 115])
    print('train_set:',len(train_sampler),'test_set:',len(test_sampler))

    train_data_loader = DataLoader(
        dataset=all_data_set, 
        batch_size=BATCH_SIZE , 
        shuffle=False, 
        num_workers=0,
        sampler=train_sampler)

    test_data_loader=DataLoader(
        dataset=all_data_set,
        batch_size=BATCH_SIZE ,
        shuffle=False,
        num_workers=0,
        sampler=test_sampler)

    #print(train_set.num_classes)

    if i==1:
        model = models.resnet101(pretrained=True)
        #fc_features=model.fc.in_features
        #model.fc=nn.Linear(fc_features,5)
        # f=lambda x:math.ceil(x/32-7+1)
        # my_output_module = nn.Linear(f(19)*f(19)*2048, REG_OUTPUT)

        # model.fc = my_output_module
        # model.fc=nn.Linear(f(256)*f(256)*2048, REG_OUTPUT)
        model=model.cuda()
        model=nn.DataParallel(model, device_ids = DEVICE_IDS)
    if i!=1:
        model=torch.load(PATH_TO_WEIGHTS)
    '''if i==1:
        model=VGG16(num_classes=all_data_set.num_classes)
    elif i!=1:
        model=torch.load(PATH_TO_WEIGHTS)'''
    # model = model.cuda(CUDA_DEVICES)
    model.train()       #train

    best_model_params = copy.deepcopy(model.state_dict())    #複製參數
    best_acc = 0.0
    num_epochs = EPOCH_SIZE
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    train_loss = []
    best_loss = math.inf

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

        training_loss = 0.0
        # training_corrects = 0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.cuda() #CUDA_DEVICES)
            labels = labels.cuda() #CUDA_DEVICES)

            optimizer.zero_grad()
            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # sm = F.softmax(outputs,dim=1)
            # print("======== Softmax ========")
            # print(sm.data)
            # print("=========================")
            #print("preds:"+str(preds))
 
            if i*BATCH_SIZE%1 == 0:

                print("\n\n\n\n\n||||||||||||||||||||| BATCH-%d |||||||||||||||||||||\n" % i)
                print("\n\t=================== Labels =====================\n\t")
                print('\t'+str(labels))
                print("\n\t================= Predictions ==================\n\t")
                print('\t'+str(preds))
                print("\n\t================= Batch Loss ===================\n")
                print(f"\tTraining: {loss.data:.2f}")
                print("\n\t================= Epoch Loss ===================\n")
                print(f'\tTraining:', train_loss) 

            progress = i*BATCH_SIZE / len(train_sampler)
            print(f"\t[Training Progress]: {progress:.4f}% [Batch Loss]: {loss.data:.2f}", end='\r')
     
            training_loss += loss.item() * inputs.size(0)

        # Calulate Loss and MSELoss in current epoch       
        training_loss = float(training_loss) / len(train_sampler)

        # train_acc.append(training_acc)        #save each 10 epochs accuracy
        train_loss.append(float(training_loss))

        print("########################\nFinish Epoch\n#########################\n")

        if training_loss < best_loss:
            
            best_loss = training_loss
            best_model_params = copy.deepcopy(model.state_dict())
            torch.save(model, RB_PATH_TO_WEIGHTS)    #save model 存整個model

        print("Best Loss: %.2f" % best_loss)

    model.load_state_dict(best_model_params)    #model load new best parmsmodel載入參數
    torch.save(model, PATH_TO_WEIGHTS)    #save model 存整個model
    
    return([] ,train_loss, 0, test_data_loader)

#if __name__ == '__main__':
    #train()
