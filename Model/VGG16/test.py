import torch
#from utils import parse_args
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torchvision.models as models
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from dataset import IMAGE_Dataset
import math
CUDA_DEVICES = 1
DATASET_ROOT2 = '/home/pwrai/0912test_photo_preprocessed'
PATH_TO_WEIGHTS = './model_label_classification.pth'


def test(test_acc,test_data_loader):
    '''data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    test_set = IMAGE_Dataset(Path(DATASET_ROOT2), data_transform)
    data_loader = DataLoader(
        dataset=test_set, batch_size=32, shuffle=True, num_workers=1)'''
    classes = [_dir.name for _dir in Path(DATASET_ROOT2).glob('*')]     #資料夾名稱
    '''
    model=models.resnet101()
    fc_features=model.fc.in_features
    model.fc=nn.Linear(fc_features,5)
    model.load_state_dict(torch.load(PATH_TO_WEIGHTS))'''
    #model=nn.DataParallel(model)
    model=torch.load(PATH_TO_WEIGHTS)

    model = model.cuda(CUDA_DEVICES)
    model.eval()        #test

    total_correct = 0
    total = 0
    class_correct = list(0. for i in enumerate(classes))
    class_total = list(0. for i in enumerate(classes))
    with torch.no_grad():
        for inputs, labels in test_data_loader:
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            print('---labels---')
            print(labels)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)       #預測機率最高的class
            print('---predicted---')
            print(predicted)
            # totoal
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            # batch size
            for i in range(labels.size(0)):
                label =labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy on the ALL test images: %d %%'
          % (100 * total_correct / total))
    test_acc.append(total_correct/total)

    for i, c in enumerate(classes):
        print('Accuracy of %5s(%d photos) : %2d %%' % (
        c,class_total[i], 100 * class_correct[i] / class_total[i]))
    return(test_acc)


#if __name__ == '__main__':
 #   test()
