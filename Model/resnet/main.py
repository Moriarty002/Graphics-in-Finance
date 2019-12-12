import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
from train import train
from test import test

#matplotlib.use('Agg')

train_acc=[]
train_loss=[]
test_acc=[]

# range(1,2) is [1]
for i in range(1, 2):

  (train_acc,train_loss, train_loss2,test_data_loader)=train(i,train_acc,train_loss, CUDA_DEVICES = 0)
  test_acc=test(test_acc,test_data_loader, CUDA_DEVICES = 0)

lentr=len(train_loss)
lentr+=1


'''
lente=len(test_acc)
lente+=1
lente*=10

plt.style.use("ggplot")
plt.figure()
plt.subplot(221)
plt.plot(np.arange(1,lentr),train_acc,label='train_acc')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(loc="lower right")
plt.subplot(222)
plt.plot(np.arange(10,lente,10),test_acc,label='test_acc')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(loc="lower right")

#print("train_loss")
#print(train_loss)

plt.subplot(223)


plt.plot(np.arange(1,lentr),train_loss,label='L1Loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.savefig("lossPlot.png")

plt.plot(np.arange(1,lentr),train_loss2,label='MSELoss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.savefig("lossPlot2.png")
#plt.show()
'''
