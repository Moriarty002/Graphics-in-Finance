import numpy as np
from train import train
from test import test

train_acc=[]
train_loss=[]
test_acc=[]

for i in range(1,2): 
  print("%d0 epochs"%i)
  (train_acc,train_loss,test_data_loader)=train(i,train_acc,train_loss)
  #test_acc=test(test_acc,test_data_loader)


lentr=len(train_acc)
lentr+=1
lente=len(test_acc)
lente+=1
lente*=10
'''
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
plt.plot(np.arange(1,lentr),train_loss,label='train_loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.savefig("plot.png")
#plt.show()
'''
