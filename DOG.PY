import time

class Animal(object):
    def __init__(self, name):
        print('__init__方法被调用')
        self.__name = name
    def __del__(self):
        print("__del__方法被调用")
        print("%s对象马上被干掉了..." % self.__name)

dogg = Animal("大狗狗")
dogg = Animal("小貓貓")
dogg=Animal("123")

time.sleep(10)
print('結束')
