
class A:

    def __init__(self):
        self.__a = 'test'

    def f(self):
        return self.__a

x = A()
print(x.__a)