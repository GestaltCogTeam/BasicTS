
import inspect

def test(a, b):
    c = 1
    return c

variate_name = inspect.signature(test).parameters.keys()
for name in variate_name:
    print(name)
    