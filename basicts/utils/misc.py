import time

def clock(func):
    def clocked(*args, **kw):
        t0 = time.perf_counter()
        result = func(*args, **kw)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        print('%s: %0.8fs...' % (name, elapsed))
        return result
    return clocked