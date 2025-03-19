"""
Tools for working with generators
"""

from functools import singledispatchmethod
from collections.abc import Iterable, Iterator, Generator
from typing import Any


def enumerate_generator(generator):
    """
    Enumerate a generator without generating all the elements
    """
    # TODO deal with non generators
    index = 0
    for elem in generator:
        yield index, elem
        index += 1

class SubGenerator(Iterator):
    def __init__(self, main_generator, position:int|slice):
        self.main_generator = main_generator
        self.position = position
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.main_generator.buffer):
            self.main_generator.complete_buffer()
        res = self.main_generator.buffer[self.index][self.position]
        self.index += 1
        return res

class PersistentTupleGeneratorWrapper(Iterator):
    def __init__(self, tuple_generator:Iterator[tuple]):
        self.generator = tuple_generator
        self.index = 0
        self.buffer = []

    def complete_buffer(self):
        self.buffer.append(next(self.generator))

    def __next__(self):
        if self.index >= len(self.buffer):
            self.complete_buffer()
        result = self.buffer[self.index]
        self.index += 1
        return result

    def __iter__(self):
        return self

    def __getitem__(self, index:int|slice):
        return self.get_subgenerator(index)

    def get_subgenerator(self, index:int|slice):
        return SubGenerator(self, index)

def allow_sequences_of(*types:type):
    def decorator(func):
        def error_impl(self, it):
            raise TypeError(f"Unsupported type {type(it)} for method {func.__name__} in class {self.__class__.__name__}")
        def default_impl(self, it):
            return func(self, it)
        def iterator_func(self, it:Iterator):
            for i in it:
                yield func(self, i)
        def iterable_func(self, it:Iterable):
            return iterator_func(self, iter(it))
        if Any in types or ... in types or len(types) == 0:
            res = singledispatchmethod(func)
        else:
            res = singledispatchmethod(error_impl)
            for t in types:
                res.register(t)(default_impl)
        res.register(Iterator)(iterator_func)
        res.register(Iterable)(iterable_func)
        return res
    return decorator