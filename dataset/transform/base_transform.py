

class BaseTransform(object):

    def __init__(self) -> None:
        pass

    def __call__(self, results):
        pass

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'()'
        return repr_str
