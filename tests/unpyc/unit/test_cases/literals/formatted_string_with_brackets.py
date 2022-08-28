
def _repr_fn(fields):
    return _create_fn('__repr__', ('self',), ['return self.__class__.__qualname__ + f"(' + ', '.join([f'{f.name}={{self.{f.name}!r}}' for f in fields]) + ')"'])