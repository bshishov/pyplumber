from typing import Dict, Any

import logging
import gc
import sys


__all__ = ['MemoryDump', 'obj_info']


logger = logging.getLogger()


def obj_info(obj, include_referents=True) -> Dict[str, Any]:
    if include_referents:
        referents = [obj_info(r, include_referents=False)
                     for r in gc.get_referents(obj) if hasattr(r, '__class__')]
    else:
        referents = []
    return {
        'id': id(obj),
        'size': sys.getsizeof(obj),
        'repr': repr(obj),
        'class': obj.__class__.__qualname__,
        'referents': referents
    }


class MemoryDump:
    def __init__(self):
        self.objects = []
        self.size = 0
        for obj in gc.get_objects():
            size = sys.getsizeof(obj, 0)
            if not hasattr(obj, '__class__'):
                continue
            referents = [id(ref) for ref in gc.get_referents(obj) if hasattr(ref, '__class__')]
            self.objects.append({
                'id': id(obj),
                'size': size,
                'referents': referents,
                'class': obj.__class__.__qualname__
            })
            self.size += size

    def sort_by_size(self):
        self.objects.sort(key=lambda o: o['size'], reverse=True)
