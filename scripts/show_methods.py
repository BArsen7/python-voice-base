from main import *
from base.core import BaseLogic

for object in BaseLogic.__subclasses__():
    for method in object().get_methods():
        print(f"{object.__name__} -> {method}")

