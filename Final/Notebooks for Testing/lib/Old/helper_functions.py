"""
This module contains functions that might prove helpful for the Facility Location model as build by Tim Romijn

Created in March 2018

author: Tim Romijn
"""

__all__ = ['print_nodes', "sum_attribute"]

def print_nodes(nodes):
    for x in nodes:
#         print(x.type, x.location)
        print(x.__dict__)

    
def sum_attribute(obj_list,attribute):
    return sum(getattr(obj,attribute) for obj in obj_list)