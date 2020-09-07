"""
Some methods to accompany the illustrative notebook

@author: Angel Villar-Corrales
"""

def print_pairs(pairs, n_pairs=8, prepend=" "*4):
    """
    Displaying list of tuples in a nice way

    Args:
    -----
    pairs: list
        list of tuples [(k1,v1), (k2,v2), ...]
    n_pairs: integer
        number of tuples to display in one row
    """

    print(f"{prepend}", end="  ")
    for i,(k,v) in enumerate(pairs):
        print(f"{k}:{v}", end="  ")
        if(i % n_pairs == 0 and (i > 0 or n_pairs == 1 )):
            print("")
            print(f"{prepend}", end="  ")

    return
