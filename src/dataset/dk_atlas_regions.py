import numpy as np

names = np.load('../../synthseg_parcellation_names.npy')

for n in names:
    if n != "Background":
        print(f'"{n.split("ctx-")[1]}",')