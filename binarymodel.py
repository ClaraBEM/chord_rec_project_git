import numpy as np

N_chord = 24;
N_pitch = 12;
chord_binary_model = np.zeros(shape=(N_pitch,N_chord))

maj = np.array([1,0,0,0,1,0,0,1,0,0,0,0])
min = np.array([1,0,0,1,0,0,0,1,0,0,0,0])

for num in range(0,N_chord):
    if num%2 != 0:
        chord_binary_model[:,num] = np.roll(maj,int(num/2))
    else:

        chord_binary_model[:,num] = np.roll(min,int(num/2))

print(chord_binary_model)
