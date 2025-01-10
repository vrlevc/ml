import numpy as np

# Create a one-dimensional array
one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(one_dimensional_array)

# Create a two-dimensional array
two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
print(two_dimensional_array)

# Create a sequence of integers
sequence_of_integers = np.arange(5, 12)
print(sequence_of_integers)

# Generate random integers between 50 and 100
random_integers_between_50_and_100 = np.random.randint(low=50, high=101, size=(6,))
print(random_integers_between_50_and_100)

# Generate random floats between 0 and 1
random_floats_between_0_and_1 = np.random.random((6,))
print(random_floats_between_0_and_1) 

# Shift random floats to be between 2 and 3
random_floats_between_2_and_3 = random_floats_between_0_and_1 + 2.0
print(random_floats_between_2_and_3)

# Scale random integers to be between 150 and 300
random_integers_between_150_and_300 = random_integers_between_50_and_100 * 3
print(random_integers_between_150_and_300)

# Task: Create a feature array and corresponding label array with noise
# Create a feature array from 6 to 20
feature = np.arange(start=6, stop=21)
print(feature)

# Create a label array based on the feature array
label = feature * 3 + 4
print(label)

# Add random noise to the label array
noise = np.random.random((label.size,)) * 4 - 2
print(noise)
label = label + noise
print(label)