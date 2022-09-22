# CNN
# Make a valid convolution, same convolution and full convolution for f (image) and g (kernel)
# Using no libraries

# This is NOT a full CNN, just a convolution, full CNN will be made later

# Declare image
f = [[1, 0, 0, 2],
    [1, 4, 1, 0],
    [0, 3, 2, 0], 
    [4, 0, 0, 5]]
f_s = len(f), len(f[0])

# Declare kernel
g = [[2, 1, 6],
    [1, 0, 0],
    [5, 0, 3]]
g_s = len(g), len(g[0])

# Make a valid convolution
def valid_convolution(f, g):
    # Get the shape of the output
        # We are using no padding, so result shape is 2x2
    out_s = (f_s[0] - g_s[0] + 1, f_s[1] - g_s[1] + 1)
    # Make matrix for the output
        # The result is a smaller matrix than the image
    output = [[0 for col in range(out_s[0])] for row in range(out_s[1])]

    # Make convolution
    for i in range(out_s[0]):
        for j in range(out_s[1]):
            for k in range(g_s[0]):
                for l in range(g_s[1]):
                    output[i][j] += f[i + k][j + l] * g[k][l]

    return output

# Make a same convolution
def same_convolution(f, g):
    # Get shape of the output
        # Output shape will be 4x4
    out_s = f_s
    # Make matrix for the output
        # The result is the same shape as the image
    output = [[0 for col in range(out_s[0])] for row in range(out_s[1])]

    # Make matrix for the padded image
        # To make one-layer padding, we add 1 to each side of
        # the image
    f_pad = [[0 for col in range(f_s[0] + 2 * 1)] for row in range(f_s[1] + 2 * 1)]
    # Pad the image
    for i in range(f_s[0]):
        for j in range(f_s[1]):
            # Fill the center of the f_pad matrix with the image
            f_pad[i + 1][j + 1] = f[i][j]

    # Make convolution
    # Iterate over every pixel in the output
    # Add the result of each convolution to its pixel
    for i in range(out_s[0]):
        for j in range(out_s[1]):
            # Iterate over every pixel in the kernel
            # Multiply the 3x3 pixels in the kernel with the 
            # 3x3 pixels in the padded image section
            # Resulting sum of the products is saved in output
            for k in range(g_s[0]):
                for l in range(g_s[1]):
                    output[i][j] += f_pad[i + k][j + l] * g[k][l]

    return output

# Make a full convolution
def full_convolution(f, g):
    # Get shape of the output
        # Output shape will be 6x6
    out_s = (f_s[0] + g_s[0] - 1, f_s[1] + g_s[1] - 1)
    # Make matrix for the output
        # The result is a larger matrix than the image
    output = [[0 for col in range(out_s[0])] for row in range(out_s[1])]

    # Make matrix for the padded image
        # To make circular padding, we add the shape of the 
        # kernel to the shape of the image and subtract 1
        # * 2 because we add to both sides
    f_pad = [[0 for col in range(f_s[0] + 2 * (g_s[0] - 1))] for row in range(f_s[1] + 2 * (g_s[1] - 1))]
    # Pad the image
    for i in range(f_s[0]):
        for j in range(f_s[1]):
            f_pad[i + g_s[0] - 1][j + g_s[1] - 1] = f[i][j]

    # Make convolution
    for i in range(out_s[0]):
        for j in range(out_s[1]):
            for k in range(g_s[0]):
                for l in range(g_s[1]):
                    output[i][j] += f_pad[i + k][j + l] * g[k][l]

    return output



# Print the output
print("\nValid convolution:\n")
print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in valid_convolution(f, g)]))

print("\nSame convolution:\n")
print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in same_convolution(f, g)]))

print("\nFull convolution:\n")
print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in full_convolution(f, g)]))

