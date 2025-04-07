import torch
import torch_dct
import numpy as np
import matplotlib.pyplot as plt


class blue_loss():

    def __init__(self, init_image) -> None:
        self.lower_better = True
        self.original_image_dct = torch_dct.dct_2d(init_image)

        print("original_image_dct shape: ", self.original_image_dct.shape))



    def __call__(self, image, current_iteration, max_iterations):

        # Compute the DCT of the image then compare it to the circle array
        dct_image = torch_dct.dct_2d(image)
        # dct_image[0, :, :self.vertical_cutoff, :self.horizontal_cutoff] = 0

        # Increase the frequencies that we want to keep based on the current iteration
        percentage_cutoff = (current_iteration+1) / max_iterations

        width = image.shape[3]
        height = image.shape[2]

        # Use an inverted exponential function to increase the cutoff

        horizontal_cutoff = int( (1 - np.exp(-2*percentage_cutoff) ) * width)
        vertical_cutoff = int( (1 - np.exp(-2*percentage_cutoff) ) * height)

        # # Test remove the cutoff
        # horizontal_cutoff = width
        # vertical_cutoff = height

        loss = torch.nn.functional.mse_loss(dct_image[0, :, :vertical_cutoff, :horizontal_cutoff], self.original_image_dct[0, :, :vertical_cutoff, :horizontal_cutoff])

        return loss

