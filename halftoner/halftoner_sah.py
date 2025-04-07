import numpy as np
import skimage

class SAHer():

    def __init__(self, init_image, target_image, loss_function):
        self.init_image = init_image
        self.target_image = target_image
        self.current_image = np.copy(init_image)

        self.current_temperature = 0.2
        self.anneal_factor = 0.8
        self.temperature_limit = 0.01
        self.iterations_per_loop = np.prod(self.current_image.shape[:2])

        self.last_swap_indices = None

        self.loss_function = loss_function

        self.current_loss = self.loss_function(self.current_image, self.target_image)

        self.losses = []


    def random_swap(self):

        first_index = ( np.random.randint(0, self.current_image.shape[0]), np.random.randint(0, self.current_image.shape[1]) )
        
        swapped = False

        while not swapped:
            second_index = ( np.random.randint(0, self.current_image.shape[0]), np.random.randint(0, self.current_image.shape[1]) )

            if self.current_image[*first_index] != self.current_image[*second_index]:
                swapped = True

        self.current_image[*first_index], self.current_image[*second_index] = self.current_image[*second_index], self.current_image[*first_index]

        self.last_swap_indices = (first_index, second_index)

    def undo_swap(self):
        if self.last_swap_indices is not None:
            self.current_image[self.last_swap_indices[0]], self.current_image[self.last_swap_indices[1]] = self.current_image[self.last_swap_indices[1]], self.current_image[self.last_swap_indices[0]]


    def step(self):
        self.random_swap()

        loss = self.loss_function(self.current_image, self.target_image)

        self.losses.append(loss)

        delta_loss = loss - self.current_loss

        if np.random.rand() < np.exp( np.min( (0.0, -delta_loss / self.current_temperature) ) ):
            self.current_loss = loss
        else:
            self.undo_swap()

        self.current_temperature *= self.anneal_factor


    def optimize(self):
        while self.current_temperature > self.temperature_limit:
            for _ in range(self.iterations_per_loop):
                self.step()

            print(f'Current loss: {self.current_loss}')

        return self.current_image


    def run_once(self):
        for _ in range(self.iterations_per_loop):
            self.step()

        return self.current_image


def SSIM_and_MSE( test_image, target_image ):
    w_g = 0.5
    w_t = 0.5


    ssim =  skimage.metrics.structural_similarity(  target_image, test_image,
                                                gradient=False, data_range=1.0,
                                                win_size=11,
                                                gaussian_weights=False,
                                                sigma=1.5,
                                                full=False,
                                                use_sample_covariance=False,
                                                channel_axis = 2 )

    blurred_test = skimage.filters.gaussian( test_image, sigma=1.5, channel_axis = 2 )

    mse = skimage.metrics.mean_squared_error( target_image, blurred_test )

    return w_g * mse + w_t * (1 - ssim)


    