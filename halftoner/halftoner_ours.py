import taichi as ti
import taichi.math as tm
import halftoner.particle_simulator as ps
import timeit
import numpy as np
import importlib
from matplotlib import pyplot as plt
import torchvision
import imageio.v3 as iio
import subprocess
import os
import halftoner.helpers as mh
import uuid
import torch

class simulator():

    def __init__(self, rgb_colors, particles_types ) -> None:
        
        self.nominal_field_base = None

        self.nominal_field_ti = None

        self.nominal_field_shape = None
        self.nominal_field_shape_xyz = None

        self.max_velocity = None

        self.rgb_colors = ti.Matrix.field(rgb_colors.shape[0], rgb_colors.shape[1], dtype=float, shape=(), needs_grad = True  )
        self.particles_type = ti.Vector.field(particles_types.shape[1], dtype=ti.f32, shape=(particles_types.shape[0]))

        self.rgb_colors.from_numpy(rgb_colors)
        self.particles_type.from_numpy(particles_types)

        self.colors_no = rgb_colors.shape[1]
        self.types_no = particles_types.shape[0]

        print(f'Initialized with {self.types_no} types and {self.colors_no} colors')

        self.type_field_shape_xyz = None
        
        self.particles_pos = None


    def nominal_field_to_taichi_fields( self, nominal_field ):
        self.nominal_field_base = nominal_field
        self.nominal_field_3d = nominal_field

        self.nominal_field_shape = ti.Vector(self.nominal_field_3d.shape[:3], ti.i32)
        self.nominal_field_shape_xyz = ti.Vector([self.nominal_field_shape[1], self.nominal_field_shape[0], self.nominal_field_shape[2]])

        self.Y, self.X, self.Z, _ = self.nominal_field_3d.shape

        self.nominal_field_ti = ti.Vector.field(self.colors_no, dtype=ti.f32, shape=(self.Y, self.X, self.Z))
        self.nominal_field_ti.from_numpy(self.nominal_field_3d)

        self.init_field = ti.Vector.field(self.colors_no, dtype=ti.f32, shape=(self.Y, self.X, self.Z))

        self.type_field_shape = nominal_field.shape[:3]
        self.type_field_shape_xyz = ti.Vector( [self.type_field_shape[1], self.type_field_shape[0], self.type_field_shape[2]] )

        self.type_field = ti.Vector.field(self.colors_no, dtype=float, shape=self.type_field_shape, needs_grad=True)

        self.color_field = ti.Vector.field(self.colors_no, dtype=float, shape=self.type_field_shape, needs_grad=True)
        self.torch_color_field = self.color_field.to_torch(device="cuda:0").permute(2,3,0,1)

    def create_taichi_fields(self):
        self.particles_per_cell = self.types_no
     
        self.particles_pos = ps.create_particles_fields(self.nominal_field_3d,
                                                              self.types_no,
                                                               True)

    def create_blur_kernel(self, kernel_size = 3, sigma = 1):
        self.torch_blurrer = torchvision.transforms.GaussianBlur(kernel_size, sigma=sigma)

    
    def set_target(self, target):
        self.nominal_field_torch = self.nominal_field_ti.to_torch(device="cuda:0").permute(2,3,0,1)
        # self.blurred_target_torch = self.torch_blurrer(self.nominal_field_torch)

        
    def init_taichi_fields(self, init_image, method='discrete'):
        self.init_field.from_numpy(init_image[:,:,:,np.newaxis])

        # plt.imshow(self.init_field.to_numpy()[:,:,0,:])
        # plt.title('Init field')
        # plt.colorbar()
        # plt.show()

        if method == 'discrete':
            pass
        elif method == 'continuous':
            print('Continuous init')

            ps.init_fields_continuous(self.particles_pos, 
                        self.nominal_field_ti,
                        self.init_field,
                        self.types_no)
            
        elif method == 'halftoned':
            print('Halftoned init')
            ps.init_fields_halftoned(self.particles_pos, 
                        self.nominal_field_ti,
                        self.init_field, 
                        self.types_no)

    def reset_simulator(self, reset_particles = False):
        self.type_field.fill(0)
        self.color_field.fill(0)
        if reset_particles:
            ps.reset_particle_simulator()
       

    def random_kick_positions(self, kick_vector, max_velocity):
        ps.randomly_kick_particles(self.particles_pos, kick_vector, max_velocity)  
   

    def compute_type_field(self, levels = 0):
        ps.forward_particles_to_field_continuous(self.particles_pos, 
                                                self.particles_type, 
                                                self.nominal_field_shape_xyz, 
                                                self.type_field, 
                                                self.type_field_shape_xyz,
                                                self.types_no,
                                                levels)
    
    def show_type_field(self):
        color_field_numpy = np.flip( self.type_field.to_numpy(), axis = 0)
            
        weighted_color_field = color_field_numpy[:,:,0,1] / (color_field_numpy[:,:,0,:].sum(axis=2) + 1e-6)

        plt.imshow(weighted_color_field)
        plt.colorbar()
        plt.show()

    def compute_color_field(self):
        ps.type_field_to_color_field(self.type_field, 
                                self.color_field, 
                                self.rgb_colors )

    def show_color_field(self):
        color_field_numpy = np.flip( self.color_field.to_numpy(), axis = 0)[:,:,0,:]
        plt.imshow(color_field_numpy)
        plt.title('Color field')
        plt.show()

    def curb_your_positions(self, max_position):
        ps.curb_velocities(self.particles_pos, max_position)


class image_halftoner():

    def __init__(self) -> None:
        pass

    def init_simulator(self, rgb_colors, particles_types):
        self.sim = simulator(rgb_colors, particles_types)

    def init_fields(self, target_image, init_image, sim_init_method, blur_kernel_size = 11, blur_sigma = 1, levels=0):
        self.sim.nominal_field_to_taichi_fields( target_image[:,:,np.newaxis,:])
        self.sim.create_taichi_fields()
        self.sim.init_taichi_fields(np.atleast_3d(init_image), method=sim_init_method)
        self.sim.create_blur_kernel(blur_kernel_size, blur_sigma)
        self.sim.set_target(target_image)
        self.levels = levels


    def reset_fields(self, init_image, sim_init_method):
        self.sim.init_taichi_fields(np.atleast_3d(init_image), method=sim_init_method)


    def init_optimizer(self):
        self.loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.loss.grad[None] = 1.0
        self.loss[None] = 0.0

        self.learning_rate = 1e-2


    def run_optimizer(self, iterations, losses_objs, initial_learning_rate = 1e-6, minimum_learning_rate = 1e-6, show=False, continue_from_previous=False):
        ps.show = show
        mh.show = show

        losses = []

        self.loss[None] = 0.0
        self.learning_rate = initial_learning_rate

        for i in range(iterations):
            self.sim.reset_simulator()
            losses_objs.set_current_weights(i)

            with ti.ad.Tape(self.loss, validation=True):
                self.sim.compute_type_field(self.levels)
                self.sim.compute_color_field() 
                ps.forward_custom_loss(self.sim, losses_objs, self.loss, i, iterations)


            # Use cosine annealing to adjust the learning rate
            if not continue_from_previous:
                self.learning_rate = minimum_learning_rate + (initial_learning_rate - minimum_learning_rate) * (1 + np.cos(i / iterations * np.pi)) / 2

            ps.update_field_by_grads(self.sim.particles_pos, self.sim.particles_pos.grad, self.learning_rate , 1)

            # print(f'Loss: {self.loss[None]}')
            losses.append(self.loss[None])

        return self.sim.color_field.to_numpy(), losses

    def estimate_max_loss(self, losses_objs, noise_path):
        noise_image = np.load(noise_path)

        noise_image = np.atleast_3d( mh.tile_and_crop(noise_image, self.sim.nominal_field_shape_xyz) )

        # Stack up the noise image to match the number of colors
        noise_image = np.repeat(noise_image, self.sim.colors_no, axis=2)

        noise_image_torch = torch.tensor(noise_image, device="cuda:0").unsqueeze(0).permute(0, 3, 2, 1)

        for loss_obj in losses_objs:
            if loss_obj['name'] == 'blue':
                # print(f'Computing BLUE loss...')

                loss_obj['loss_estimated_max'] = loss_obj['loss'](noise_image_torch, 1, 1).item()

            else:
                loss_obj['loss_estimated_max'] = loss_obj['loss'](noise_image_torch, self.sim.nominal_field_torch).item()



# Load a grayscale image and convert it to a random image such that the overall 'grayness' is preserved
def grayscale_to_bw_random(img):
    total_gray = int( np.round(np.sum(img) ) )

    new_img = np.zeros(shape=np.prod(img.shape))
    new_img[:total_gray] = 1
    np.random.default_rng().shuffle(new_img)

    return new_img.reshape(img.shape)

# Load a grayscale image and convert it to a random image such that the overall 'grayness' is preserved
def grayscale_to_bw_stochastic(img):
    new_img = np.zeros(shape=img.shape)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if np.random.rand() < img[row, col]:
                new_img[row, col] = 1

    return new_img

# Run SAH
def run_sah(img):
    # Save the image
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    iio.imwrite('sah/input.pgm', img)

    # Run the external program
    subprocess.call(["./sah", 'input.pgm', 'output.pgm' ], cwd='./sah')

    # Load the output
    return iio.imread('sah/output.pgm') / 255

def load_and_run_sah(img_name):
    img = iio.imread(f'pictures/pictures_bw//{img_name}.exr')
    return run_sah(img)


# Run SAH
def run_sah_custom(img):
    # Save the image
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    iio.imwrite('sah/input.pgm', img)

    # Run the external program
    sah_run = subprocess.run(["./sah", 'input.pgm', 'output.pgm', '1'], cwd='./sah', stdout=subprocess.PIPE)

    print(sah_run.stdout)
    # Load the output
    return iio.imread('sah/output.pgm') / 255

def load_and_run_sah_custom(img_name):
    img = iio.imread(f'pictures/pictures_bw//{img_name}.exr')
    return run_sah_custom(img)


# Run Ostromoukhov's algorithm on a given image, save the img as input.pgm to the Ostromoukhov's folder
# run the external program and load the output
def run_ostromoukhov(img,  multiplier = 1):
    # Save the image
    # img = np.clip(img, 0, 1)
    img = np.atleast_3d( (img * multiplier).astype(np.uint8) )

    img_id = uuid.uuid4()
    iio.imwrite(f'ostromoukhov/{img_id}.pgm', img[:,:,0])

    # Run the external program
    subprocess.call(["./varcoeffED", f'{img.shape[1]}', f'{img.shape[0]}', f'{img_id}.pgm', f'{img_id}.pgm' ], cwd='./ostromoukhov')

    # Load the output
    return iio.imread(f'ostromoukhov/{img_id}.pgm') / 255

def load_and_run_ostromoukhov(img_name):
    img = iio.imread(f'pictures/pictures_bw//{img_name}.exr')
    

    return run_ostromoukhov(img)

def load_and_run_ostromoukhov_path(img_path):
    img = iio.imread(img_path)

    return run_ostromoukhov(img)


def generate_init_image(target_image, method, img_name=None, **kwargs):
    if method=='random':
        return grayscale_to_bw_stochastic(target_image)
    elif method=='noise':
        return grayscale_to_bw_random(target_image)
    elif method=='uniform':
        return np.ones_like(target_image) * kwargs['value']
    elif method=='ostromoukhov':
        return run_ostromoukhov(target_image, kwargs['multiplier'])
    elif method == 'sah':
        return run_sah(target_image)
    elif method == 'continuous':
        return target_image
    elif method == 'blue_dither':
        blue_noise = np.load( kwargs['blue_noise_path'] )
        blue_noise_image_sized = mh.tile_and_crop( blue_noise, target_image.shape)
        # blue_noise_image_sized = np.interp(blue_noise_image_sized, (0,1), (0.1, 0.9))

        print(f'Max blue noise  {np.max(blue_noise_image_sized)}\
              Min blue noise {np.min(blue_noise_image_sized)}')

        dithered = np.where((target_image) >= np.atleast_3d(blue_noise_image_sized), 1, 0)
        
        return dithered
    elif method == 'blue_add':
        blue_noise = np.load( kwargs['blue_noise_path'] )
        blue_noise_image_sized = mh.tile_and_crop( blue_noise, target_image.shape)
        # blue_noise_image_sized = np.interp(blue_noise_image_sized, (0,1), (0.1, 0.9))

        dithered = target_image + np.atleast_3d(kwargs['max_noise'] * blue_noise_image_sized)
        dithered/= np.max(dithered)

        return dithered
    elif method == 'mask':
        return kwargs['mask'] * target_image
    else:
        raise Exception(f'Unknown method {method}')