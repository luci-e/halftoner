from halftoner.halftoner_ours import *
from halftoner.blue_loss import blue_loss
from halftoner.helpers import tile_and_crop
from halftoner.losspp import losspp

import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import pyiqa
import torch

import multiprocessing


class halftoner_worker(multiprocessing.Process):


    def __init__(self, ctp_q, ptc_q, **kwargs ):
        multiprocessing.Process.__init__(self, **kwargs)
        self.ctp_q = ctp_q
        self.ptc_q = ptc_q


    def ht_init(self, 
             image_path, 
             init_method, 
             init_method_args,
             particles_types,
             particles_colors,
             sim_init_method,
             blur_kernel_size = 11,
             blur_sigma = 1.5,):
        
        self.device = torch.device("cuda")

        # Load the image
        self.target_image = np.atleast_3d( iio.imread(image_path).astype(np.float32) ) / 255

        self.target_image_torch = torch.tensor(self.target_image, device=self.device).unsqueeze(0).permute(0, 3, 1, 2)

        self.init_method = init_method
        self.init_method_args = init_method_args
        self.sim_init_method = sim_init_method

        # Generate the initial image
        self.init_image = generate_init_image(self.target_image, init_method, **init_method_args)

        self.halftoner = image_halftoner()
        self.halftoner.init_simulator(particles_colors, particles_types)
        self.halftoner.init_fields(self.target_image, self.init_image, sim_init_method, blur_kernel_size, blur_sigma)
        self.halftoner.init_optimizer()

        self.losses_objs = losspp( [
            {   'name' : 'mse',
                'weight': .98,
                'loss' : pyiqa.losses.losses.mse_loss,
                'loss_estimated_max': -1,
                'lower_better': True,
            },

            {   'name' : 'ssim',
                'weight': .02,
                'loss' : pyiqa.create_metric('ssim', device=self.device, as_loss=True),
                'loss_estimated_max': -1,
                'lower_better': False,
            },

            {   'name' : 'blue',
                'weight' : .2,
                'loss' : blue_loss(self.target_image_torch, 5),
                'loss_estimated_max': -1,
                'lower_better': True,
            },

            # {   'name' : 'lpips',
            #     'weight': .2,
            #     'loss' : pyiqa.create_metric('lpips', device=self.device, as_loss=True),
            # },
        ])

        # self.normalize_losses(self.losses_objs)

    def run(self):
        # We're gonna just wait for messages in the queue that we're going to map to functions
        # each message is a tuple with the function name and the arguments
        while True:
            message = self.ctp_q.get()

            command = message[0]

            # print(message)

            if command == 'ht_init':
                self.ht_init(*message[1:])
            elif command == 'halftone_individual':
                img, loss_history = self.halftone_individual(*message[1:])
                self.ptc_q.put((img, loss_history))
            elif command == 'estimate_max_loss':
                self.estimate_max_loss(message[1])
            elif command == 'update_weights':
                self.set_losses_weights(*message[1:])
            elif command == 'set_target_weight':
                self.set_target_weight(*message[1:])


    def halftone_individual(self, chromosome, id, iterations, save_path=None, reset=True, initial_learning_rate = 1e-5, continue_from_previous=False):        
        try:
            img, loss_history = self.run_optimization(iterations, reset=reset, initial_learning_rate=initial_learning_rate, continue_from_previous=continue_from_previous)

            if save_path is not None:
                save_file = f'{save_path}/{id}.png'
                iio.imwrite(save_file, (img[:,:,0,0]*255).astype(np.uint8))

            return img, loss_history
        except Exception as e:
            print(e)
            print ('Error in halftone_individual')
            print(chromosome)
            return None, None

    def normalize_losses(self, losses_objs):
        losses_objs.normalize_losses()

    def set_losses_weights(self, weights, normalize=False, reset_max=True):
        for i, loss_obj in enumerate(self.losses_objs):
            loss_obj['weight'] = weights[i]

            if reset_max:
                loss_obj['loss_estimated_max'] = -1

        if normalize:
            self.normalize_losses(self.losses_objs)

    def set_target_weight(self, target_weights, interpolation='linear', interp_args=None):
        self.losses_objs.set_target_weights(target_weights, interpolation, interp_args)

    def run_optimization(self, iterations, reset=False, initial_learning_rate = 1e-5, continue_from_previous=False):
        if reset:
            self.halftoner.reset_fields(self.init_image, self.sim_init_method)
        
        return self.halftoner.run_optimizer(iterations, self.losses_objs, initial_learning_rate=initial_learning_rate, continue_from_previous=continue_from_previous)
    
    def estimate_max_loss(self, noise_path):
        self.halftoner.estimate_max_loss(self.losses_objs, noise_path)

    