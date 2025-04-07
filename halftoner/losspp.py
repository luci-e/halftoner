import numpy as np

# A loss class that wraps a losses objects but also provides utility functions
# to normalize the losses, furthermore it implements the [] operator to access
# the losses by name


# Example of losses_objs:

# losses_objs = [
#     {   'name' : 'mse',
#         'weight': .98,
#         'loss' : pyiqa.losses.losses.mse_loss,
#         'loss_estimated_max': -1,
#         'lower_better': True,
#     },

#     {   'name' : 'ssim',
#         'weight': .02,
#         'loss' : pyiqa.create_metric('ssim', device=self.device, as_loss=True),
#         'loss_estimated_max': -1,
#         'lower_better': False,
#     },

#     {   'name' : 'blue',
#         'weight' : .2,
#         'loss' : blue_loss(self.target_image_torch, 5),
#         'loss_estimated_max': -1,
#         'lower_better': True,
#     },

#     # {   'name' : 'lpips',
#     #     'weight': .2,
#     #     'loss' : pyiqa.create_metric('lpips', device=self.device, as_loss=True),
#     # },
# ]


class losspp():

    def __init__(self, losses_objs):
        self.losses_objs = losses_objs
        self.interpolation = None

    def __getitem__(self, key):
        for loss_obj in self.losses_objs:
            if loss_obj['name'] == key:
                return loss_obj
            
    # Implement the iterator protocol on the losses_objs
    def __iter__(self):
        self.iter_index = 0
        return self
    
    def __next__(self):
        if self.iter_index < len(self.losses_objs):
            self.iter_index += 1
            return self.losses_objs[self.iter_index - 1]
        else:
            raise StopIteration

    def normalize_losses(self):
        # Normalize the losses
        total_weight = sum([loss_obj['weight'] for loss_obj in self.losses_objs])
        for loss_obj in self.losses_objs:
            loss_obj['weight'] /= total_weight

    def set_target_weights(self, target_weights, interpolation='linear', interp_args=None):
        self.interp_args = interp_args
        self.current_iteration = 0
        self.target_weights = target_weights
        self.starting_weight = np.array( [loss_obj['weight'] for loss_obj in self.losses_objs] )

        self.interpolation = interpolation

        print(f'Interp args are {interp_args}')
        print(f'Starting weight is {self.starting_weight}')
        print(f'Target weight is {self.target_weights}')
        print(f'Interpolation is {self.interpolation}')


    def set_current_weights(self, iter_no):
        if self.interpolation == 'linear':
            self.linear_interpolation(iter_no)
        elif self.interpolation == 'annealing':
            self.annealing_interpolation(iter_no)

    # Annealing interp behaves like an exponential moving average
    def annealing_interpolation(self, iter_no):
        self.current_iteration = iter_no

        if self.current_iteration >= self.interp_args['max_iterations']:
            self.starting_weight = self.target_weights
        else:
            alpha = self.interp_args['alpha']
            self.starting_weight = alpha * self.starting_weight + (1 - alpha) * self.target_weights


        print(f'Current weights: {self.starting_weight}')
        print(f'Target weights: {self.target_weights}')

        for i, loss_obj in enumerate(self.losses_objs):
            loss_obj['weight'] = self.starting_weight[i]
        
    def linear_interpolation(self, iter_no):
        self.current_iteration = iter_no

        if self.current_iteration >= self.interp_args['max_iterations']:
            self.starting_weight = self.target_weights
        else:
            self.starting_weight = self.starting_weight + (self.target_weights - self.starting_weight) * (self.current_iteration / self.interp_args['max_iterations'])

        for i, loss_obj in enumerate(self.losses_objs):
            loss_obj['weight'] = self.starting_weight[i]
    