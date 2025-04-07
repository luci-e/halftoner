import taichi as ti
import taichi.math as tm
import numpy as np
import torch
import os
import imageio.v3 as iio
import halftoner.helpers as mh
import matplotlib.pyplot as plt
import skimage as ski
import torchvision

show = False
total_iterations = 0
potato = 0

limit = 1

ti.init(arch=ti.cuda, device_memory_fraction=0.9,
        print_ir=False, debug=False, kernel_profiler=True)

# Create a particles field where each cell contains and upsampling number of particles
# additionally create a field that contains the color of each particle
# the initial field has shape (N, M) and it contains the type of each cell


def create_particles_fields(nominal_field, particles_type_no, needs_grad=False):
    R, C, L, _ = nominal_field.shape

    particles_no = R*C*L*particles_type_no

    print(f'Field has shape {nominal_field.shape} and we have {particles_type_no} particle types, \
     for a total of {particles_no} particles')

    particles_pos = ti.field(dtype=float, shape=(particles_no), needs_grad=needs_grad)

    return particles_pos


@ti.kernel
def init_fields_halftoned(particles_pos: ti.template(), 
                nominal_field: ti.template(), 
                init_field: ti.template(), 
                particles_type_no: ti.int32):

    print(f'Nominal field shape is {nominal_field.shape}')
    
    R, C, L = nominal_field.shape
    dist = (limit/10)

    print(f'There are currently {particles_type_no} particle types')

    for col, row, layer in ti.ndrange(C, R, L):

        particle_no = (col + row * C + layer * C * R) * particles_type_no

        for t in range(particles_type_no):
            selected = init_field[row, col, layer][t]
            original = nominal_field[row, col, layer][t]

            position_z = -(1-original) * dist

            if selected > 0:
                position_z = ti.random()
            else:
                position_z = -ti.random() - 1e-6

            particles_pos[particle_no+t] = position_z

@ti.kernel
def init_fields_continuous(particles_pos: ti.template(), 
                nominal_field: ti.template(), 
                init_field: ti.template(),
                particles_type_no: ti.int32):
    
    R, C, L = nominal_field.shape

    # print(f'There are currently {particles_type_no} particle types')

    for col, row, layer in ti.ndrange(C, R, L):

        particle_no = (col + row * C + layer * C * R) * particles_type_no

        for t in range(particles_type_no):
            position_z = -(1-init_field[row, col, layer][t]) * limit/10 - 1e-3

            particles_pos[particle_no+t] = position_z

# Convert a particles field to a multidimensional field of integers that contains the weighted particle type by distance
@ti.kernel
def particles_to_field_continous(particles_pos: ti.template(), 
                                 particles_type: ti.template(), 
                                 nominal_field_shape_xyz: ti.types.vector(3, int), 
                                 type_field: ti.template(), 
                                 type_field_shape_xyz: ti.types.vector(3, int)):
        
    types_no = len( type_field[0, 0, 0])

    # print(f'Found {types_no} types')

    # print(f'There are currently {particles_pos.shape[0]} particles that is {particles_pos.shape[0]//types_no} particles per type')

    for i in ti.grouped(particles_pos):

        particle_group_no = i // types_no
        particle_t = i % types_no

        # print(f'Processing particle {i}')

        col = particle_group_no % nominal_field_shape_xyz[0]
        row = (particle_group_no // nominal_field_shape_xyz[0]) 
        layer = particles_pos[i]

        if layer >= 0.0 and layer < limit:
            type_field[row, col, 0] += particles_type[particle_t]

@ti.kernel
def particles_to_field_continous_multilevel(particles_pos: ti.template(), 
                                 particles_type: ti.template(), 
                                 nominal_field_shape_xyz: ti.types.vector(3, int), 
                                 type_field: ti.template(), 
                                 type_field_shape_xyz: ti.types.vector(3, int),
                                 particles_type_no: int,
                                 levels: float):
        
    types_no = particles_type_no

    # print(f'Found {types_no} types')

    # print(f'There are currently {particles_pos.shape[0]} particles that is {particles_pos.shape[0]//types_no} particles per type')

    for i in ti.grouped(particles_pos):

        # print(f'Processing particle {i}')

        particle_group_no = i // types_no
        particle_t = i % types_no

        # print(f'Processing particle {i}')
        
        #TODO THE BUG IS HERE the i is no longer the index of the particle because 
        # we need to account for the number of types

        col = particle_group_no % nominal_field_shape_xyz[0]
        row = (particle_group_no // nominal_field_shape_xyz[0]) 
        layer = particles_pos[i]
        
        if layer >= 0.0 and layer < limit:
            level_step = limit / levels
            dist = ti.abs(limit - layer)

            level = float( int( dist / level_step ) )

            weight = (levels - level) / levels

            type_field[row, col, 0] += particles_type[particle_t] * weight


@ti.kernel
def particles_to_field_continous_grads(particles_pos: ti.template(), 
                                       particles_type: ti.template(), 
                                       nominal_field_shape_xyz: ti.types.vector(3, int), 
                                       type_field: ti.template(), 
                                       type_field_shape_xyz: ti.types.vector(3, int)):
    
    types_no = len( type_field[0, 0, 0])

    for i in ti.grouped(particles_pos):

        particle_group_no = i // types_no
        particle_t = i % types_no

        # print(f'Processing particle {i}')
        
        #TODO THE BUG IS HERE the i is no longer the index of the particle because 
        # we need to account for the number of types

        col = particle_group_no % nominal_field_shape_xyz[0]
        row = (particle_group_no // nominal_field_shape_xyz[0]) 
        layer = particles_pos[i]

        z_distance = limit - layer
        # nearnesss = ti.exp(-z_distance**2)

        distance_derivative = -1
        nearnesss_derivative = -2 * z_distance * ti.exp(-z_distance**2)
        derivative_dist = distance_derivative * nearnesss_derivative

        ti.atomic_add( particles_pos.grad[i],  (type_field.grad[row, col, 0] * particles_type[particle_t]).sum() * derivative_dist )


                
@ti.ad.grad_replaced
def forward_particles_to_field_continuous(particles_pos: ti.template(), 
                                          particles_type: ti.template(), 
                                          nominal_field_shape_xyz: ti.types.vector(3, int), 
                                          type_field: ti.template(), 
                                          type_field_shape_xyz: ti.types.vector(3, int),
                                          particles_type_no: int,
                                          levels: float):
    if levels == 0:    
        particles_to_field_continous(particles_pos, particles_type, nominal_field_shape_xyz, type_field, type_field_shape_xyz)
    else:
        particles_to_field_continous_multilevel(particles_pos, particles_type, nominal_field_shape_xyz, type_field, type_field_shape_xyz, particles_type_no, levels)


@ti.ad.grad_for(forward_particles_to_field_continuous)
def backwards_particles_to_field_continuous(particles_pos: ti.template(), 
                                            particles_type: ti.template(), 
                                            nominal_field_shape_xyz: ti.types.vector(3, int), 
                                            type_field: ti.template(), 
                                            type_field_shape_xyz: ti.types.vector(3, int),
                                            particles_type_no: int,
                                            levels: int):   
    particles_type_no = len( type_field[0, 0, 0])
    
    if show:
        # print(f'Type field grads shape: {type_field.grad.shape}')

        for i in range(particles_type_no):
            plt.imshow(type_field.grad.to_numpy()[:,:,0,i])
            plt.title(f'Type field grads particle type {i}')
            plt.colorbar()
            plt.show()

    # print(f'Computing positions grads...')

    particles_to_field_continous_grads(particles_pos, particles_type, nominal_field_shape_xyz, type_field, type_field_shape_xyz)

    # print(f'Particles pos grads raw shape: {particles_pos.grad.shape}')


    if show:

        pos_grads_numpy = particles_pos.grad.to_numpy()

        pos_grads_numpy = pos_grads_numpy.reshape((nominal_field_shape_xyz[1], 
                                                  nominal_field_shape_xyz[0], 
                                                  particles_type_no))
        

        for t in range(particles_type_no):
            plt.imshow(pos_grads_numpy[:,:,t])
            plt.title(f'Particles pos grads particle z {t}')
            # plt.axis('off')
            plt.colorbar()
            plt.show()
    
    

# Convert a type field to an st and alfa field that can be rendered
# sts and alfa contain the st and alfa values for each type in the given order
@ti.kernel
def type_field_to_color_field(type_field: ti.template(), 
                                color_field: ti.template(),
                                rgb_colors: ti.template()):
    
    for i in ti.grouped(type_field):
        color_field[i] = ( rgb_colors[None] @  type_field[i] )



@ti.ad.grad_replaced
def forward_custom_loss(    sim,
                            losses_objs, # a list of objects in the form {name, weight, loss, buffer}
                            loss: ti.template(),
                            current_iteration,
                            max_iterations
                            ):
    
    if show:
        print(f'Sim color field shape is {sim.color_field.shape}')
        print(f'Torch color field shape is {sim.torch_color_field.shape}')
        print(f'Target field has shape {sim.nominal_field_torch.shape}')

    sim.torch_color_field = sim.color_field.to_torch(device="cuda:0").permute(2,3,0,1)
    sim.torch_color_field.requires_grad = True

    torch_loss = torch.tensor(0.0, device="cuda:0")
    blurred_img = sim.torch_blurrer(sim.torch_color_field)

    # We additionally check if the loss estimated max field is set, if not we set it 
    # and use it to normalize the loss

    for loss_obj in losses_objs:
        if loss_obj['name'] == 'blue':
            # print(f'Computing BLUE loss...')

            blue_loss = loss_obj['loss'](blurred_img, current_iteration, max_iterations)
            # custom_loss = loss_obj['loss'](sim.torch_color_field, current_iteration, max_iterations)

            if loss_obj['loss_estimated_max'] == -1:
                loss_obj['loss_estimated_max'] = blue_loss.item()
                print(f'Estimated max is {loss_obj["loss_estimated_max"]}')


            loss[None] += loss_obj['weight'] * blue_loss.item() / loss_obj['loss_estimated_max']
            torch_loss += loss_obj['weight'] * blue_loss / loss_obj['loss_estimated_max']

            loss_obj['loss_estimated_max'] = blue_loss.item()

            # print(f'Blue loss is {blue_loss.item() * loss_obj["weight"] / loss_obj["loss_estimated_max"]}')

        else:
            # print(f'Computing {loss_obj["name"]} loss...')

            # custom_loss = loss_obj['loss'](sim.torch_color_field, sim.nominal_field_torch)
            custom_loss = loss_obj['loss'](blurred_img, sim.nominal_field_torch)

            if loss_obj['loss_estimated_max'] == -1:
                loss_obj['loss_estimated_max'] = custom_loss.item()
                print(f'Estimated max is {loss_obj["loss_estimated_max"]}')


            if loss_obj['lower_better']:
                loss[None] += loss_obj['weight'] * custom_loss.item() / loss_obj['loss_estimated_max']
                torch_loss += loss_obj['weight'] * custom_loss / loss_obj['loss_estimated_max'] 
            else:
                loss[None] += loss_obj['weight'] * (1-custom_loss.item() / loss_obj['loss_estimated_max'] ) 
                torch_loss += loss_obj['weight'] * (1-custom_loss / loss_obj['loss_estimated_max'] ) 

            loss_obj['loss_estimated_max'] = custom_loss.item()

            # print(f'{loss_obj["name"]} loss is {custom_loss.item() * loss_obj["weight"] / loss_obj["loss_estimated_max"]}')

    torch_loss.backward()
   

@ti.ad.grad_for(forward_custom_loss)
def backward_custom_loss(   sim,
                            losses_objs, # a list of objects in the form {name, weight, loss, buffer}
                            loss: ti.template(),
                            current_iteration,
                            max_iterations
                            ):
    
    sim.color_field.grad.fill(0)

    if show:
        print(f'Sim color field grad shape is {sim.color_field.grad.shape}')

        torch_color_field_cpu = sim.torch_color_field.grad.cpu()

        for color in range(sim.colors_no):
            plt.imshow(torch_color_field_cpu[0, color, :, :])
            plt.title(f'Color {color} of torch color field')
            plt.colorbar()
            plt.show()


    temp_grad_buffer = torch.nn.functional.normalize(sim.torch_color_field.grad)

    sim.color_field.grad.from_torch(temp_grad_buffer.permute(2,3,0,1))


def reset_particle_simulator():
    pass

@ti.kernel
def init_sim_field(sim_field: ti.template(), 
                   init_values: ti.template()):
    
    for i in ti.ndrange((0, sim_field.shape[1])):
        sim_field[0, i] += init_values[i]


@ti.kernel
def update_field_by_grads(initial_field: ti.template(), 
                          grads: ti.template(), 
                          learning_rate: float,
                          max: float
                          ):
    k = 0.01

    for i in ti.ndrange((0, initial_field.shape[0])):
        initial_field[i] -= learning_rate *grads[i]





# @ti.kernel
# def randomly_kick_particles_all(particles_vel_init: ti.template(), 
#                             max_velocity: ti.template()):
#     for i in ti.ndrange((0, particles_vel_init.shape[0])):
#         particles_vel_init[i] += ti.Vector([ti.random()-0.5, ti.random()-0.5, ti.random()-0.5]) * max_velocity

# @ti.kernel
# def randomly_kick_particles(particles_vel_init: ti.template(),
#                             kick_vector: ti.template(),
#                             max_velocity: ti.template()):
#     for i in ti.ndrange((0, particles_vel_init.shape[0])):
#         particles_vel_init[i] += ti.Vector([ti.random()-0.5, ti.random()-0.5, ti.random()-0.5]) * max_velocity * kick_vector[i]

# @ti.kernel
# def matrix_field_diff(a: ti.template(), 
#                       b: ti.template(), 
#                       diff: ti.template()):
    
#     for r, c, l in ti.ndrange(*a.shape):
#         diff[r, c, l] = (a[r, c, l] - b[r, c, l])

# @ti.kernel
# def sum_to_matrix_field(a: ti.template(), 
#                       b: ti.template()):
    
#     for r, c, l in ti.ndrange(*a.shape):
#         a[r, c, l] += b[r, c, l]


# @ti.kernel
# def matrix_field_ratio(a: ti.template(), 
#                       b: ti.template(), 
#                       ratio: ti.template()):
    
#     for r, c, l in ti.ndrange(*a.shape):
#         ratio[r, c, l] = (a[r, c, l] / b[r, c, l])

# @ti.kernel
# def compute_max( array_field: ti.template(),
#                  max: ti.template()):
#     for i in ti.grouped(array_field):
#         max[None] = ti.atomic_max(max[None], array_field[i])


# @ti.kernel
# def normalize_array( array_field: ti.template(),
#                      max: ti.template()):
#     R, C, L = array_field.shape

#     for r, c, l in ti.ndrange(*array_field.shape):
#         array_field[r, c, l] /= max[None]

# @ti.kernel
# def normalize_array_to_1( array_field: ti.template() ):
    
#     # print(f'Shape of array field is {array_field[0,0,0].get_shape()}')

#     max_val = ti.Vector.zero( dt=ti.f32, n=len( array_field[0,0,0]) )
#     # print(f'Max val is {max_val}')

#     for i in ti.grouped(array_field):
#         max_val = ti.atomic_max(max_val, ti.abs(array_field[i]))
    
#     # print(f'Max val is {max_val}')

#     for i in ti.grouped(array_field):
#         array_field[i] /= (max_val+1e-20)

# @ti.kernel
# def in_place_scalar_matrix_mult( scalar: float,
#                         matrix: ti.template()):
#     for i in ti.grouped(matrix):
#         matrix[i] *= scalar


# @ti.kernel
# def copy_linear_to_array( linear_field: ti.template(),
#                           array_field: ti.template()):
#     R, C, L = array_field.shape

#     for r, c, l in ti.ndrange(*array_field.shape):
#         array_field[r, c, l] = linear_field[ ( r * C ) + ( c ) + ( l * C * R ) ]

# @ti.kernel
# def copy_array_to_linear( array_field: ti.template(),
#                           linear_field: ti.template()):
#     R, C, L = array_field.shape

#     for r, c, l in ti.ndrange(*array_field.shape):
#         linear_field[ ( r * C ) + ( c ) + ( l * C * R ) ] = array_field[r, c, l]


# @ti.kernel
# def array3_mean( array_field: ti.template()) -> ti.types.vector(3, float):
#     R, C, L = array_field.shape

#     mean = ti.Vector.zero( dt=ti.f32, n=3 )

#     # print(f'Mean has shape {mean.get_shape()}')

#     for i in ti.grouped(array_field):
#         mean += array_field[i]

#     mean /= (R*C*L)

#     return mean

# @ti.kernel
# def array_vector_division( array: ti.template(),
#                            vector: ti.template()):
#     for i in ti.grouped(array):
#         array[i] /= vector[i]
