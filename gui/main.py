import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

import dearpygui.dearpygui as dpg
import imageio.v3 as iio
from PIL import Image
import numpy as np
import os
import threading
import sys
import pyiqa
import uuid

sys.path.append('../')

from halftoner.halftoner_worker import halftoner_worker
from halftoner import blue_loss

# A class used to wrap the halftoner_worker class
class worker():

    def __init__(   self,
                    image_path, 
                    init_method, 
                    init_method_args,
                    particles_types,
                    particles_colors,
                    sim_init_method,
                    blur_kernel_size = 11,
                    blur_sigma = 1.5,):
        self.ctp_q = multiprocessing.Queue()
        self.ptc_q = multiprocessing.Queue()

        print('Starting halftoner worker')

        self.ht_worker = halftoner_worker(self.ctp_q, self.ptc_q, daemon=True)
        self.ht_worker.start()
        self.ctp_q.put(['ht_init', 
                        image_path,
                        init_method,
                        init_method_args,
                        particles_types,
                        particles_colors,
                        sim_init_method,
                        blur_kernel_size,
                        blur_sigma])
        
        self.lock = threading.Lock()


current_weights = np.ones(3)

init_method = 'blue_add'
init_method_args = { 'blue_noise_path': './noise/blue_noise_0_1.npy',
           'max_noise': 1}

particles_types = np.array([[1]])
particles_colors =  np.array([[1]])

sim_init_method = 'continuous'


# Global variables
original_image = None
halftone_buffer = [None, None]
halftone_locks = [threading.Lock(), threading.Lock()]
halftone_timestamps = [1, 1]
halftone_img_timestamp = 1
displayed_image_timestamp = 0
annealing_alpha = 0.1
max_iters = 200

half_worker = None
init_complete = False

update_continuous = False
halftone_complete = False
cont_iterations = 200
learning_rate = 1e-5

device = 'cuda'

halftoner_thread = None

old_image = None
old_comparison_counter = 0
old_comparison_max = 100

def load_image(sender, app_data):
    global original_image, modified_image, half_worker, losses_objs
    filepath = app_data['file_path_name']

    print(f'Loading image from {filepath}')

    if not filepath:
        return
    
    with dpg.texture_registry():

        if original_image is not None:
            dpg.delete_item("original_texture")
            dpg.delete_item("modified_texture")


        unmodified_image = iio.imread(filepath)
        original_image = Image.fromarray( unmodified_image )

        print(f'Original image has mode {original_image.mode}')

        # Convert the image to RGBA if it is not already
        if original_image.mode != 'RGBA':
            original_image = original_image.convert('RGBA')

        print(f'Original image has shape {original_image.size}')

        image_np = np.array(original_image).astype(np.float32) / 255.0
        halftone_buffer[0] = image_np.copy()
        halftone_buffer[1] = image_np.copy()

        width = image_np.shape[1]
        height = image_np.shape[0]

        # Set the window size to accomodate two images side by side
        dpg.set_viewport_width(width*3+200)
        dpg.set_viewport_height(height+300)

        dpg.add_static_texture(width=width, height=height, default_value=image_np.flatten(), tag="original_texture")
        dpg.add_dynamic_texture(width=width, height=height, default_value=image_np.flatten(), tag="modified_texture")
        dpg.add_dynamic_texture(width=width, height=height, default_value=image_np.flatten(), tag="old_texture")
        dpg.add_dynamic_texture(width=width, height=height, default_value=image_np.flatten(), tag="diff_texture")

        half_worker = worker(
            filepath,
            init_method,
            init_method_args,
            particles_types,
            particles_colors,
            sim_init_method
        )

        half_worker.ctp_q.put(['estimate_max_loss', './noise/blue_noise_0_1.npy'])

        global init_complete
        init_complete = True


    with dpg.window(label="Original Image", width=width+20, height=height+50, pos=(0, 0)):
        dpg.add_image("original_texture")

    with dpg.window(label="Halftoned Image", width=width+20, height=height+5000, pos=(width+20, 0)):
        dpg.add_image("modified_texture")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Halftone", callback=do_halftone)
            dpg.add_button(label="Reset simulation", callback=reset_simulation)

        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Continuous Halftone", tag="continuous_halftone", callback=toggle_continuous)
            dpg.add_button(label="Start halftoner thread", callback=start_halftoner_thread)

        dpg.add_slider_int(label="Iterations", tag="iterations", min_value=1, max_value=100, default_value=50, callback=update_iterations)

        dpg.add_slider_float(label='Learning rate', tag='learning_rate', min_value=-6, max_value=-3, default_value=-5, callback=update_learning_rate)

        dpg.add_slider_float(label='Alpha', tag='annealing_alpha', min_value=0, max_value=1, default_value=0.1, callback=update_alpha)
        dpg.add_slider_int(label='Max Iterations', tag='max_iters', min_value=0, max_value=10000, default_value=200, callback=update_max_iters)

        dpg.add_slider_float(label="Weight 1", tag="weight_1", min_value=0.0, max_value=1000.0, default_value=1.0, callback=update_weights)
        dpg.add_slider_float(label="Weight 2", tag="weight_2", min_value=0.0, max_value=1000.0, default_value=1.0, callback=update_weights)
        dpg.add_slider_float(label="Weight 3", tag="weight_3", min_value=0.0, max_value=1000.0, default_value=1.0, callback=update_weights)

    with dpg.window(label='Old Image', width=width+20, height=height+50, pos=((width+20)*2, 0)):
        dpg.add_image("old_texture")

    with dpg.window(label='Diff Image', width=width+20, height=height+50, pos=((width+20)*3, 0)):
        dpg.add_image("diff_texture")

def reset_simulation():
    half_worker.ctp_q.put(['update_weights', current_weights, True, True])
    do_halftone(reset=True)

def update_alpha():
    global annealing_alpha
    annealing_alpha = dpg.get_value("annealing_alpha")
    print(f'Annealing alpha is {annealing_alpha}')

def update_max_iters():
    global old_comparison_max
    old_comparison_max = dpg.get_value("max_iters")
    print(f'Max iterations is {max_iters}')

def update_learning_rate():
    global learning_rate
    learning_rate = 10**dpg.get_value("learning_rate")

def update_weights():
    current_weights[0] = dpg.get_value("weight_1")
    current_weights[1] = dpg.get_value("weight_2")
    current_weights[2] = dpg.get_value("weight_3")

    print(f'Current weights are {current_weights}')

    # half_worker.ctp_q.put(['update_weights', current_weights, False, False])

    # if not update_continuous:
    #     print(f'Setting immediate weights to {current_weights}')
    #     half_worker.ctp_q.put(['update_weights', current_weights, True, True])
    # else:
    #     print(f'Setting target weights to {current_weights}')
    half_worker.ctp_q.put(['set_target_weight', current_weights, 'annealing', {'alpha': annealing_alpha, 'max_iterations': max_iters}])

    print(f'Current weights are {current_weights}')

def toggle_continuous():
    global update_continuous
    update_continuous = not update_continuous

    print(f'Update continuous is {update_continuous}')


def start_halftoner_thread():
    global halftoner_thread
    print('Starting halftoner thread')
    halftoner_thread = threading.Thread(target=do_halftone_continuous)
    halftoner_thread.start()

def update_iterations():
    global cont_iterations
    cont_iterations = dpg.get_value("iterations")
    print(f'Continuous iterations is {cont_iterations}')

def do_halftone(reset=False, continue_from_previous=False):
    global half_worker

    if half_worker.lock.acquire(blocking=False):
        # Start a thread to do the halftoning and wait on the queue
        # When it's done set the halftone_complete flag to True

        if update_continuous:
            iterations = cont_iterations
            continue_from_previous = True
        else:
            iterations = 200


        def wait_for_halftone():
            global halftone_buffer, halftone_locks, halftone_complete, halftone_timestamps, halftone_img_timestamp

            half_worker.ctp_q.put(['halftone_individual', current_weights,  uuid.uuid4(), iterations, None, reset, learning_rate])#, continue_from_previous])

            next_buffer_idx = halftone_img_timestamp % 2

            # Try acquiring one of the locks from the halftone_locks list
            lock = halftone_locks[next_buffer_idx]
            if lock.acquire(blocking=False):
                img, _ = half_worker.ptc_q.get()

                # print(f'Halftone image has shape {modified_image.shape} and type {modified_image.dtype}')

                halftone_buffer[next_buffer_idx] = np.array( Image.fromarray( img[:,:,0,0] ).convert('RGBA') ).astype(np.float32)

                halftone_img_timestamp+=1
                halftone_timestamps[next_buffer_idx] = halftone_img_timestamp

                print(f'Halftone complete with timestamp {halftone_img_timestamp} and index {next_buffer_idx}')

                halftone_locks[next_buffer_idx].release()
                half_worker.lock.release()

        t = threading.Thread(target=wait_for_halftone)
        t.start()

        return t
    
    return None


def do_halftone_continuous():
    global update_continuous
    global half_worker

    while update_continuous:
        print('Halftoning')

        t = do_halftone()

        if t is not None:
            t.join()
            print('Halftone complete')

        
def update_texture(tag, image):
    if image is not None:
        dpg.set_value(tag, image.flatten())

def toggle_continuous(sender, app_data):
    global update_continuous
    print(f'Update continuous is {app_data}')
    update_continuous = app_data

def save_image():
    global modified_image
    if modified_image is None:
        return
    save_path = dpg.get_value("save_path_input")
    if save_path:
        iio.imwrite(save_path, modified_image)

dpg.create_context()

with dpg.file_dialog(directory_selector=False, show=False, callback=load_image, id="file_dialog", height=400):
    dpg.add_file_extension("Images {.png,.jpg,.jpeg}", color=(0, 255, 125, 255))


dpg.create_viewport(title='Image Editor', width=1200, height=1000)

with dpg.viewport_menu_bar():
    with dpg.menu(label="File"):
        dpg.add_menu_item(label="Open", callback= lambda: dpg.show_item("file_dialog"))

if __name__ == "__main__":
    dpg.setup_dearpygui()
    dpg.show_viewport()

    dpg.add_window(tag='Halftoner', width=1200, height=1000)
    dpg.set_primary_window('Halftoner', True)

    # below replaces, start_dearpygui()
    while dpg.is_dearpygui_running():
        # insert here any code you would like to run in the render loop
        # you can manually stop by using stop_dearpygui()
        # print("this will run every frame")
        dpg.render_dearpygui_frame()

        if init_complete:
            # Check whether any two of the timestamps are in the future
            max_timestamp = np.argmax(halftone_timestamps)

            if halftone_timestamps[max_timestamp] > displayed_image_timestamp:
                # Try acquiring the lock for the image with the max timestamp
                if halftone_locks[max_timestamp].acquire(blocking=False):
                    # Update the displayed image
                    print(f'Updating image with timestamp {halftone_timestamps[max_timestamp]} from index {max_timestamp}')
                    update_texture("modified_texture", halftone_buffer[max_timestamp])
                    displayed_image_timestamp = halftone_timestamps[max_timestamp]

                    if old_image is not None:
                        print(f'Computing diff texture')
                        # print(f'Old image has shape {old_image.shape} and type {old_image.dtype}')
                        diff_image = np.abs(halftone_buffer[max_timestamp] - old_image)
                        diff_image[:,:,3] = 255
                        # print(f'Diff image has shape {diff_image.shape} and type {diff_image.dtype}')
                        update_texture("diff_texture", diff_image)
                    
                    if old_comparison_counter >= old_comparison_max:
                        old_image = halftone_buffer[max_timestamp].copy()
                        update_texture("old_texture", old_image)
                        old_comparison_counter = 0

                    halftone_locks[max_timestamp].release()
                    old_comparison_counter+=cont_iterations
                    print(f'Old comparison counter is {old_comparison_counter}')


    dpg.destroy_context()

