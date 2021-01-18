import numpy as np
import os
import yaml
import re
from collections import defaultdict
import json.decoder
import argparse
from zlib import crc32
import glob

import matplotlib.pyplot as plt

from util import kinetic_energy, potential_energy, hamiltonian, total_angular_momentum, total_linear_momentum, pbc_rms_error, pbc_mean_relative_energy_error

def plot_trajectory(states, box_size=6, small_box_size=None, filename=None, base_size=1.0):

    def fade(lenght, color):
        rgba = np.ones((lenght, 4))
        shade = np.linspace(0, 0.8, lenght)[:, np.newaxis]
        rgba[:, :] = 1*(1-shade) + np.array(color)[np.newaxis, :]*shade
        rgba[:, 3] = 0.1
        return rgba

    plt.figure(figsize=(10, 10))
    positions = states[:, :, -4:-2]
    masses = states[:, :, 0]
    
    cmap = plt.get_cmap("tab10")

    for n in range(positions.shape[1]):
        color = cmap(n/positions.shape[1])
        plt.scatter(positions[:, n, 0], positions[:, n, 1], s=50*masses[-1,n]*base_size, color=fade(positions.shape[0], color))
    plt.scatter(positions[-1, :, 0], positions[-1, :, 1], s=100*np.abs(states[-1, :, -5])*base_size,  c=cmap((states[-1, :, -5] < 0).astype(np.int)*4))
    if small_box_size is not None:
        plt.vlines(small_box_size/2,-small_box_size/2,small_box_size/2,  linestyles='dashed', linewidth=3)
        plt.vlines(-small_box_size/2,-small_box_size/2,small_box_size/2,  linestyles='dashed', linewidth=3)
        plt.hlines(small_box_size/2,-small_box_size/2,small_box_size/2,  linestyles='dashed', linewidth=3)
        plt.hlines(-small_box_size/2,-small_box_size/2,small_box_size/2,  linestyles='dashed', linewidth=3)
    plt.xlim(-box_size/2, box_size/2)
    plt.ylim(-box_size/2, box_size/2)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_energies(data, physical_const=2, box_size=6, softening=False, filename=None, softening_radius=0.001):
    kinetic = np.empty(len(data))
    potential = np.empty(len(data))
    mm = np.empty((len(data), 2))
    am = np.empty((len(data), 2))
    for t in range(len(data)):
        kinetic[t] = kinetic_energy(data[t])
        potential[t] = potential_energy(data[t], physical_const=physical_const, box_size=box_size, softening=softening, softening_radius=softening_radius)
        mm[t] = total_linear_momentum(data[t])
        am[t] = total_angular_momentum(data[t])
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot('121')
    ax1.plot(kinetic, label='Kinetic energy', linestyle='--', color='navy')
    if softening:
        ax1.plot(potential, label='Potential energy (with softening)', linestyle='-.', color='navy')
        ax1.plot(kinetic+potential, label='Total energy - Hamiltonian (with softening)', color='navy')
    else:
        ax1.plot(potential, label='Potential energy', linestyle='-.', color='navy')
        ax1.plot(kinetic+potential, label='Total energy - Hamiltonian', color='navy')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Energy of the system')
    ax1.legend()
    ax2 = fig.add_subplot('122')
    ax2.plot(np.linalg.norm(mm, axis=-1), label='Linear momentum', color='teal')
    ax2.plot(np.linalg.norm(am, axis=-1), label='Angular momentum', color='darkmagenta')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Total momentum')
    ax2.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_test_trajectories(dataset, model, trajectories=[0], data_dir='..\data', particle_size=100):

    with open(os.path.join(data_dir, dataset, "params.yaml")) as yaml_file:
            params = yaml.load(yaml_file, Loader=yaml.FullLoader)
            box_size = float(params["box_size"])
            time_step = float(params["dt"])
            n_particles = int(params["n_particles"])
            trajectory_count = int(params["test_size"])
            trajectory_len = int(params["timesteps"])
            softening = bool(params["softening"])
            softening_radius = float(params["softening_radius"])
            if "physical_const" in params:
                physical_const = float(params["physical_const"])
            else:
                physical_const = float(params["G"])

    def populate_energy_momentum_subplot(ax1, ax2, data, prefix, energy_color='navy', linear_momentum_color='teal', angular_momentum_color='darkmagenta'):
        kinetic = np.empty(len(data))
        potential = np.empty(len(data))
        mm = np.empty((len(data), 2))
        am = np.empty((len(data), 2))
        for t in range(len(data)):
            kinetic[t] = kinetic_energy(data[t])
            potential[t] = potential_energy(data[t], physical_const=physical_const, box_size=box_size, softening=softening, softening_radius=softening_radius)
            mm[t] = total_linear_momentum(data[t])
            am[t] = total_angular_momentum(data[t])
        
        ax1.plot(kinetic, label=prefix+' kinetic energy', linestyle='--', color=energy_color)
        if softening:
            ax1.plot(potential, label=prefix+' potential energy (with softening)', linestyle='-.', color=energy_color)
            ax1.plot(kinetic+potential, label=prefix+' hamiltonian (with softening)', color=energy_color)
        else:
            ax1.plot(potential, label='potential energy', linestyle='-.', color=energy_color)
            ax1.plot(kinetic+potential, label=prefix+' hamiltonian', color=energy_color)
        ax2.plot(np.linalg.norm(mm, axis=-1), label=prefix+' linear momentum', color=linear_momentum_color)
        ax2.plot(np.linalg.norm(am, axis=-1), label=prefix+' angular momentum', color=angular_momentum_color)

    def populate_trajectory_subplot(ax, data):
        positions = data[:, :, -4:-2]
        masses = data[:, :, 0]

        def fade(lenght, color):
            rgba = np.ones((lenght, 4))
            shade = np.linspace(0, 0.8, lenght)[:, np.newaxis]
            rgba[:, :] = 1*(1-shade) + np.array(color)[np.newaxis, :]*shade
            rgba[:, 3] = 0.1
            return rgba
    
        cmap = plt.get_cmap("tab10")

        ax.scatter(positions[-1, :, 0], positions[-1, :, 1], s=particle_size)
        for n in range(positions.shape[1]):
            color = cmap(n/positions.shape[1])
            ax.scatter(positions[:, n, 0], positions[:, n, 1], s=0.5*particle_size*masses[-1,n], color=fade(positions.shape[0], color))
        ax.scatter(positions[-1, :, 0], positions[-1, :, 1], s=particle_size*np.abs(data[-1, :, -5]), c=cmap((data[-1, :, -5] < 0).astype(np.int)*4))
        ax.axis(xmin=-box_size/2, xmax=box_size/2, ymin=-box_size/2, ymax=box_size/2)
        ax.set_xticks([])
        ax.set_yticks([])

    
    for i in trajectories:

        target_trajectory = np.load(os.path.join(data_dir, dataset, f"test/simulated_trajectory_{i}.npy"))
        predicted_trajectory = np.load(os.path.join(data_dir, dataset, f"test_predictions/{model}/predicted_trajectory_{i}.npy"))

        fig = plt.figure(figsize=(16, 16))

        plt.title(f'Predicted vs True trajectory. Test sample {i}. RMSE: {pbc_rms_error(predicted_trajectory[1:,:,1:], target_trajectory[1:,:,1:], box_size=box_size)}')
        plt.axis('off')

        ax1 = fig.add_subplot('221')
        ax2 = fig.add_subplot('222')

        populate_energy_momentum_subplot(ax1, ax2, predicted_trajectory, prefix='predicted', energy_color='red', linear_momentum_color='gold', angular_momentum_color='crimson')
        populate_energy_momentum_subplot(ax1, ax2, target_trajectory, prefix='true', energy_color='navy', linear_momentum_color='teal', angular_momentum_color='darkmagenta')

        ax1.legend()
        ax2.legend()

        ax3 = fig.add_subplot('223')
        populate_trajectory_subplot(ax3, predicted_trajectory)
        ax3.set_title('Predicted trajectory')
        ax4 = fig.add_subplot('224')
        populate_trajectory_subplot(ax4, target_trajectory)
        ax4.set_title('True trajectory')

        fig.show()

def plot_custom_data(data_dict, output_file=None, plot_dir='plots'):

    for plot_name, plot_dict in data_dict.items():

        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot()

        for model_name, model_data in plot_dict['data'].items():
            ax1.plot(model_data['x'], model_data['y'], model_data.get('line_type', '-o'), label=model_name, color=model_data['color'])
            if 'std' in model_data:
                ax1.fill_between(model_data['x'], np.array(model_data['y'])-np.array(model_data['std']), np.array(model_data['y'])+np.array(model_data['std']), alpha=0.2, color=model_data['color'])
            if 'annotations' in model_data:
                for annotation in model_data['annotations']:
                    ax1.annotate(annotation["text"], xy=annotation["pos"],  xycoords='data',
                    xytext=(0, 9), textcoords='offset points', horizontalalignment='center')

        ax1.legend()

        if 'log_x' in plot_dict:
            if plot_dict['log_x']:
                ax1.set_xscale('log')
        if 'log_y' in plot_dict:
            if plot_dict['log_y']:
                ax1.set_yscale('log')

        ax1.set_title(plot_name)
        ax1.set_xlabel(plot_dict['x_label'])
        ax1.set_ylabel(plot_dict['y_label'])
    
        if output_file is not None:
            if not any(output_file in output_file for file_format in ['.png', '.pdf']):
                output_file = output_file.split('.')[0]+'.png'
            plt.savefig(os.path.join(plot_dir, output_file), bbox_inches='tight')
        else:
            plt.show()

def build_test_error_data_dict(data, colors, rollout_steps, errors='position', data_dir='data', experiment_dir='', plot_dir='plots', output_file=None, log_x=False, log_y=False):
    # Dict to store values {model: {dataset_sizes: [], errors: [], color: ''}}
    error_dict = defaultdict(lambda: defaultdict(list))

    # Setup fallback color scheme
    cmap = plt.get_cmap("tab10")

    for dataset, models in data.items():
        # Load dataset params
        with open(os.path.join(data_dir, dataset, "params.yaml")) as yaml_file:
            params = yaml.load(yaml_file, Loader=yaml.FullLoader)
            box_size = float(params["box_size"])
            time_step = float(params["dt"])
            n_particles = int(params["n_particles"])
            trajectory_count = int(params["test_size"])
            trajectory_len = int(params["timesteps"])
            softening = bool(params["softening"])
            softening_radius = float(params["softening_radius"])
            if "physical_const" in params:
                physical_const = float(params["physical_const"])
            else:
                physical_const = float(params["G"])

        # Load true trajecotries
        target_trajectories = []
        for i in range(trajectory_count):
            target_trajectory = np.load(os.path.join(data_dir, dataset, f"test/simulated_trajectory_{i}.npy"))
            target_trajectories.append(target_trajectory)
        target_trajectories = np.stack(target_trajectories, axis=0)

        for model, model_path in models.items():
            rmse = []
            if len(model_path) > 0:
                model_paths = glob.iglob(os.path.join(data_dir, experiment_dir, dataset, f"test_predictions/{model_path}"))
                for path in model_paths:
                    # Load predicted trajectories
                    predicted_trajectories = []
                    for i in range(trajectory_count):
                        predicted_trajectory = np.load(os.path.join(path, f"predicted_trajectory_{i}.npy"))
                        predicted_trajectories.append(predicted_trajectory)

                    target_step_search = re.search(r'target_step_(\d+)', model_path)
                    if target_step_search:
                        target_step = int(target_step_search.group(1))
                    else:
                        target_step = 1
                    # Error over all trajectories
                    predicted_trajectories = np.stack(predicted_trajectories, axis=0) 
                    if errors == 'energy':
                        rmse.append(pbc_mean_relative_energy_error(predicted_trajectories[:,:rollout_steps+1], box_size=box_size, physical_const=physical_const, softening=softening, softening_radius=softening_radius))
                    elif errors == 'position':
                        rmse.append(pbc_rms_error(predicted_trajectories[:,1:rollout_steps+1,:,1:], target_trajectories[:,target_step:(rollout_steps+1)*target_step:target_step,:,1:], box_size=box_size))

                rmse = np.stack(rmse)
                error_dict[model]['x'].append(n_particles)
                error_dict[model]['y'].append(np.mean(rmse))
                error_dict[model]['std'].append(np.std(rmse))
                if model in colors:
                    error_dict[model]['color'] = colors[model]
                else:
                    error_dict[model]['color'] = cmap(float(crc32(model.encode("utf-8")) & 0xffffffff) / 2**32) 

    x_label = 'Number of particles'
    if errors == 'energy':
        y_label = f'Mean |ΔE/E| between last and true energy over all test trajectories'
        plot_title = f'Mean relative energy error between true energy and energy at the last time step (mean(|ΔE/E|)) ({rollout_steps} steps)'
    elif errors == 'position':
        y_label = f'RMSE over unrolled test trajecotry ({rollout_steps} steps)'
        plot_title = f'Position accuracy degradation when using more particles ({rollout_steps} step rollout)'

    data_dict = {plot_title: {'data': error_dict, 'x_label': x_label, 'y_label': y_label, 'log_x': log_x, 'log_y': log_y}}
    
    if output_file is not None:
        with open(os.path.join(plot_dir, output_file.split('.')[0]+'.json'), 'w') as json_file:
            json.dump(dict(data_dict), json_file)

    return data_dict

def plot_test_errors(data, colors, rollout_steps, errors='position', data_dir='..\data', experiment_dir='', plot_dir='..\plots', output_file=None, log_x=False, log_y=False):
    
    data_dict = build_test_error_data_dict(data, colors, rollout_steps, errors=errors, data_dir=data_dir, experiment_dir=experiment_dir, plot_dir=plot_dir, output_file=output_file, log_x=log_x, log_y=log_y)
    plot_custom_data(data_dict, output_file=output_file, plot_dir=plot_dir)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', action='store', default="data",
                        dest='data_dir',
                        help='Set data directory')
    parser.add_argument('--plot_dir', action='store', default="plots",
                        dest='plot_dir',
                        help='Set plot directory')
    parser.add_argument('--errors', action='store', default="position",
                        dest='errors',
                        help='Set error type to plot')
    parser.add_argument('--data', action='store', default="{}", type=json.loads,
                        dest='data',
                        help='Set json containing predictions to use for error ploting')
    parser.add_argument('--colors', action='store', default="{}", type=json.loads,
                        dest='colors',
                        help='Set colors to use for models in data dict (supply a json). If not supplied random colors are used')
    parser.add_argument('--rollout_steps', action='store', default=200, type=int,
                        dest='rollout_steps',
                        help='Set number of steps the trajectories are unrolled for')
    parser.add_argument('--experiment_dir', action='store', default="",
                        dest='experiment_dir',
                        help='Set experiment sub-directory')
    parser.add_argument('--output_file', action="store", default=None,
                        dest='output_file',
                        help='Set filename for plot png and yaml files')
    parser.add_argument('--log_x', action="store_true", default=False,
                        dest='log_x',
                        help='Use log scale for X axis')
    parser.add_argument('--log_y', action="store_true", default=False,
                        dest='log_y',
                        help='Use log scale for Y axis')
    arguments = parser.parse_args()

    # Plot test errors
    plot_test_errors(data=arguments.data, colors=arguments.colors, rollout_steps=arguments.rollout_steps, errors=arguments.errors, data_dir=arguments.data_dir,
                     experiment_dir=arguments.experiment_dir, plot_dir=arguments.plot_dir, output_file=arguments.output_file, log_x=arguments.log_x, log_y=arguments.log_y)
