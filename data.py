import numpy as np
import os
import time
import yaml
from math import pi, sqrt
from torch.utils.data import Dataset
import torch
import multiprocessing as mp
import shutil
import pickle
import itertools
import random
import string
from scipy.stats import truncnorm

from visualize import plot_trajectory, plot_energies
from util import create_folder, recreate_folder, pbc_diff

def get_accelerations(m, x, physical_const=2, box_size=6, prediction_mask=None, softening=False, softening_radius=0.5, charge=None):
    # Only predict acceleration for some particles (active in current time step)
    if prediction_mask is None:
        prediction_mask = np.ones((x.shape[0]), dtype=bool)
    a = np.zeros((x.shape[0], 2))
    for i in range (x.shape[0]):
        if prediction_mask[i]:
            vector_from_i_to_attractor = x - x[i]
            vector_from_i_to_attractor[vector_from_i_to_attractor > box_size/2] = vector_from_i_to_attractor[vector_from_i_to_attractor > box_size/2] - box_size # Periodic boundry conditions
            vector_from_i_to_attractor[vector_from_i_to_attractor <= -box_size/2] = vector_from_i_to_attractor[vector_from_i_to_attractor < -box_size/2] + box_size # Periodic boundry conditions
            vector_from_i_to_attractor[i] = np.zeros((2)) # make sure current particle is not attracted to itself
            dist = np.linalg.norm(vector_from_i_to_attractor, axis=1)
            dist[i] = 1 # avoid division by 0 error
            if softening:
                # Use plummer gravitational softening
                if charge is None:
                    # Acceleration for gravitational interaction
                    a[i] = np.sum((vector_from_i_to_attractor.T * np.divide((physical_const * m), np.power(dist**2 + softening_radius**2, 3/2))).T, axis=0) # a = Gm/sqrt(dist^2 + softening_radius^2)^(3/2) * pos_dif_vector
                else:
                    # Acceleration for Coulomb interaction
                    a[i] = - np.sum((vector_from_i_to_attractor.T * np.divide((physical_const * charge[i] * charge), np.power(dist**2 + softening_radius**2, 3/2))).T, axis=0) / m[i]
            else:
                if charge is None:
                    # Acceleration for gravitational interaction
                    a[i] = np.sum((vector_from_i_to_attractor.T * np.divide((physical_const * m), dist**3)).T, axis=0) # a = Gm/dist^3 * pos_dif_vector
                else:
                    # Acceleration for Coulomb interaction
                    a[i] = - np.sum((vector_from_i_to_attractor.T * np.divide((physical_const * charge[i] * charge), dist**3)).T, axis=0) / m[i]
    return a[prediction_mask] # Only return predicted accelerations

def get_accelerations_gpu(m, x, physical_const=2, box_size=6, prediction_mask=None, softening=False, softening_radius=0.5):
    # Only predict acceleration for some particles (active in current time step)
    if prediction_mask is None:
        prediction_mask = torch.ones((x.shape[0]), dtype=torch.bool)

    pred_count = torch.sum(prediction_mask)
    self_id_rows = torch.arange(pred_count)
    self_id_cols = torch.arange(x.shape[0])[prediction_mask]

    vector_from_i_to_attractor = x.unsqueeze(0).expand(pred_count, x.shape[0], 2) - x[prediction_mask].unsqueeze(1).expand(pred_count, x.shape[0], 2)
    vector_from_i_to_attractor[vector_from_i_to_attractor > box_size/2] -= box_size # Periodic boundry conditions
    vector_from_i_to_attractor[vector_from_i_to_attractor <= -box_size/2] += box_size # Periodic boundry conditions
    vector_from_i_to_attractor[self_id_rows, self_id_cols] = torch.zeros((2)) # make sure current particle is not attracted to itself

    dist = torch.norm(vector_from_i_to_attractor, dim=2)
    dist[self_id_rows, self_id_cols] = 1 # avoid division by 0 error

    m = m[prediction_mask][:, np.newaxis] 

    if softening:
        # Use plummer gravitational softening
        a = torch.sum((vector_from_i_to_attractor.transpose(2, 1) * torch.div((physical_const * m), (dist**2 + softening_radius**2)**(3/2))[:,np.newaxis,:]).transpose(2, 1), dim=1) # a = GM/sqrt(dist^2 + softening_radius^2)^(3/2) * pos_dif_vector
    else:
        a = torch.sum((vector_from_i_to_attractor.transpose(2, 1) * torch.div((physical_const * m), dist**3).unsqueeze(1)).transpose(2, 1), dim=1) # a = GM/dist^3 * pos_dif_vector
    
    return a # Only return predicted accelerations

def leapfrog(state, prediction_mask=None, dt=0.01, softening_radius=0.001, softening=False, physical_const=2, box_size=6, simulation_type='gravity', a_0=None):
    # Only update states for particles that are active in the current time step
    if prediction_mask is None:
        prediction_mask = np.ones((state.shape[0]), dtype=bool)
    if simulation_type == 'coulomb':
        m = state[:, 0]
        charge = state[:, 1]
        x_0 = state[:, 2:4]
        v_0 = state[:, 4:6]
    else:
        m = state[:, 0]
        x_0 = state[:, 1:3]
        v_0 = state[:, 3:5]
        charge = None

    # Kick Drift Kick
    if a_0 is None:
        a_0 = get_accelerations(m, x_0, physical_const=physical_const, box_size=box_size, prediction_mask=prediction_mask, softening=softening, softening_radius=softening_radius, charge=charge)
    x_1 = x_0 
    x_1[prediction_mask] += v_0[prediction_mask]*dt + (1/2)*a_0*dt**2
    x_1[x_1 >= box_size/2] -= box_size
    x_1[x_1 < -box_size/2] += box_size
    a_1 = get_accelerations(m, x_1, physical_const=physical_const, box_size=box_size, prediction_mask=prediction_mask, softening=softening, softening_radius=softening_radius, charge=charge)
    v_1 = v_0 
    v_1[prediction_mask] += (1/2)*(a_0 + a_1)*dt
    if charge is not None:
        return np.concatenate((m[:, np.newaxis], charge[:, np.newaxis], x_1, v_1), axis=1)
    else:
        return np.concatenate((m[:, np.newaxis], x_1, v_1), axis=1)

def leapfrog_gpu(state, prediction_mask=None, dt=0.01, softening_radius=0.001, softening=False, physical_const=2, box_size=6, a_0=None):
    # Only update states for particles that are acctive in the current time step
    if prediction_mask is None:
        prediction_mask = torch.ones((state.shape[0]), dtype=torch.bool)
    m = state[:, 0]
    x_0 = state[:, 1:3]
    v_0 = state[:, 3:5]

    # Kick Drift Kick
    if a_0 is None:
        a_0 = get_accelerations_gpu(m, x_0, physical_const=physical_const, box_size=box_size, prediction_mask=prediction_mask, softening=softening, softening_radius=softening_radius)
    x_1 = x_0 
    x_1[prediction_mask] += v_0[prediction_mask]*dt + (1/2)*a_0*dt**2
    x_1[x_1 >= box_size/2] -= box_size
    x_1[x_1 < -box_size/2] += box_size
    a_1 = get_accelerations_gpu(m, x_1, physical_const=physical_const, box_size=box_size, prediction_mask=prediction_mask, softening=softening, softening_radius=softening_radius)
    v_1 = v_0 
    v_1[prediction_mask] += (1/2)*(a_0 + a_1)*dt
    return torch.cat((m.unsqueeze(1), x_1, v_1), dim=1)

# hierarchical variable time steps. Time steps are split into levels where at level i time step is dt/2^i
# This makes time steps sinchronized at each top level dt  
def time_step(state, particle_mask=None, dt_particle=None, dt_level=0.01, softening_radius=0.001, niu=0.01, is_topmost=True, softening=True, physical_const=2, box_size=6, simulation_type='gravity', accelerations=None): # for no plummer_softening:  softening_radius=0.2, niu=0.0005, with softening: softening_radius=0.0001, niu=0.1
    # Particles pushed down to this time step have mask=true
    if particle_mask is None:
        particle_mask = np.ones((state.shape[0]), dtype=bool)

    if simulation_type == 'coulomb':
        m = state[:, 0]
        charge = state[:, 1]
        x = state[:, 2:4]
    else:
        m = state[:, 0]
        x = state[:, 1:3]
        charge =  None

    # If particle step sizes were not provided by the function up in the recursion - calculate them
    if accelerations is None:
        accelerations = get_accelerations(m, x, physical_const=physical_const, box_size=box_size, softening=softening, softening_radius=softening_radius, charge=charge)
    if dt_particle is None:
        a_magnitude = np.linalg.norm(accelerations, axis=1)
        dt_particle = niu * np.sqrt(np.divide(softening_radius, a_magnitude))

    # Push particles that should have smaller step size down one level
    small_dt_mask = (dt_particle < dt_level)
    small_exist = np.any(small_dt_mask)
    if small_exist:
        # Get updated states from the stepsizes below
        small_state = time_step(state.copy(), particle_mask=small_dt_mask, dt_particle=dt_particle, dt_level=dt_level/2, softening_radius=softening_radius, niu=niu, is_topmost=False,
                            softening=softening, physical_const=physical_const, box_size=box_size, simulation_type=simulation_type, accelerations=accelerations)

    # Calculate one step for particles left at the current level
    current_level_mask = np.logical_and(~small_dt_mask,particle_mask)
    if np.any(current_level_mask):
        state = leapfrog(state, prediction_mask=current_level_mask, dt=dt_level, softening_radius=softening_radius, softening=softening, physical_const=physical_const, box_size=box_size, simulation_type=simulation_type, a_0=accelerations[current_level_mask])
    # Merge
    if small_exist:
        state[small_dt_mask] = small_state[small_dt_mask]

    # If the function is not top most in the recursion do the second step
    if not is_topmost:

        if simulation_type == 'coulomb':
            m = state[:, 0]
            charge = state[:, 1]
            x = state[:, 2:4]
        else:
            m = state[:, 0]
            x = state[:, 1:3]
            charge =  None

        # Calculate new step sizes for particles at this level
        accelerations[particle_mask] = get_accelerations(m, x, physical_const=physical_const, box_size=box_size, prediction_mask=particle_mask, softening=softening, softening_radius=softening_radius, charge=charge)
        a_magnitude = np.linalg.norm(accelerations[particle_mask], axis=1)
        dt_particle[particle_mask] = niu * np.sqrt(np.divide(softening_radius, a_magnitude))
        
        # Push particles that should have smaller step size down one level
        small_dt_mask = (dt_particle < dt_level)
        small_exist = np.any(small_dt_mask)
        if small_exist:
            # Get updated states from the stepsizes below
            small_state = time_step(state.copy(), particle_mask=small_dt_mask, dt_particle=dt_particle, dt_level=dt_level/2, softening_radius=softening_radius, niu=niu, is_topmost=False,
                                softening=softening, physical_const=physical_const, box_size=box_size, simulation_type=simulation_type, accelerations=accelerations)

        # Calculate one step for particles left at the current level
        current_level_mask = np.logical_and(~small_dt_mask,particle_mask)
        if np.any(current_level_mask):
            state = leapfrog(state, prediction_mask=current_level_mask, dt=dt_level, softening_radius=softening_radius, softening=softening, physical_const=physical_const, box_size=box_size, simulation_type=simulation_type, a_0=accelerations[current_level_mask])
        # Merge
        if small_exist:
            state[small_dt_mask] = small_state[small_dt_mask]

    # Return predicted steps one level up
    return state

def time_step_gpu(state, particle_mask=None, dt_particle=None, dt_level=0.01, softening_radius=0.001, niu=0.01, is_topmost=True, softening=True, physical_const=2, box_size=6, accelerations=None): # for no plummer_softening:  softening_radius=0.2, niu=0.0005, with softening: softening_radius=0.0001, niu=0.1
    # Particles pushed down to this time step have mask=true
    if particle_mask is None:
        particle_mask = torch.ones((state.shape[0]), dtype=torch.bool)
    m = state[:, 0]
    x = state[:, 1:3]

    # If particle step sizes were not provided by the function up in the recursion - calculate them
    if accelerations is None:
        accelerations = get_accelerations_gpu(m, x, physical_const=physical_const, box_size=box_size, softening=softening, softening_radius=softening_radius)
    if dt_particle is None:
        a_magnitude = torch.norm(accelerations, dim=1)
        dt_particle = niu * torch.sqrt(torch.div(softening_radius, a_magnitude))

    # Push particles that should have smaller step size down one level
    small_dt_mask = (dt_particle < dt_level)
    small_exist = torch.any(small_dt_mask)
    if small_exist:
        # Get updated states from the stepsizes below
        state_small = time_step_gpu(state, particle_mask=small_dt_mask, dt_particle=dt_particle, dt_level=dt_level/2, softening_radius=softening_radius, niu=niu, is_topmost=False,
                            softening=softening, physical_const=physical_const, box_size=box_size, accelerations=accelerations)

    # Calculate one step for particles left at the current level
    current_level_mask = (~small_dt_mask & particle_mask)
    state = leapfrog_gpu(state, prediction_mask=current_level_mask, dt=dt_level, softening_radius=softening_radius, softening=softening, physical_const=physical_const, box_size=box_size, a_0=accelerations[current_level_mask])
    # Merge
    if small_exist:
        state[small_dt_mask] = state_small[small_dt_mask]

    # If the function is not top most in the recursion do the second step
    if not is_topmost:
        m = state[:, 0]
        x = state[:, 1:3]

        # Calculate new step sizes for particles at this level
        accelerations[particle_mask] = get_accelerations_gpu(m, x, physical_const=physical_const, box_size=box_size, prediction_mask=particle_mask, softening=softening, softening_radius=softening_radius)
        a_magnitude = torch.norm(accelerations[particle_mask], dim=1)
        dt_particle[particle_mask] = niu * torch.sqrt(torch.div(softening_radius, a_magnitude))
        
        # Push particles that should have smaller step size down one level
        small_dt_mask = (dt_particle < dt_level)
        small_exist = torch.any(small_dt_mask)
        if small_exist:
            # Get updated states from the stepsizes below
            state_small = time_step_gpu(state, particle_mask=small_dt_mask, dt_particle=dt_particle, dt_level=dt_level/2, softening_radius=softening_radius, niu=niu, is_topmost=False,
                                softening=softening, physical_const=physical_const, box_size=box_size, accelerations=accelerations)

        # Calculate one step for particles left at the current level
        current_level_mask = (~small_dt_mask & particle_mask)
        state = leapfrog_gpu(state, prediction_mask=current_level_mask, dt=dt_level, softening_radius=softening_radius, softening=softening, physical_const=physical_const, box_size=box_size, a_0=accelerations[current_level_mask])
        # Merge
        if small_exist:
            state[small_dt_mask] = state_small[small_dt_mask]

    # Return predicted steps one level up
    return state

def simulate(filename="simulated_trajectory.npy", timesteps=200, dt=0.01, softening_radius=0.001, niu=0.01, n_particles=3, physical_const=2, box_size=6, display=False, save=True, softening=True,
            simulation_type='gravity', initialization="random", fixed_initial_state=None):
    
    if simulation_type == 'gravity':
        # Empty trajectory table
        trajectory = np.zeros((timesteps+1, n_particles, 5)) # [time, particles, [mass, x, y, v_x, v_y]]
        # Initial state
        initial_state = np.zeros((trajectory.shape[1], 5))
        # All masses are = 1
        initial_state[:, 0] = 1 
    
        if initialization == 'random':
            # Initialize x,y position uniformly at random inside the box
            initial_state[:, 1:3] = np.random.RandomState().uniform(-box_size / 2, box_size / 2, size=(initial_state.shape[0], 2))
            # Initialize velocities v_x, v_y uniformly at random over (-1,1)
            initial_state[:, 3:5] = np.random.RandomState().uniform(-1, 1, size=(initial_state.shape[0], 2))
        elif initialization == 'fixed':
            initial_state = fixed_initial_state.copy()
        elif initialization == 'plummer':
            # Use standard (dimensionless) nbody units/starting conditions (total mass = 1, physical_const = 1, R = 3*pi/16, E ~ -0.25)
            R = 3*pi/16
            rfrac = (box_size/2) * 1/R
            # Get radius distribution by picking mass randomly and finding appropriate radius from m^-1(r), ensure max radius is not larger than box_size/2
            mass_cutoff = (rfrac ** 3)/pow(1.0 + rfrac ** 2, 1.5)
            radius =  1.0 /  np.sqrt( np.power(np.random.RandomState().uniform(0,mass_cutoff,(n_particles,1)), -2.0/3.0) - 1.0)
            # Each particle gets perturbed circular velocity
            velocity = np.sqrt( np.power(radius,2) * np.power((np.power(radius,2)+1.0), -3.0/2.0)) * (1 + np.random.RandomState().normal(0,0.01,(n_particles,1)))
            # Take random angle and transform to cartesian coordinates
            phi = np.random.RandomState().uniform(0.0,6,(n_particles,1))
            x = radius * np.cos(phi) * R
            y = radius * np.sin(phi) * R
            # Velocity vector is perpendicular to radius vector and randomly clock or counter clockwise (+ random noise)
            sign = (2*np.random.RandomState().randint(0,2,size=(n_particles,1))-1)
            v_phi = phi + sign * pi/2 + pi * np.random.RandomState().normal(0,0.1,(n_particles,1))
            vx = velocity * np.cos(v_phi) / sqrt(R)
            vy = velocity * np.sin(v_phi) / sqrt(R)
            
            initial_state[:, 1:3] = np.concatenate([x,y], axis=1)
            initial_state[:, 3:5] = np.concatenate([vx,vy], axis=1)
            # Params for truncated standard distribbution
            clip_left = 1e-6
            clip_right = 2
            mean = 1
            std = 0.5
            a, b = (clip_left - mean) / std, (clip_right - mean) / std
            # Perturb masses (draw from the truncated normal distribution)
            initial_state[:, 0] = truncnorm.rvs(a, b, loc=mean, scale=std, size=n_particles)
            # Normalize masses so that total mass is 1
            initial_state[:, 0] = initial_state[:, 0] / np.sum(initial_state[:, 0])
    elif simulation_type=='coulomb':
        # Empty trajectory table
        trajectory = np.zeros((timesteps+1, n_particles, 6)) # [time, particles, [mass, charge, x, y, v_x, v_y]]
        # Initial state
        initial_state = np.zeros((trajectory.shape[1], 6))
        # All masses are = 1
        initial_state[:, 0] = 1 

        if initialization == 'random':
            # Initialize charge uniformly at random over (0.5, 1.5) and assign random sign
            sign = (2*np.random.RandomState().randint(0,2,size=(n_particles))-1)
            initial_state[:, 1] = np.multiply(sign, np.random.RandomState().uniform(0.5, 1.5, size=(initial_state.shape[0])))
            # Initialize x,y position uniformly at random inside the box
            initial_state[:, 2:4] = np.random.RandomState().uniform(-box_size / 2, box_size / 2, size=(initial_state.shape[0], 2))
            # Initialize velocities v_x, v_y uniformly at random over (-1,1)
            initial_state[:, 4:6] = np.random.RandomState().uniform(-1, 1, size=(initial_state.shape[0], 2))


    trajectory[0] = initial_state
    
    for t in range(1,timesteps+1):
        # trajectory[t] = leapfrog(trajectory[t-1].copy(), dt=dt, softening_radius=softening_radius, softening=softening, physical_const=physical_const, box_size=box_size)
        start_time = time.time()
        trajectory[t] = time_step(trajectory[t-1].copy(), dt_level=dt, softening_radius=softening_radius, niu=niu, softening=softening, physical_const=physical_const, box_size=box_size, simulation_type=simulation_type)
        end_time = time.time()
        # print("Trajectory time step %i created (%f s)" % (t, end_time - start_time))
    
    if save:
        np.save(filename, trajectory)
    
    if display:
        plot_trajectory(trajectory, box_size=box_size, filename=None)
        plot_energies(trajectory, box_size=box_size, physical_const=physical_const, softening=softening, softening_radius=softening_radius)   

def simulate_gpu(filename="simulated_trajectory.npy", timesteps=200, dt=0.01, softening_radius=0.001, niu=0.01, n_particles=3, physical_const=2, box_size=6, display=False, save=True, softening=True,
            initialization="random", fixed_initial_state=None):
    torch.set_default_dtype(torch.float64)
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    # Empty trajectory table
    trajectory = np.zeros((timesteps, n_particles, 5)) # [time, particles, [mass, x, y, v_x, v_y]]

    # Initial state
    initial_state = torch.zeros((trajectory.shape[1], 5))
    # All masses are = 1
    initial_state[:, 0] = 1 

    if initialization == 'random':
        # Initialize x,y position uniformly at random inside the box
        initial_state[:, 1:3] = (-box_size) * torch.rand(initial_state.shape[0], 2) + (box_size / 2)
        # Initialize velocities v_x, v_y uniformly at random over (-1,1)
        initial_state[:, 3:5] = (-2) * torch.rand(initial_state.shape[0], 2) + 1
    elif initialization == 'fixed':
        initial_state = torch.from_numpy(fixed_initial_state.copy())

    trajectory[0] = initial_state.cpu().numpy()
    for t in range(1,timesteps+1):
        initial_state = time_step_gpu(initial_state, dt_level=dt, softening_radius=softening_radius, niu=niu, softening=softening, physical_const=physical_const, box_size=box_size)
        trajectory[t] = initial_state.cpu().numpy()

    if save:
        np.save(filename, trajectory)
    
    if display:
        plot_trajectory(trajectory, box_size=box_size, filename=None)
        plot_energies(trajectory, box_size=box_size, physical_const=physical_const, softening=False)
        if softening:
            plot_energies(trajectory, box_size=box_size, physical_const=physical_const, softening=softening, softening_radius=softening_radius)

def build_dataset(folder="dataset", train_size=1000, validation_size=200, test_size=200, timesteps=200, dt=0.01, softening_radius=0.001, niu=0.01, n_particles=3, physical_const=2, box_size=6, softening=True,
                simulation_type='gravity', initialization=None, as_partial=False, gpu=False, n_workers=1, data_dir="data"):
    if as_partial:
        # Put trajectories in a random subfolder of dataset partial folder
        folder_path = os.path.join(data_dir,folder+"_partials",''.join(random.choices(string.ascii_uppercase + string.digits, k=10)))
    else:
        # Put trajectories in a subfolder in data folder
        folder_path = os.path.join(data_dir,folder)

    # Remove old subfolder if it exists
    recreate_folder(folder_path)

    if simulation_type == 'coulomb' and gpu:
        raise Exception("Coulomb simulation only implimented for CPU")

    # if plummer initialization is used set physical_const = 1 because we use standard (dimentionless) n-body units
    if initialization == 'plummer' and simulation_type == 'gravity':
        physical_const = 1

    # Save params in yaml file:
    params_filename = os.path.join(folder_path, "params.yaml")
    with open(params_filename, 'w') as yaml_file:
        params_dict = {'folder': folder, 'train_size': train_size, 'validation_size': validation_size, 'test_size': test_size, 'box_size': box_size, 'physical_const': physical_const, 'timesteps': timesteps,
                        'n_particles': n_particles, 'dt':dt, 'softening_radius': softening_radius, 'niu':niu, 'softening': softening, 'simulation_type': simulation_type, 'initialization': initialization}
        yaml.dump(params_dict, yaml_file)

    for current_set in [('train', train_size), ('validation',validation_size), ('test', test_size)]:
        current_set_folder_path = os.path.join(folder_path,current_set[0])
        recreate_folder(current_set_folder_path)

        # If n_workers is -1 use all cpus in the system
        if n_workers == -1:
            n_workers = mp.cpu_count()
            
        # Use multiprocessing if more than 1 worker is requested
        if n_workers > 1:
            print("Using %s threads" % n_workers)
            start_time = time.time()
            pool = mp.Pool(processes=n_workers)
            for i in range(current_set[1]):
                if gpu:
                     pool.apply_async(simulate_gpu, tuple(), dict(filename=os.path.join(current_set_folder_path,"simulated_trajectory_{i}.npy".format(i=i)),
                                                                            timesteps=timesteps, n_particles=n_particles, physical_const=physical_const, box_size=box_size, display=False, save=True, dt=dt,
                                                                            softening_radius=softening_radius, niu=niu, softening=softening, initialization=initialization))
                else:
                    pool.apply_async(simulate, tuple(), dict(filename=os.path.join(current_set_folder_path,"simulated_trajectory_{i}.npy".format(i=i)),
                                                                            timesteps=timesteps, n_particles=n_particles, physical_const=physical_const, box_size=box_size, display=False, save=True, dt=dt,
                                                                            softening_radius=softening_radius, niu=niu, softening=softening, simulation_type=simulation_type, initialization=initialization))
            pool.close()
            pool.join()
            end_time = time.time()
            print("Trajectories created for %s (%f s)" % (current_set[0], end_time - start_time))
        else:
            # Single thread alternative with time tracking and progress logging
            for i in range(current_set[1]):
                    filename = os.path.join(current_set_folder_path,"simulated_trajectory_{i}.npy".format(i=i))
                    if gpu:
                        start_time = time.time()
                        simulate_gpu(filename=filename, timesteps=timesteps, n_particles=n_particles, physical_const=physical_const, box_size=box_size, display=False, save=True, dt=dt,
                                softening_radius=softening_radius, niu=niu, softening=softening, initialization=initialization)
                        end_time = time.time()
                    else:
                        start_time = time.time()
                        simulate(filename=filename, timesteps=timesteps, n_particles=n_particles, physical_const=physical_const, box_size=box_size, display=False, save=True, dt=dt,
                                softening_radius=softening_radius, niu=niu, softening=softening, simulation_type=simulation_type, initialization=initialization)
                        end_time = time.time()
                    print("Trajectory %i out of %i created for %s (%f s)" % (i+1, current_set[1], current_set[0], end_time - start_time))

    print("Dataset created")


def merge_partials_to_dataset(partial_folder="dataset_partials", dataset_folder="dataset", train_size=1000, validation_size=200, test_size=200, data_dir="data"):
    # Put trajectories in a subfolder in data folder
    dataset_folder_path = os.path.join(data_dir, dataset_folder)

    # Remove old subfolder if it exists
    recreate_folder(dataset_folder_path)
    # Create folders for each split
    create_folder(os.path.join(dataset_folder_path, "train"))
    create_folder(os.path.join(dataset_folder_path, "validation"))
    create_folder(os.path.join(dataset_folder_path, "test"))

    # Get partial files from partial subfolder in data folder
    partial_folder_path = os.path.join(data_dir, partial_folder)

    params_coppied = False
    
    # Output trajectories from partial datasets to the output dataset
    i = 0
    for root, dirs, files in os.walk(partial_folder_path):
     for file in files:
        file_path = os.path.join(root, file)
        if os.path.splitext(file)[1] == '.npy':
            # Copy trajectory files to an appropriate folder
            if i < train_size:
                split = 'train'
                idx = i
            elif i < train_size + validation_size:
                split = 'validation' 
                idx = i - train_size
            elif i < train_size + validation_size + test_size:
                split = 'test'
                idx = i - train_size - validation_size
            else:
                break
            shutil.copy(file_path, os.path.join(dataset_folder_path, split, f"simulated_trajectory_{idx}.npy"))
            i += 1
        elif file == 'params.yaml' and not params_coppied:
            # Copy first found param file and change split size values accordingly
            with open(file_path) as yaml_file:
                params = yaml.load(yaml_file, Loader=yaml.FullLoader)
                params['train_size'] = train_size
                params['validation_size'] = validation_size
                params['test_size'] = test_size
                with open(os.path.join(dataset_folder_path, 'params.yaml'), 'w') as output_yaml_file:
                    yaml.dump(params, output_yaml_file)
                    params_coppied = True

def build_nearest_neighbour_graph(filename, trajectory_path, box_size, neighbour_count):
    trajectory = np.load(trajectory_path)
    trajectory_len = trajectory.shape[0]
    n_particles = trajectory.shape[1]
    n_edges = n_particles*neighbour_count
    # Store graph as an edge list (sender, reciever)
    graph = np.zeros([trajectory_len, n_edges, 2])
    
    # Get closest neighbours for each node
    indices = list(range(1, n_particles))
    # Iterate over recievers
    for i in range(n_particles):

        distances = np.linalg.norm(pbc_diff(trajectory[:, indices, -4:-2], trajectory[:, i, -4:-2][:, np.newaxis, :], box_size=box_size), axis=-1)

        if i < n_particles-1:
            indices[i] -= 1
        
        closest_neighbours = np.transpose(np.stack([np.argpartition(distances, neighbour_count-1, axis=-1)[:, :neighbour_count], np.full((trajectory_len, neighbour_count), i)], axis=1), axes=(0,2,1))
        # Fix the sender id (we dropped self connection on reciever) for senders that have true id > reciever id (as currently their id is lower by 1)
        closest_neighbours[:, :, 0][closest_neighbours[:, :, 0] >= closest_neighbours[:, :, 1]] += 1

        graph[:, i*neighbour_count:(i+1)*neighbour_count, :] = closest_neighbours

    # Save graph
    np.save(filename, graph)

def generate_nearest_neighbour_graphs(dataset="dataset", neighbour_count=15, n_workers=1, data_dir="data", only_test=False):
    dataset_folder = os.path.join(data_dir, dataset)

    # Get relevant dataset params
    with open(os.path.join(dataset_folder, "params.yaml")) as yaml_file:
            params = yaml.load(yaml_file, Loader=yaml.FullLoader)
            box_size = float(params["box_size"])
            n_particles = int(params["n_particles"])
            train_size = int(params["train_size"])
            validation_size = int(params["validation_size"])
            test_size = int(params["test_size"])
            trajectory_len = int(params["timesteps"])
    
    # Set sets for which the graphs are generated
    if only_test:
        sets = [('test', test_size)]
    else:
        sets = [('train', train_size), ('validation',validation_size), ('test', test_size)]
    
    # For each trajectory build a nearest neighbour graph
    for current_set in sets:
        current_set_folder_path = os.path.join(dataset_folder, current_set[0])
        # Make a folder to store nearest neighbour graphs
        current_set_graph_folder_path = os.path.join(current_set_folder_path, 'graphs', f'{neighbour_count}_nn')
        recreate_folder(current_set_graph_folder_path)

        # If n_workers is <1 (i.e. -1) use all cpus in the system
        if n_workers < 1:
            n_workers = mp.cpu_count()
        
        # Use multiprocessing:
        print("Using %s threads" % n_workers)
        if n_workers == 1:
            for i in range(current_set[1]):
                trajectory_path = os.path.join(current_set_folder_path, f"simulated_trajectory_{i}.npy")
                build_nearest_neighbour_graph(filename=os.path.join(current_set_graph_folder_path, f"trajectory_{i}_graphs.npy"), trajectory_path=trajectory_path,
                                                box_size=box_size, neighbour_count=neighbour_count)
        else:
            pool = mp.Pool(processes=n_workers)
            for i in range(current_set[1]):
                trajectory_path = os.path.join(current_set_folder_path, f"simulated_trajectory_{i}.npy")
                pool.apply_async(build_nearest_neighbour_graph, tuple(), dict(filename=os.path.join(current_set_graph_folder_path, f"trajectory_{i}_graphs.npy"), trajectory_path=trajectory_path,
                                                                                box_size=box_size, neighbour_count=neighbour_count))
            pool.close()
            pool.join()
        
        print("Graphs created for %s set" % current_set[0])
    print("Finished")

def get_cells(ids, positions, levels_remaining, ref_point=[0,0], box_size=6):
    cells = []
    x_pos = positions[:, 0] >= ref_point[0]
    x_neg = positions[:, 0] < ref_point[0]
    y_pos = positions[:, 1] >= ref_point[1]
    y_neg = positions[:, 1] < ref_point[1]
    if levels_remaining > 1:
        n_cells = 4**levels_remaining
        n_rows = 2**levels_remaining # rows of cells in current level
        half_row_len = n_rows // 2 # number of cells in half of a row in current level (full row lenght in level-1)
        new_box_size = box_size/2
        ref_point_step = new_box_size/2
        # 1st cell
        cell_mask = np.logical_and(x_neg, y_pos)
        new_ref_point = [ref_point[0] - ref_point_step, ref_point[1] + ref_point_step]
        cells_1st = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)
        # 2nd cell
        cell_mask = np.logical_and(x_pos, y_pos)
        new_ref_point = [ref_point[0] + ref_point_step,  ref_point[1] + ref_point_step]
        cells_2nd = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)
        # 3rd cell
        cell_mask = np.logical_and(x_neg, y_neg)
        new_ref_point = [ref_point[0] - ref_point_step,  ref_point[1] - ref_point_step]
        cells_3th = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)
        # 4th cell
        cell_mask = np.logical_and(x_pos, y_neg)
        new_ref_point = [ref_point[0] + ref_point_step,  ref_point[1] - ref_point_step]
        cells_4th = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)
        # Reorder cells to be in row major order
        cells += list(itertools.chain(*zip(*[cells_1st[i::half_row_len] for i in range(half_row_len)], *[cells_2nd[i::half_row_len] for i in range(half_row_len)])))
        cells += list(itertools.chain(*zip(*[cells_3th[i::half_row_len] for i in range(half_row_len)], *[cells_4th[i::half_row_len] for i in range(half_row_len)])))
    else:
        # 1st cell
        cell_mask = np.logical_and(x_neg, y_pos)
        cells.append(ids[cell_mask])
        # 2nd cell
        cell_mask = np.logical_and(x_pos, y_pos)
        cells.append(ids[cell_mask])
        # 3rd cell
        cell_mask = np.logical_and(x_neg, y_neg)
        cells.append(ids[cell_mask])
        # 4th cell
        cell_mask = np.logical_and(x_pos, y_neg)
        cells.append(ids[cell_mask])
    return cells

def build_hierarchical_graph(graph_filename, assignment_filename, super_vertex_filename, super_vertex_id_filename, super_graph_filename, trajectory_path, box_size, levels):

    if levels < 2:
        raise ValueError('Must have at least 2 levels')

    trajectory = np.load(trajectory_path)

    trajectory_len = trajectory.shape[0]
    n_particles = trajectory.shape[1]
    n_cells = 4**levels
    row_len = 2**levels

    # Trajectory graphs and super vertices
    # Store graph as an edge list (sender, reciever) for each trajectory step
    trajectory_graphs = []
    trajectory_cell_assignments = []
    trajectory_super_vertices = []
    trajectory_edges_from_other_super_nodes = []
    trajectory_super_vertex_edges = []
    trajectory_super_vertex_ids = []
    if levels > 2:
        n_edges_per_super_vertex = 9*4 - 9
    else:
        n_edges_per_super_vertex = 16 - 9

    def vert_shift(cell_id, shift, row_len=row_len, n_cells=n_cells):
        cell_id = shift*row_len + cell_id
        if cell_id >= n_cells:
            return cell_id - n_cells
        elif cell_id < 0:
            return cell_id + n_cells
        else:
            return cell_id

    def horiz_shift(cell_id, shift, row_len=row_len):
        row_id = cell_id // row_len
        row_prefix = row_len * row_id
        cell_id = shift + cell_id - row_prefix
        if cell_id >= row_len:
            return row_prefix + cell_id - row_len
        elif cell_id < 0:
            return row_prefix + cell_id + row_len
        else:
            return row_prefix + cell_id

    def get_neighboring_ids(idx, row_len=row_len, n_cells=n_cells):
        if n_cells == 4:
            neighbor_ids = [0,1,2,3]
        else:
            neighbor_ids = [vert_shift(horiz_shift(idx,-1,row_len=row_len),-1,row_len=row_len,n_cells=n_cells),  vert_shift(idx,-1,row_len=row_len,n_cells=n_cells), vert_shift(horiz_shift(idx,1,row_len=row_len),-1,row_len=row_len,n_cells=n_cells), 
                            horiz_shift(idx,-1,row_len=row_len),                                                 idx,                                                horiz_shift(idx,1,row_len=row_len),
                            vert_shift(horiz_shift(idx,-1,row_len=row_len),1,row_len=row_len,n_cells=n_cells),   vert_shift(idx,1,row_len=row_len,n_cells=n_cells),  vert_shift(horiz_shift(idx,1,row_len=row_len),1,row_len=row_len,n_cells=n_cells)]
        return neighbor_ids

    def build_clusters(row_len, cluster_row_len):
        return np.tile(np.tile(np.arange(row_len).reshape(cluster_row_len, 2), [1,2]) + np.array([0,0,row_len,row_len]), [cluster_row_len,1]) + np.repeat(np.arange(cluster_row_len), cluster_row_len)[:,np.newaxis] * 2 * row_len

    # cell clusters - higher lever super nodes, each has 4 cell super nodes in it
    n_cell_clusters = n_cells//4
    cluster_row_len = row_len//2
    cell_clusters = build_clusters(row_len, cluster_row_len)

    # Iterate over trajectory
    for t in range(trajectory_len):

        graph = []
        cell_assignments = []
        super_vertices = []

        super_vertex_edges = np.zeros((n_cells*n_edges_per_super_vertex, 2), dtype=np.uint16)

        # Get list of indices to pass to cell function
        indices = np.arange(n_particles, dtype=np.uint16)

        # Split particles into cells for trajectory step (quadtree)
        cells = get_cells(indices, trajectory[t, :, -4:-2], levels, ref_point=[0,0], box_size=box_size)

        # Track cells that have no particles to remove edges from/to them
        non_empty_cells = []

        # Iterate over cells
        for q, cell in enumerate(cells):
            # 9 cell box around and including q
            extended_ids = get_neighboring_ids(q, row_len=row_len, n_cells=n_cells)
            extended_cell = np.concatenate([cells[i] for i in extended_ids], axis=None)
            # Edges between vertices belonging to the same cell and from vertices from nearby cells to vertices in current cell (reciever: cell vertices, sender: extended_cell vertices)
            edges_within = np.array(np.meshgrid(cell,extended_cell), dtype=np.uint16).T.reshape(-1,2)[:,(1,0)]
            # Drop self connections
            edges_within = edges_within[edges_within[:,0] != edges_within[:,1]]
            graph.append(edges_within)

            # Cluster id this cell belongs to
            cluster_id = cluster_row_len * ((q // row_len) // 2) + (q % row_len) // 2
            neighboring_cluster_ids = get_neighboring_ids(cluster_id, row_len=cluster_row_len, n_cells=n_cell_clusters)
            cells_in_neighboring_clusters = cell_clusters[neighboring_cluster_ids].reshape(-1)

            # Edges between current super vertex and other super vertices in neighboring clusters but not in extended_ids
            other_cell_ids = cells_in_neighboring_clusters[~np.isin(cells_in_neighboring_clusters, extended_ids)]
            super_vertex_edges[q*n_edges_per_super_vertex:(q+1)*n_edges_per_super_vertex, :] = np.stack([other_cell_ids, np.repeat(q, n_edges_per_super_vertex)]).T

            # Edges from vertices in this cell to cell super vertex - super verices have ids starting at 0
            edges_to_super_node = np.stack([np.repeat(q, len(cell)), cell]).T.astype(np.uint16, copy=False)
            cell_assignments.append(edges_to_super_node)

            # Compute super vertex params [total mass, center of mass (x,y), center of mass velocity (x,y)] - 5 params total
            if len(cell) > 0:
                particles_in_cell = trajectory[t][cell]
                super_vertices.append(np.concatenate([np.sum(particles_in_cell[:,:-4], axis=0), np.average(particles_in_cell[:,-4:], axis=0, weights=particles_in_cell[:,0])], axis=-1))
                non_empty_cells.append(q)

        cell_assignments = np.concatenate(cell_assignments)
        super_vertices = np.stack(super_vertices, axis=0)

        # Generate new ids for non epty cells
        non_empty_cells = np.array(non_empty_cells, dtype=np.uint16)
        new_cell_ids = np.arange(non_empty_cells.shape[0], dtype=np.uint16)

        # Remove edges that belong to empty super vertices
        super_vertex_edges = super_vertex_edges[np.all(np.isin(super_vertex_edges, non_empty_cells), axis=1)]
 
        # Re-index all the non empty cells with new ids
        cell_assignments[:,0] = new_cell_ids[np.digitize(cell_assignments[:,0].ravel(), non_empty_cells, right=True)].reshape(cell_assignments.shape[0])
        super_vertex_edges = new_cell_ids[np.digitize(super_vertex_edges.ravel(), non_empty_cells, right=True)].reshape(super_vertex_edges.shape)

        # Sort assignments w.r.t. vertex ids to use in scatter and gather operations
        cell_assignments = cell_assignments[cell_assignments[:,1].argsort()]

        assignments = [cell_assignments]
        super_vertices = [super_vertices]
        super_vertex_edges = [super_vertex_edges]
        super_vertex_ids = [non_empty_cells]

        # Build higher level super graphs
        for level in reversed(range(2, levels)):
            
            n_higher_level_clusters = 4**(level-1)
            n_current_level_clusters = 4**level
            higher_level_row_len = 2**(level-1)
            current_level_row_len = 2**level
            lower_level_row_len = 2**(level+1)
            lower_level_super_vertices = super_vertices[-1]
            lower_level_super_vertex_ids = super_vertex_ids[-1]

                                    
            higher_level_clusters = build_clusters(current_level_row_len, higher_level_row_len)
            clusters = build_clusters(lower_level_row_len, current_level_row_len)
            assingments_to_current_level_super_vertices = []
            current_level_super_vertex_features = []
            current_level_super_vertex_edges = []

            non_empty_clusters = []

            for c, cluster in enumerate(clusters):
                # Get all non empty cells from lower level that belong to current cluster
                cluster = np.arange(len(lower_level_super_vertex_ids))[np.isin(lower_level_super_vertex_ids, cluster)]

                if len(cluster) > 0:
                    neighbour_ids = get_neighboring_ids(c, row_len=current_level_row_len, n_cells=n_current_level_clusters)
                    
                    # Higher level cluster id this cluster belongs to
                    if n_higher_level_clusters == 4:
                        cells_in_neighboring_clusters = higher_level_clusters.reshape(-1)
                    else:
                        parent_cluster_id = higher_level_row_len * ((c // current_level_row_len) // 2) + (c % current_level_row_len) // 2
                        neighboring_cluster_ids = get_neighboring_ids(parent_cluster_id, row_len=higher_level_row_len, n_cells=n_higher_level_clusters)
                        cells_in_neighboring_clusters = higher_level_clusters[neighboring_cluster_ids].reshape(-1)

                    # Edges between current super vertex and other super vertices in neighboring clusters but not in extended_ids
                    cells_in_neighboring_clusters = cells_in_neighboring_clusters[~np.isin(cells_in_neighboring_clusters, neighbour_ids)]
                    current_level_super_vertex_edges.append(np.stack([cells_in_neighboring_clusters, np.repeat(c, len(cells_in_neighboring_clusters))]).T)

                    assingments_to_current_level_super_vertices.append(np.stack([np.repeat(c, len(cluster)), cluster]).T.astype(np.uint16, copy=False))

                    # Compute super vertex params [total mass, center of mass (x,y), center of mass velocity (x,y)] - 5 params total
                    cells_in_cluster = lower_level_super_vertices[cluster]
                    current_level_super_vertex_features.append(np.concatenate([np.sum(cells_in_cluster[:,:-4], axis=0), np.average(cells_in_cluster[:,-4:], axis=0, weights=cells_in_cluster[:,0])], axis=-1))
                    non_empty_clusters.append(c)

                
            assingments_to_current_level_super_vertices = np.concatenate(assingments_to_current_level_super_vertices)
            current_level_super_vertex_features = np.stack(current_level_super_vertex_features)
            current_level_super_vertex_edges = np.concatenate(current_level_super_vertex_edges)

            # Re-index non-empty higher level super nodes
            non_empty_clusters = np.array(non_empty_clusters, dtype=np.uint16)
            new_current_level_super_vertex_ids = np.arange(non_empty_clusters.shape[0], dtype=np.uint16)

            # Remove edges that belong to empty clusters
            current_level_super_vertex_edges = current_level_super_vertex_edges[np.all(np.isin(current_level_super_vertex_edges, non_empty_clusters), axis=1)]
    
            # Re-index all the non empty clusters with new ids
            assingments_to_current_level_super_vertices[:,0] = new_current_level_super_vertex_ids[np.digitize(assingments_to_current_level_super_vertices[:,0].ravel(), non_empty_clusters, right=True)].reshape(assingments_to_current_level_super_vertices.shape[0])
            current_level_super_vertex_edges = new_current_level_super_vertex_ids[np.digitize(current_level_super_vertex_edges.ravel(), non_empty_clusters, right=True)].reshape(current_level_super_vertex_edges.shape)

            assingments_to_current_level_super_vertices = assingments_to_current_level_super_vertices[assingments_to_current_level_super_vertices[:,1].argsort()]

            assignments.append(assingments_to_current_level_super_vertices)
            super_vertices.append(current_level_super_vertex_features)
            super_vertex_edges.append(current_level_super_vertex_edges)
            super_vertex_ids.append(non_empty_clusters)

        trajectory_graphs.append(np.concatenate(graph))
        trajectory_cell_assignments.append(assignments)
        trajectory_super_vertices.append(super_vertices)
        trajectory_super_vertex_edges.append(super_vertex_edges)
        trajectory_super_vertex_ids.append(super_vertex_ids)

            
    # Save graph, assignments to super vertices, super vertex features, super vertex graph
    # Graph is saved as pickle of a list that holds numpy array of edges for each batch, because same number of edges cannot be guaranteed.
    with open(graph_filename, 'wb') as pickle_file:
        pickle.dump(trajectory_graphs, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(assignment_filename, 'wb') as pickle_file:
        pickle.dump(trajectory_cell_assignments, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(super_vertex_filename, 'wb') as pickle_file:
        pickle.dump(trajectory_super_vertices, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(super_vertex_id_filename, 'wb') as pickle_file:
        pickle.dump(trajectory_super_vertex_ids, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)  
    with open(super_graph_filename, 'wb') as pickle_file:
        pickle.dump(trajectory_super_vertex_edges, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

def generate_hierarchical_graphs(dataset="dataset", levels=3, n_workers=1, data_dir="data", only_test=False):
    dataset_folder = os.path.join(data_dir, dataset)

    if levels < 2:
        raise ValueError('Must have at least 2 levels')

    # Get relevant dataset params
    with open(os.path.join(dataset_folder, "params.yaml")) as yaml_file:
            params = yaml.load(yaml_file, Loader=yaml.FullLoader)
            box_size = float(params["box_size"])
            n_particles = int(params["n_particles"])
            train_size = int(params["train_size"])
            validation_size = int(params["validation_size"])
            test_size = int(params["test_size"])
            trajectory_len = int(params["timesteps"])

    # Set sets for which the graphs are generated
    if only_test:
        sets = [('test', test_size)]
    else:
        sets = [('train', train_size), ('validation',validation_size), ('test', test_size)]
    
    # For each trajectory build a hierarchical graph
    for current_set in sets:
        current_set_folder_path = os.path.join(dataset_folder, current_set[0])
        # Make a folder to store hierarchical graphs
        current_set_graph_folder_path = os.path.join(current_set_folder_path, 'graphs', f'{levels}_level_hierarchical')
        recreate_folder(current_set_graph_folder_path)

        # If n_workers is <1 (i.e. -1) use all cpus in the system
        if n_workers < 1:
            n_workers = mp.cpu_count()
        
        # Use multiprocessing:
        print("Using %s threads" % n_workers)
        if n_workers == 1:
            for i in range(current_set[1]):
                trajectory_path = os.path.join(current_set_folder_path, f"simulated_trajectory_{i}.npy")
                build_hierarchical_graph(graph_filename=os.path.join(current_set_graph_folder_path, f"trajectory_{i}_graphs.pkl"),
                                                    assignment_filename=os.path.join(current_set_graph_folder_path, f"trajectory_{i}_assignments.pkl"),
                                                    super_vertex_filename=os.path.join(current_set_graph_folder_path, f"trajectory_{i}_super_vertices.pkl"),
                                                    super_vertex_id_filename=os.path.join(current_set_graph_folder_path, f"trajectory_{i}_super_vertex_ids.pkl"),
                                                    super_graph_filename=os.path.join(current_set_graph_folder_path, f"trajectory_{i}_super_graphs.pkl"),
                                                    trajectory_path=trajectory_path, box_size=box_size, levels=levels)
        else:
            pool = mp.Pool(processes=n_workers)
            for i in range(current_set[1]):
                trajectory_path = os.path.join(current_set_folder_path, f"simulated_trajectory_{i}.npy")
                pool.apply_async(build_hierarchical_graph, tuple(), dict(graph_filename=os.path.join(current_set_graph_folder_path, f"trajectory_{i}_graphs.pkl"),
                                                                                    assignment_filename=os.path.join(current_set_graph_folder_path, f"trajectory_{i}_assignments.pkl"),
                                                                                    super_vertex_filename=os.path.join(current_set_graph_folder_path, f"trajectory_{i}_super_vertices.pkl"),
                                                                                    super_vertex_id_filename=os.path.join(current_set_graph_folder_path, f"trajectory_{i}_super_vertex_ids.pkl"),
                                                                                    super_graph_filename=os.path.join(current_set_graph_folder_path, f"trajectory_{i}_super_graphs.pkl"),
                                                                                    trajectory_path=trajectory_path, box_size=box_size, levels=levels))
            pool.close()
            pool.join()

        print("Graphs created for %s set" % current_set[0])
    print("Finished")


class TrajectoryDataset(Dataset):
    def __init__(self, folder_path, split='train', rollout=False, graph_type=None, pre_load_graphs=True, target_step=1):
        self.folder_path = folder_path
        self.split = split
        self.split_folder = os.path.join(folder_path, split)

        # Load parameters from yaml file
        with open(os.path.join(self.folder_path, "params.yaml")) as yaml_file:
            params = yaml.load(yaml_file, Loader=yaml.FullLoader)
            self.box_size = float(params["box_size"])
            self.time_step = float(params["dt"])
            self.n_particles = int(params["n_particles"])
            self.trajectory_count = int(params[split+"_size"])
            self.trajectory_len = int(params["timesteps"])+1
            self.softening = bool(params["softening"])
            self.softening_radius = float(params["softening_radius"])
            self.simulation_type = params.get("simulation_type", "gravity")
            self.physical_const = float(params["physical_const"])

        # Set number of features each particle has
        if self.simulation_type == "coulomb":
            self.n_features = 6
        else:
            self.n_features = 5

        # Flag to pre load graph data (should be false for large graphs/datasets)
        self.pre_load_graphs = pre_load_graphs

        # Set how many steps into the future target will be. By default target is the next step
        self.target_step = target_step
        
        # Create a numpy array to store all trajectories (only if graphs are preloaded or if we have less than 10k particles and trajecotries fit into RAM)
        if self.pre_load_graphs or self.n_particles < 10000 or self.split == 'test':
            self.trajectories = torch.zeros((self.trajectory_count, self.trajectory_len, self.n_particles, self.n_features)) # each particle state is [m, x, y, v_x, v_y]
        else:
            self.trajectories = None

        # Store graphs associated with each trajectory if specified (each timestep in trajectory is expected to have a graph associated in this case)
        self.graph_type = str(graph_type)

        self.graph_folder = os.path.join(self.split_folder, 'graphs', self.graph_type)

        # If rollout data is requested return full trajectories, otherwise one t, t+1 pair is returned
        self.rollout = rollout

        # Count total number of training samples available
        if self.rollout:
            self.no_of_samples = self.trajectory_count
        else:
            self.no_of_samples = (self.trajectory_len - self.target_step) * self.trajectory_count

        if self.pre_load_graphs:
            if '_nn' in self.graph_type:
                self.n_neighbours = int(self.graph_type.split('_')[0])
                self.n_edges = self.n_particles * self.n_neighbours
                # Create a numpy array to store all the graphs for all of the trajecotries (edge list of (sender, reciever) for each state/timestep in the trajectory)
                self.graphs = np.zeros((self.trajectory_count, self.trajectory_len, self.n_edges, 2), dtype=np.uint16)
            elif '_level_hierarchical' in self.graph_type:
                self.levels = int(self.graph_type.split('_')[0])
                # Store graphs for each trajecotry in a python list, because number of edges can differ
                self.graphs = []
                # Store assignemts of vertices to cells in a list (pairs of super vertex - cell id and vertex id)
                self.assignments = []
                # Crete a list to store all the super vertex features  (total mass, center of mass position, center of mass velocity) - 5 params
                self.super_vertices = []
                # Create a list to store ids of super vertices in a full grid
                # self.super_vertex_ids = []
                # Create a list array to store all the super graphs for all of the trajecotries (edge list of (sender, reciever) for each state/timestep in the trajectory)
                self.super_graphs = []
            else:
                self.graph_type = ""

            # Populate the numpy array(s)
            for i in range(self.trajectory_count):

                trajectory_path = os.path.join(folder_path, split, f"simulated_trajectory_{i}.npy")
                trajectory = np.load(trajectory_path)
                self.trajectories[i] = torch.tensor(trajectory.astype(np.float32))

                if '_nn' in self.graph_type:
                    graph_path = os.path.join(self.graph_folder, f"trajectory_{i}_graphs.npy")
                    graph = np.load(graph_path)
                    self.graphs[i] = graph
                elif '_level_hierarchical' in self.graph_type:
                    graph_path=os.path.join(self.graph_folder, f"trajectory_{i}_graphs.pkl")
                    graph = pickle.load(open(graph_path, "rb"))
                    self.graphs.append(graph)
                    assignment_path = os.path.join(self.graph_folder, f"trajectory_{i}_assignments.pkl")
                    assignment = pickle.load(open(assignment_path, "rb"))
                    self.assignments.append(assignment)
                    super_vertex_path = os.path.join(self.graph_folder, f"trajectory_{i}_super_vertices.pkl")
                    super_vertex_features = pickle.load(open(super_vertex_path, "rb"))
                    self.super_vertices.append(super_vertex_features)
                    super_graph_path = os.path.join(self.graph_folder, f"trajectory_{i}_super_graphs.pkl")
                    super_graph = pickle.load(open(super_graph_path, "rb"))
                    self.super_graphs.append(super_graph)
        else:
            if '_nn' in self.graph_type:
                self.n_neighbours = int(self.graph_type.split('_')[0])
                self.n_edges = self.n_particles * self.n_neighbours
            elif '_level_hierarchical' in self.graph_type:
                self.levels = int(self.graph_type.split('_')[0])
            else:
                self.graph_type = ""

            # Pre-parse graphs into tensors for each time step in trajectory if not done already (not needed for for test/rollout)
            self.graph_tensor_folder = os.path.join(self.graph_folder, 'tensors')
            if not os.path.isdir(self.graph_tensor_folder) and not self.rollout:
                create_folder(self.graph_tensor_folder)
                for i in range(self.trajectory_count):
                    if '_nn' in self.graph_type:
                        graph_path = os.path.join(self.graph_folder, f"trajectory_{i}_graphs.npy")
                        graph = np.load(graph_path)
                        for t, tensor in enumerate(graph):
                            tensor_path = os.path.join(self.graph_tensor_folder, f"trajectory_{i}_graph_{t}.tpkl")
                            torch.save(torch.from_numpy(tensor.astype(np.int64)), tensor_path)
                    elif '_level_hierarchical' in self.graph_type:
                        graph_path=os.path.join(self.graph_folder, f"trajectory_{i}_graphs.pkl")
                        graph = pickle.load(open(graph_path, "rb"))
                        graph = [torch.from_numpy(tensor.astype(np.int64)) for tensor in graph]
                        assignment_path = os.path.join(self.graph_folder, f"trajectory_{i}_assignments.pkl")
                        assignment = pickle.load(open(assignment_path, "rb"))
                        assignment = [[torch.from_numpy(tensor.astype(np.int64)) for tensor in arr] for arr in assignment]
                        super_vertex_path = os.path.join(self.graph_folder, f"trajectory_{i}_super_vertices.pkl")
                        super_vertex_features = pickle.load(open(super_vertex_path, "rb"))
                        super_vertex_features = [[torch.from_numpy(tensor.astype(np.float32)) for tensor in arr] for arr in super_vertex_features]
                        super_graph_path = os.path.join(self.graph_folder, f"trajectory_{i}_super_graphs.pkl")
                        super_graph = pickle.load(open(super_graph_path, "rb"))
                        super_graph = [[torch.from_numpy(tensor.astype(np.int64)) for tensor in arr] for arr in super_graph]
                        for t, timestep_vals in enumerate(zip(graph, assignment, super_vertex_features, super_graph)):
                            tensor_path = os.path.join(self.graph_tensor_folder, f"trajectory_{i}_timestep_{t}.tpkl")
                            torch.save(timestep_vals, tensor_path)

            # Populate only the trajectory tensor if it exists/fits into memory
            if self.trajectories is not None:
                for i in range(self.trajectory_count):
                    trajectory_path = os.path.join(self.split_folder, f"simulated_trajectory_{i}.npy")
                    trajectory = np.load(trajectory_path)
                    self.trajectories[i] = torch.from_numpy(trajectory.astype(np.float32))
            else:
                # Pre-parse trajectories into tensors for each time step in trajectory if not done already (not needed for for test/rollout)
                self.trajectory_tensor_folder = os.path.join(self.split_folder, 'tensors')
                if not os.path.isdir(self.trajectory_tensor_folder) and not self.rollout:
                    create_folder(self.trajectory_tensor_folder)
                    for i in range(self.trajectory_count):
                        trajectory_path = os.path.join(self.split_folder, f"simulated_trajectory_{i}.npy")
                        trajectory = np.load(trajectory_path)
                        for t, tensor in enumerate(trajectory):
                            tensor_path = os.path.join(self.trajectory_tensor_folder, f"simulated_trajectory_{i}_{t}.tpkl")
                            torch.save(torch.from_numpy(tensor.astype(np.float32)), tensor_path)

    def __len__(self):
        return self.no_of_samples

    def __getitem__(self, idx):
        if self.rollout:
            # Dynamically load the trajectories if they are not in memory
            if self.trajectories is not None:
                trajectory = self.trajectories[idx,::self.target_step]
            else:
                trajectory = torch.from_numpy(np.load(os.path.join(self.split_folder, f"simulated_trajectory_{idx}.npy")).astype(np.float32))[::self.target_step]
            if '_nn' in self.graph_type:
                # Only get graph for the first time step
                if self.pre_load_graphs:
                    graph = self.graphs[idx, 0]
                else:
                    graph = np.load(os.path.join(self.graph_folder, f"trajectory_{idx}_graphs.npy"), mmap_mode='r')[0]
                # Return: inputs, targets, R_s, R_r
                return trajectory[:-1], trajectory[1:], torch.from_numpy(graph[:,0].astype(np.int64)), torch.from_numpy(graph[:,1].astype(np.int64))
            elif '_level_hierarchical' in self.graph_type:
                if self.pre_load_graphs:
                    graph = self.graphs[idx][0] 
                    assignment = self.assignments[idx][0] 
                    super_vertex_features = self.super_vertices[idx][0]  
                    super_graph = self.super_graphs[idx][0]  
                else:
                    graph = pickle.load(open(os.path.join(self.graph_folder, f"trajectory_{idx}_graphs.pkl"), "rb"))[0]
                    assignment = pickle.load(open(os.path.join(self.graph_folder, f"trajectory_{idx}_assignments.pkl"), "rb"))[0]
                    super_vertex_features = pickle.load(open(os.path.join(self.graph_folder, f"trajectory_{idx}_super_vertices.pkl"), "rb"))[0]
                    super_graph = pickle.load(open(os.path.join(self.graph_folder, f"trajectory_{idx}_super_graphs.pkl"), "rb"))[0]
                # Return: inputs, targets, R_s, R_r, assignment, V_super, super_vertex_ids, super_graph
                return trajectory[:-1], trajectory[1:], torch.from_numpy(graph[:,0].astype(np.int64)), torch.from_numpy(graph[:,1].astype(np.int64)), assignment, super_vertex_features, super_graph
            else:
                # Return: inputs, targets  
                return trajectory[:-1], trajectory[1:]
        else:
            # Get trajectory from idx
            trajectory_idx = idx // (self.trajectory_len - self.target_step)
            # Get particular time step from idx
            t_idx = idx % (self.trajectory_len - self.target_step)
            if self.trajectories is not None:
                trajectory = self.trajectories[trajectory_idx]
                trajectory_input, trajectory_target = trajectory[t_idx], trajectory[t_idx + self.target_step]
            else:
                trajectory_input = torch.load(os.path.join(self.trajectory_tensor_folder, f"simulated_trajectory_{trajectory_idx}_{t_idx}.tpkl"))
                trajectory_target = torch.load(os.path.join(self.trajectory_tensor_folder, f"simulated_trajectory_{trajectory_idx}_{t_idx + self.target_step}.tpkl"))
            if '_nn' in self.graph_type:
                if self.pre_load_graphs:
                    graph = torch.from_numpy(self.graphs[trajectory_idx, t_idx].astype(np.int64))
                else:
                    graph = torch.load(os.path.join(self.graph_tensor_folder, f"trajectory_{trajectory_idx}_graph_{t_idx}.tpkl"))
                # Return: inputs, targets, R_s, R_r
                return trajectory_input, trajectory_target, graph[:,0], graph[:,1]  # inputs, targets, R_s, R_r
            elif '_level_hierarchical' in self.graph_type:
                if self.pre_load_graphs:
                    graph = torch.from_numpy(self.graphs[trajectory_idx][t_idx].astype(np.int64))
                    assignment = self.assignments[trajectory_idx][t_idx]
                    super_vertex_features = self.super_vertices[trajectory_idx][t_idx]
                    # super_vertex_ids = self.super_vertex_ids[trajectory_idx][t_idx]
                    super_graph = self.super_graphs[trajectory_idx][t_idx]
                else:
                    graph, assignment, super_vertex_features, super_graph = torch.load(os.path.join(self.graph_tensor_folder, f"trajectory_{trajectory_idx}_timestep_{t_idx}.tpkl"))
                
                # Return: inputs, targets, R_s, R_r, assignment, V_super, super_graph
                return trajectory_input, trajectory_target, graph[:,0], graph[:,1], assignment, super_vertex_features, super_graph
            else:
                # Return: inputs, targets  
                return trajectory_input, trajectory_target


if __name__ == "__main__":
    # Physical constant (ravitational and Coulomb) used for all datasets
    physical_const = 2
    
    # Simulate and display a random trajectory
    # simulate(filename="simulated_trajectories.npy", timesteps=200, n_particles=3, physical_const=physical_const, box_size=6, softening_radius=0.2, niu=0.001, display=True, save=False, softening=True, initialization='random')
    
    #### Build datasets
    ### Gravitational datasets
    # build_dataset(folder="3_particles_gravity", train_size=1000, validation_size=200, test_size=200, timesteps=200, dt=0.01, softening_radius=0.2, niu=0.001, n_particles=3,
    #                physical_const=physical_const, box_size=6, softening=True, simulation_type='gravity', initialization="random")

    # build_dataset(folder="10_particles_gravity", train_size=1000, validation_size=200, test_size=200, timesteps=200, dt=0.01, softening_radius=0.2, niu=0.001, n_particles=10,
    #                physical_const=physical_const, box_size=11.0, softening=True, simulation_type='gravity', initialization="random", n_workers=-1)

    build_dataset(folder="20_particles_gravity", train_size=1000, validation_size=200, test_size=200, timesteps=200, dt=0.01, softening_radius=0.2, niu=0.001, n_particles=20,
                   physical_const=physical_const, box_size=15.5, softening=True, simulation_type='gravity', initialization="random", n_workers=-1)

    # build_dataset(folder="50_particles_gravity", train_size=1000, validation_size=200, test_size=200, timesteps=200, dt=0.01, softening_radius=0.2, niu=0.001, n_particles=50,
    #                physical_const=physical_const, box_size=24.5, softening=True, simulation_type='gravity', initialization="random", n_workers=-1)

    # build_dataset(folder="100_particles_gravity", train_size=1000, validation_size=200, test_size=200, timesteps=200, dt=0.01, softening_radius=0.2, niu=0.001, n_particles=100,
    #                physical_const=physical_const, box_size=34.6, softening=True, simulation_type='gravity', initialization="random", n_workers=-1)

    # build_dataset(folder="1000_particles_gravity", train_size=1000, validation_size=200, test_size=200, timesteps=200, dt=0.01, softening_radius=0.2, niu=0.001, n_particles=1000,
    #                physical_const=physical_const, box_size=109.5, softening=True, simulation_type='gravity', initialization="random", as_partial=True, n_workers=-1)

    # build_dataset(folder="5000_particles_gravity", train_size=1000, validation_size=200, test_size=200, timesteps=200, dt=0.01, softening_radius=0.2, niu=0.001, n_particles=5000,
    #                 physical_const=physical_const, box_size=244.9, softening=True, simulation_type='gravity', initialization="random", as_partial=True, n_workers=8, gpu=True)

    # build_dataset(folder="10000_particles_gravity", train_size=1000, validation_size=200, test_size=200, timesteps=200, dt=0.01, softening_radius=0.2, niu=0.001, n_particles=10000,
    #                 physical_const=physical_const, box_size=346.4, softening=True, simulation_type='gravity', initialization="random", as_partial=True, n_workers=2, gpu=True)

    ## Merge partial datasets built for 1000+ particles
    # merge_partials_to_dataset(partial_folder='1000_particles_gravity_partials', dataset_folder='1000_particles_gravity', train_size=1000, validation_size=200, test_size=200)
    # merge_partials_to_dataset(partial_folder='5000_particles_gravity_partials', dataset_folder='5000_particles_gravity', train_size=1000, validation_size=200, test_size=200)
    # merge_partials_to_dataset(partial_folder='10000_particles_gravity_partials', dataset_folder='10000_particles_gravity', train_size=1000, validation_size=200, test_size=200)

    ## Build nearest neighbour graphs
    generate_nearest_neighbour_graphs(dataset="20_particles_gravity", neighbour_count=15, n_workers=-1)
    # generate_nearest_neighbour_graphs(dataset="50_particles_gravity", neighbour_count=15, n_workers=-1)
    # generate_nearest_neighbour_graphs(dataset="100_particles_gravity", neighbour_count=15, n_workers=-1)
    # generate_nearest_neighbour_graphs(dataset="1000_particles_gravity", neighbour_count=15, n_workers=-1)
    # generate_nearest_neighbour_graphs(dataset="5000_particles_gravity", neighbour_count=15, n_workers=-1)
    # generate_nearest_neighbour_graphs(dataset="10000_particles_gravity", neighbour_count=15, n_workers=-1)

    ## Build hierarchical cell graphs (min 2 levels)
    # generate_hierarchical_graphs(dataset="10_particles_gravity", levels=2, n_workers=-1)
    generate_hierarchical_graphs(dataset="20_particles_gravity", levels=2, n_workers=-1)
    # generate_hierarchical_graphs(dataset="100_particles_gravity", levels=3, n_workers=-1)
    # generate_hierarchical_graphs(dataset="1000_particles_gravity", levels=5, n_workers=-1)
    # generate_hierarchical_graphs(dataset="5000_particles_gravity", levels=6, n_workers=-1)
    # generate_hierarchical_graphs(dataset="10000_particles_gravity", levels=7, n_workers=-1)


    ### Coulomb datasets
    # build_dataset(folder="100_particles_coulomb", train_size=1000, validation_size=200, test_size=200, timesteps=200, dt=0.01, softening_radius=0.2, niu=0.001, n_particles=100,
    #                physical_const=physical_const, box_size=34.5, softening=True, simulation_type='coulomb', initialization="random", n_workers=-1)

    # build_dataset(folder="1000_particles_coulomb", train_size=1000, validation_size=200, test_size=200, timesteps=200, dt=0.01, softening_radius=0.2, niu=0.001, n_particles=1000,
    #                physical_const=physical_const, box_size=110.0, softening=True, simulation_type='coulomb', initialization="random", as_partial=True, n_workers=-1)

    # merge_partials_to_dataset(partial_folder='1000_particles_coulomb_partials', dataset_folder='1000_particles_coulomb', train_size=1000, validation_size=200, test_size=200)

    ## Build nearest neighbour graphs
    # generate_nearest_neighbour_graphs(dataset="100_particles_coulomb", neighbour_count=15, n_workers=-1)
    # generate_nearest_neighbour_graphs(dataset="1000_particles_coulomb", neighbour_count=15, n_workers=-1)

    ## Build hierarchical cell graphs (min 2 levels)
    # generate_hierarchical_graphs(dataset="100_particles_coulomb", levels=3, n_workers=-1)
    # generate_hierarchical_graphs(dataset="1000_particles_coulomb", levels=5, n_workers=-1)


    ### Graphs for 100 particle dataset with different level counts
    # generate_hierarchical_graphs(dataset="100_particles_gravity", levels=2, n_workers=-1)
    # generate_hierarchical_graphs(dataset="100_particles_gravity", levels=4, n_workers=-1)
    # generate_hierarchical_graphs(dataset="100_particles_gravity", levels=5, n_workers=-1)

