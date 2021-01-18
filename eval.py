import torch
import numpy as np
import os
import shutil
import argparse
import re

from data import TrajectoryDataset
from model import DeltaGN, HierarchicalDeltaGN, HOGN, HierarchicalHOGN
from util import pbc_rms_error, pbc_mean_relative_energy_error, recreate_folder, create_folder, full_graph_senders_and_recievers, nn_graph_senders_and_recievers, hierarchical_graph_senders_and_recievers, collate_into_one_graph


def evaluate_model(model_file="", dataset="3_particles", model_dataset="", graph_type="", model_dir="models", data_dir="data", experiment_dir="", pre_load_graphs=True, start_id=0, end_id=-1):

    # Set evaluation dataset as model dataset (dataset model was trained on) if no model dataset was specified
    if len(model_dataset) < 1:
        model_dataset = dataset

    # Model path and output folder path
    if len(experiment_dir) > 0:
        model_path = os.path.join(model_dir, experiment_dir, model_dataset, model_file)
        output_folder_path = os.path.join(data_dir, experiment_dir, dataset, 'test_predictions', model_file)
    else:
        model_path = os.path.join(model_dir, model_dataset, model_file)
        output_folder_path = os.path.join(data_dir, dataset, 'test_predictions', model_file)

    # Get model type from filename
    model_type = model_file.split('_')[0]

    # Get graph type from the filename if not set
    if len(graph_type) < 1:
        graph_type_search = re.search(r'graph_(\d+(?:_nn|_level_hierarchical))_', model_file)
        if graph_type_search:
            graph_type = graph_type_search.group(1)
        else:
            # If graph type is not found set the graph to fully connected
            graph_type = 'fully_connected'
    
    # Extract graph type specific params
    if '_nn' in graph_type:
        edges_per_node = int(graph_type.split('_')[0])
    elif '_level_hierarchical' in graph_type:
        hierarchy_levels = int(graph_type.split('_')[0])

    # Get hidden unit count from the filename
    hidden_units_search = re.search(r'hidden_units_(\d+)_', model_file)
    if hidden_units_search:
        hidden_units = int(hidden_units_search.group(1))
    else:
        # Use defaults
        hidden_units = -1

    # Get target_step from the filename
    target_step_search = re.search(r'target_step_(\d+)_', model_file)
    if target_step_search:
        target_step = int(target_step_search.group(1))
    else:
        # Use defaults
        target_step = 1

    # Get integrator type from the filename
    if 'euler' in model_file or 'rk1' in model_file:
        integrator = 'euler' 
    else:
        integrator = 'rk4' 

    # Use batch_size=1 for inference
    batch_size = 1
    
    # Load test data set
    test_set = TrajectoryDataset(folder_path=os.path.join(data_dir, dataset), split='test', rollout=True, graph_type=graph_type, target_step=target_step, pre_load_graphs=pre_load_graphs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_into_one_graph)

    # Get parameters form dataset
    box_size = test_set.box_size
    time_step = test_set.time_step
    n_particles = test_set.n_particles
    physical_const = test_set.physical_const
    softening = test_set.softening
    softening_radius = test_set.softening_radius
    simulation_type = test_set.simulation_type

    if model_type == "DeltaGN":
        model = DeltaGN(box_size=box_size, edge_output_dim=hidden_units, node_output_dim=hidden_units, simulation_type=simulation_type)
    elif model_type == "HierarchicalDeltaGN":
        model = HierarchicalDeltaGN(box_size=box_size, edge_output_dim=hidden_units, node_output_dim=hidden_units, simulation_type=simulation_type)
    elif model_type == "HOGN":
        model = HOGN(box_size=box_size, edge_output_dim=hidden_units, node_output_dim=hidden_units, global_output_dim=hidden_units, integrator=integrator, simulation_type=simulation_type)
    elif model_type == "HierarchicalHOGN":
        model = HierarchicalHOGN(box_size=box_size, edge_output_dim=hidden_units, node_output_dim=hidden_units, integrator=integrator, simulation_type=simulation_type)

    model.load_state_dict(torch.load(model_path))

    # Remove old output subfolder if it exists in case all trajectories will be built
    if start_id == 0 and end_id == -1:
        recreate_folder(output_folder_path)
    else:
        create_folder(output_folder_path)

    # Set proper end_id
    if end_id == -1:
        end_id = len(test_set)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if graph_type == 'fully_connected':
        # Build graph - R_s one hot encoding of edge sender id, R_r: one hot encoding of reciever id
        # All possible n*(n-1) are modeled as present - matrix shape (n-1) x n
        # Repeated for each sample in the batch (final size: batch x n(n-1) x n)
        R_s, R_r = full_graph_senders_and_recievers(n_particles, batch_size=batch_size, device=device)

    # Build a tensor of step sizes for each sample in the batch
    dt = torch.Tensor([time_step]).to(device).unsqueeze(0).expand(batch_size, 1) * target_step

    # Log all trajectories for RMS over all trajectories
    predicted_trajectories = []

    print("Evaluationg model %s on %s test dataset" % (model_file, dataset))

    for i, data in enumerate(test_loader, 0):
        if start_id <= i < end_id:
            # Get the inputs as full trajectory
            if graph_type == 'fully_connected':
                inputs, targets = data
            elif '_level_hierarchical' in graph_type:
                inputs, targets, R_s, R_r, assignment, V_super, super_graph = data

                R_s = R_s.to(device, non_blocking=True)
                R_r = R_r.to(device, non_blocking=True)
                assignment = [el.to(device, non_blocking=True) for el in assignment]
                V_super = [el.to(device, non_blocking=True) for el in V_super]
                super_graph = [el.to(device, non_blocking=True) for el in super_graph]
            elif '_nn' in graph_type:
                inputs, targets, R_s, R_r = data
                R_s = R_s.to(device, non_blocking=True)
                R_r = R_r.to(device, non_blocking=True)
            else:
                raise ValueError('Graph type not recognized')

            # Log the predicted trajecotry
            output_trajectory = np.zeros((inputs.shape[1]+1, inputs.shape[2], inputs.shape[3]))
            output_trajectory[0] = inputs[0][0].numpy()

            # Forward pass 
            current_state = inputs[0][0].unsqueeze(0)
            current_state = current_state.to(device)
            for j in range(inputs.shape[1]):
                if '_level_hierarchical' in graph_type:
                    output = model(current_state, R_s, R_r, assignment, V_super, super_graph, dt)
                else:
                    output = model(current_state, R_s, R_r, dt)
                current_state = torch.cat([current_state[:,:,:-4], output], dim = 2).detach() # Detach to stop graph unroll in next loop iteration 
                if (not graph_type == 'fully_connected') and (j < inputs.shape[1]-1):
                    if '_nn' in graph_type:
                        R_s, R_r = nn_graph_senders_and_recievers(current_state, n_neighbours=edges_per_node, box_size=box_size, device=device)
                    elif '_level_hierarchical' in graph_type:
                        R_s, R_r, assignment, V_super, super_graph = hierarchical_graph_senders_and_recievers(current_state, levels=hierarchy_levels, box_size=box_size, device=device)
                    else:
                        raise Exception
                output_trajectory[j+1, :, :] = current_state.cpu().detach().numpy() # [timesteps, particles, state]; state = [m,x,y,v_x,v_y]

            
            print("RMSE for trajectory %i: %f" % (i , pbc_rms_error(output_trajectory[1:,:,1:], targets.cpu().numpy()[0][:,:,1:], box_size=box_size)))
        
            # Save the predicted trajectory
            output_filename = os.path.join(output_folder_path,"predicted_trajectory_{i}.npy".format(i=i))
            np.save(output_filename, output_trajectory)

            # Log for RMS over all trajectories
            predicted_trajectories.append(output_trajectory)


    # RMS over all trajectories
    predicted_trajectories = np.stack(predicted_trajectories, axis=0) # [trajectories, timesteps, particles, state]; state = [m,x,y,v_x,v_y]
    print("RMSE over all trajectories: %f" % (pbc_rms_error(predicted_trajectories[:,1:,:,1:], test_set.trajectories[start_id:end_id,target_step::target_step,:,1:].numpy(), box_size=box_size)))
    print("Mean relative energy error over all trajectories: %f" % (pbc_mean_relative_energy_error(predicted_trajectories, box_size=box_size, physical_const=physical_const, softening=softening, softening_radius=softening_radius)))

    print('Finished evaluation')

    # Return the output path
    return output_folder_path


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', action='store', default="models",
                        dest='model_dir',
                        help='Set model directory')
    parser.add_argument('--data_dir', action='store', default="data",
                        dest='data_dir',
                        help='Set data directory')
    parser.add_argument('--model_file', action='store', default="",
                        dest='model_file',
                        help='Set model parameter file to use for evaluation')
    parser.add_argument('--model_dataset', action='store', default="",
                        dest='model_dataset',
                        help='Set dataset model was trained on (by default --dataset value is used)')
    parser.add_argument('--dataset', action='store', default="3_particles",
                        dest='dataset',
                        help='Set dataset to use (if model_dataset is set this dataset is used for evaluation only)')
    parser.add_argument('--graph_type', action='store', default="",
                        dest='graph_type',
                        help='Set type of the graph to use')
    parser.add_argument('--experiment_dir', action='store', default="",
                        dest='experiment_dir',
                        help='Set experiment sub-directory')
    parser.add_argument('--dont_pre_load_graphs', action="store_false", default=True,
                        dest='pre_load_graphs',
                        help='Do not pre load graphs into memory (for the Dataset object). Use this flag if there is not enough RAM')
    parser.add_argument('--start_id', action='store', type=int, default=0,
                        dest='start_id',
                        help='Set start id of trajectory range to evaluate')
    parser.add_argument('--end_id', action='store', type=int, default=-1,
                        dest='end_id',
                        help='Set end id of trajectory range to evaluate')
    arguments = parser.parse_args()
    # Evaluate model using parsed arguments
    evaluate_model(model_file=arguments.model_file, dataset=arguments.dataset, model_dataset=arguments.model_dataset, graph_type=arguments.graph_type, model_dir=arguments.model_dir,
                    data_dir=arguments.data_dir, experiment_dir=arguments.experiment_dir, pre_load_graphs=arguments.pre_load_graphs, start_id=arguments.start_id, end_id=arguments.end_id)
