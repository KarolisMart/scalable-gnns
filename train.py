import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from math import inf
import time
import argparse
import random
import string

from data import TrajectoryDataset
from model import PBC_MSE_loss, DeltaGN, HierarchicalDeltaGN, HOGN, HierarchicalHOGN
from util import full_graph_senders_and_recievers, create_folder, collate_into_one_graph
from eval import evaluate_model

def training_step_static_graph(model, data, R_s, R_r, dt, device, accumulate_steps, box_size):
    # Get the inputs
    inputs, targets = data

    # Push them to the GPU
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    # Forward + backward + optimize
    start_time = time.perf_counter_ns()
    outputs = model(inputs, R_s, R_r, dt=dt)
    end_time = time.perf_counter_ns()
    loss = PBC_MSE_loss(outputs, targets[:,:,-4:], box_size=box_size)
    loss = loss / accumulate_steps
    loss.backward()
    
    return loss.item(), (end_time - start_time)

def validation_step_static_graph(model, test_data, R_s, R_r, dt, device, box_size):
    # Get validation data
    inputs, targets = test_data

    # Push to GPU
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    # Get outputs
    outputs = model(inputs, R_s, R_r, dt=dt)
    # Get loss
    test_loss = PBC_MSE_loss(outputs, targets[:,:,-4:], box_size=box_size).cpu().detach()

    return test_loss.item()

def training_step_dynamic_graph(model, data, dt, device, accumulate_steps, box_size, graph_type):

    if '_level_hierarchical' in graph_type:
        inputs, targets, R_s, R_r, assignment, V_super, super_graph = data

        # Push data to the GPU
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        R_s = R_s.to(device, non_blocking=True)
        R_r = R_r.to(device, non_blocking=True)
        assignment = [el.to(device, non_blocking=True) for el in assignment]
        V_super = [el.to(device, non_blocking=True) for el in V_super]
        super_graph = [el.to(device, non_blocking=True) for el in super_graph]

        # Forward pass (and time it)
        start_time = time.perf_counter_ns()
        outputs = model(inputs, R_s, R_r, assignment, V_super, super_graph, dt=dt)
        end_time = time.perf_counter_ns()
        loss = PBC_MSE_loss(outputs, targets[:,:,-4:], box_size=box_size)

        # Backward
        loss = loss / accumulate_steps
        loss.backward()
        return loss.item(), (end_time - start_time)

    else:
        # Get the inputs (inputs, targets, edge sender and reciever)
        inputs, targets, R_s, R_r = data

        R_s = R_s.to(device, non_blocking=True)
        R_r = R_r.to(device, non_blocking=True)

        # Push data to the GPU
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass (and time it)
        start_time = time.perf_counter_ns()
        outputs = model(inputs, R_s, R_r, dt=dt)
        end_time = time.perf_counter_ns()

        # Backward
        loss = PBC_MSE_loss(outputs, targets[:,:,-4:], box_size=box_size)
        loss = loss / accumulate_steps
        loss.backward()
    
        return loss.item(), (end_time - start_time)

def validation_step_dynamic_graph(model, test_data, dt, device, box_size, graph_type):

    if '_level_hierarchical' in graph_type:
        inputs, targets, R_s, R_r, assignment, V_super, super_graph = test_data

        # Push data to the GPU
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        R_s = R_s.to(device, non_blocking=True)
        R_r = R_r.to(device, non_blocking=True)
        assignment = [el.to(device, non_blocking=True) for el in assignment]
        V_super = [el.to(device, non_blocking=True) for el in V_super]
        super_graph = [el.to(device, non_blocking=True) for el in super_graph]

        outputs = model(inputs, R_s, R_r, assignment, V_super, super_graph, dt=dt)

        test_loss = PBC_MSE_loss(outputs, targets[:,:,-4:], box_size=box_size).cpu().detach()

        return test_loss.item()

    else:

        # Get the inputs (inputs, targets, edge sender and reciever)
        inputs, targets, R_s, R_r = test_data
        R_s = R_s.to(device, non_blocking=True)
        R_r = R_r.to(device, non_blocking=True)

        # Push to GPU
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Get outputs
        outputs = model(inputs, R_s, R_r, dt=dt)

        # Get loss
        test_loss = PBC_MSE_loss(outputs, targets[:,:,-4:], box_size=box_size).cpu().detach()

        return test_loss.item()

def train_model(model_type="DeltaGN", dataset="3_particles_gravity", learning_rate=1e-3, lr_decay=0.97725, batch_size=100, epochs=200, accumulate_steps=1, model_dir="models", data_dir="data",
                hidden_units=-1, validate=True, validate_epochs=1, graph_type='fully_connected', integrator='rk4',
                pre_load_graphs=True, data_loader_workers=2, smooth_lr_decay=False, target_step=1, cpu=False, experiment_dir="", log_dir="runs", resume_checkpoint="", save_after_time=0):
    # Track time for saving after x seconds
    start_time_for_save = time.monotonic()

    # Set CPU paralelization to maximum
    torch.set_num_threads(torch.get_num_threads())

    # Load training dataset
    train_set = TrajectoryDataset(folder_path=os.path.join(data_dir, dataset), split='train', graph_type=graph_type, pre_load_graphs=pre_load_graphs, target_step=target_step)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=data_loader_workers, collate_fn=collate_into_one_graph, pin_memory=True)

    # Load validation dataset
    validation_set = TrajectoryDataset(folder_path=os.path.join(data_dir, dataset), split='validation', graph_type=graph_type, pre_load_graphs=pre_load_graphs, target_step=target_step)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=data_loader_workers, collate_fn=collate_into_one_graph, pin_memory=True)

    # Get parameters form dataset
    box_size = train_set.box_size
    time_step = train_set.time_step
    n_particles = train_set.n_particles
    simulation_type = train_set.simulation_type

    # Flag models that use integrators
    integrator_model = False

    if model_type == "DeltaGN":
        model = DeltaGN(box_size=box_size, edge_output_dim=hidden_units, node_output_dim=hidden_units, simulation_type=simulation_type)
    elif model_type == "HierarchicalDeltaGN":
        model = HierarchicalDeltaGN(box_size=box_size, edge_output_dim=hidden_units, node_output_dim=hidden_units, simulation_type=simulation_type)
    elif model_type == "HOGN":
        model = HOGN(box_size=box_size, edge_output_dim=hidden_units, node_output_dim=hidden_units, global_output_dim=hidden_units, integrator=integrator, simulation_type=simulation_type)
        integrator_model = True
    elif model_type == "HierarchicalHOGN":
        model = HierarchicalHOGN(box_size=box_size, edge_output_dim=hidden_units, node_output_dim=hidden_units, integrator=integrator, simulation_type=simulation_type)
        integrator_model = True

    device = torch.device("cuda:0" if ((not cpu) and torch.cuda.is_available()) else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#3e-4
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay) # decay every 2 * 10^5 with lower imit of 10^-7

    # Track iterations and running loss for loging
    n_iter = 0
    running_loss = 0.0

    if graph_type == 'fully_connected':
        # Build graph - R_s one hot encoding of edge sender id, R_r: one hot encoding of reciever id
        # All possible n*(n-1) are modeled as present - matrix shape (n-1) x n
        # Repeated for each sample in the batch (final size: batch x n(n-1) x n)
        R_s, R_r = full_graph_senders_and_recievers(n_particles, batch_size=batch_size, device=device)

    # Build a tensor of step sizes for each sample in the batch
    dt = torch.Tensor([time_step]).to(device).unsqueeze(0) * target_step

    # Use current/start time to identify saved model and log dir
    start_time = time.strftime("%Y%m%d-%H%M%S")

    if len(resume_checkpoint) > 0:
        print('Resuming checkpoint')
        model_name = resume_checkpoint.replace('_checkpoint.tar', '')
        checkpoint = torch.load(os.path.join(model_dir, experiment_dir, dataset, resume_checkpoint))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        starting_epoch = checkpoint['epoch']
        n_iter = starting_epoch * len(train_loader)
    else:
        # Create model filename to use for saved model params and tensorboard log dir
        rand_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        model_name = f"{model_type}{f'_{integrator}' if integrator_model else ''}_lr_{learning_rate}_decay_{lr_decay}_epochs_{epochs}_batch_size_{batch_size}_accumulate_steps_{accumulate_steps}{f'_hidden_units_{hidden_units}' if hidden_units > 0 else ''}_graph_{graph_type}_target_step_{target_step}_{start_time}_{rand_string}"
        starting_epoch = 0

    # Setup direcotries for logs and models
    if len(experiment_dir) > 0:
        logdir=os.path.join(log_dir, experiment_dir, dataset, model_name)
        model_save_path = os.path.join(model_dir, experiment_dir, dataset)
    else:
        logdir=os.path.join(log_dir, dataset, model_name)
        model_save_path = os.path.join(model_dir, dataset)
    
    # Setup Writer for Tensorboard
    writer = SummaryWriter(log_dir=logdir)

    # Log lr for tensorboard
    writer.add_scalar('LearningRate/optimizer_lr', optimizer.param_groups[0]['lr'], n_iter)

    # Create output folder for the dataset models
    create_folder(model_save_path)

    # Track forward pass time
    forward_pass_times = []

    print("Training model %s on %s dataset" % (model_name, dataset))

    for epoch in range(starting_epoch, epochs):
        optimizer.zero_grad()
        for i, data in enumerate(train_loader, 0):
            # Do one training step and get loss value
            if graph_type == 'fully_connected':
                loss_value, forward_pass_time = training_step_static_graph(model, data, R_s, R_r, dt, device, accumulate_steps, box_size)
            else:
                loss_value, forward_pass_time = training_step_dynamic_graph(model, data, dt, device, accumulate_steps, box_size, graph_type)
            running_loss += loss_value
            forward_pass_times.append(forward_pass_time)

            # Do an optimizer step on accumulated gradients
            if (i+1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                n_iter += 1

                # Log training loss every 200 optimizer steps
                if n_iter % 200 == 0:
                    writer.add_scalar('Loss/train', running_loss / 200, n_iter)
                    running_loss = 0.0

            # Decay learning rate every 200k steps with lower imit of 10^-7 (if smooth decay is not set)
            if ((n_iter % (2 * 10**5) == 0) and not smooth_lr_decay) and (optimizer.param_groups[0]['lr'] > 10**(-7)):
                scheduler.step()
                # Log lr for tensorboard
                writer.add_scalar('LearningRate/optimizer_lr', optimizer.param_groups[0]['lr'], n_iter)

            # Save model after specified number of seconds (if set)
            if (save_after_time > 0):
                if ((time.monotonic() - start_time_for_save) > save_after_time):
                    torch.save(model.state_dict(), os.path.join(model_save_path, model_name))
                    start_time_for_save = time.monotonic()
        
        # Decay learning rate every epoch with lower imit of 10^-7 (if smooth decay is set)
        if (smooth_lr_decay) and (optimizer.param_groups[0]['lr'] > 10**(-7)):
            scheduler.step()
            # Log lr for tensorboard
            writer.add_scalar('LearningRate/optimizer_lr', optimizer.param_groups[0]['lr'], n_iter)

        # Save model each epoch
        torch.save(model.state_dict(), os.path.join(model_save_path, model_name))

        # Save a checkpoint each epoch
        torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()},os.path.join(model_save_path, model_name + "_checkpoint.tar"))
            
        # Evaluate on validation set every validate_epochs epoch. Always validate and save the model after the last epoch
        # No with torch.no_grad() since HOGN needs gradients
        if (validate and (epoch % validate_epochs == 0)) or (epoch == epochs - 1) :
            model.eval()
            running_test_loss = 0.0
            for p in model.parameters():
                p.require_grads = False
            for j, test_data in enumerate(validation_loader, 0):
                # Ensure no grad is left before validation step
                optimizer.zero_grad()
                if model_type == "NewMultiLevelHOGNDown5":
                    torch.cuda.empty_cache()
                # Do a validation step
                if graph_type == 'fully_connected':
                    loss_value =  validation_step_static_graph(model, test_data, R_s, R_r, dt, device, box_size)
                else:
                    loss_value = validation_step_dynamic_graph(model, test_data, dt, device, box_size, graph_type)
                running_test_loss += loss_value
            model.train()
            for p in model.parameters():
                p.require_grads = True

            # Log validation loss for tensorboard
            validation_loss = running_test_loss / len(validation_loader)
            writer.add_scalar('Loss/validation', validation_loss, n_iter)


    writer.close()

    # Print validation loss at the end of the training
    print(f'Finished Training. Final validation loss is {validation_loss}')
    forward_pass_times = (np.array(forward_pass_times) / 1000000) # in ms instead of ns
    print(f'Forward step took {np.mean(forward_pass_times)} ms on average (std: {np.std(forward_pass_times)})')
    # Return the filename of the model
    return model_name


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', action='store', default="models",
                        dest='model_dir',
                        help='Set model directory')
    parser.add_argument('--data_dir', action='store', default="data",
                        dest='data_dir',
                        help='Set data directory')
    parser.add_argument('--model', action='store', default="DeltaGN",
                        dest='model',
                        help='Set model type to train')
    parser.add_argument('--lr', action='store', type=float, default=0.0003,
                        dest='lr',
                        help='Set learning rate')
    parser.add_argument('--lr_decay', action='store', type=float, default=0.1, # use 0.97725 instead if decaying every epoch (smooth_lr_decay flag)
                        dest='lr_decay',
                        help='Set learning rate decay')
    parser.add_argument('--batch_size', action='store', type=int, default=100,
                        dest='batch_size',
                        help='Set batch size')
    parser.add_argument('--epochs', action='store', type=int, default=200,
                        dest='epochs',
                        help='Set number of epochs')
    parser.add_argument('--accumulate_steps', action='store', type=int, default=1,
                        dest='accumulate_steps',
                        help='Set number of epochs')
    parser.add_argument('--dataset', action='store', default="20_particles_gravity",
                        dest='dataset',
                        help='Set dataset to use')
    parser.add_argument('--hidden_units', action="store", type=int, default=-1,
                        dest='hidden_units',
                        help='Set number of hidden units linear layers of MLPs will use')
    parser.add_argument('--dont_validate', action="store_false", default=True,
                        dest='validate',
                        help='Do not validate model each epoch')
    parser.add_argument('--validate_every', action="store", type=int, default=1,
                        dest='validate_every',
                        help='Validate model every n epochs')
    parser.add_argument('--graph_type', action='store', default="fully_connected",
                        dest='graph_type',
                        help='Set type of the graaph to use')
    parser.add_argument('--integrator', action='store', default="rk4",
                        dest='integrator',
                        help='Set integrator to use for HOGN and OGN models')
    parser.add_argument('--dont_pre_load_graphs', action="store_false", default=True,
                        dest='pre_load_graphs',
                        help='Do not pre load graphs into memory (for the Dataset object). Use this flag if there is not enough RAM')
    parser.add_argument('--data_loader_workers', action="store", type=int, default=2,
                        dest='data_loader_workers',
                        help='Number of dataloader workers to use')
    parser.add_argument('--smooth_lr_decay', action="store_true", default=False,
                        dest='smooth_lr_decay',
                        help='Decay LR every epoch instead of every 200k training steps')
    parser.add_argument('--target_step', action="store", type=int, default=1,
                        dest='target_step',
                        help='How many steps into the future target will be')
    parser.add_argument('--cpu', action="store_true", default=False,
                        dest='cpu',
                        help='Train model on CPU (slow, not tested properly)')
    parser.add_argument('--experiment_dir', action='store', default="",
                        dest='experiment_dir',
                        help='Set experiment sub-directory')
    parser.add_argument('--log_dir', action='store', default="runs",
                        dest='log_dir',
                        help='Set directory for tensorboard logs')
    parser.add_argument('--resume_checkpoint', action='store', default="",
                        dest='resume_checkpoint',
                        help='Load the specified checkpoint to resume training')
    parser.add_argument('--save_after_time', action="store", type=int, default=0,
                        dest='save_after_time',
                        help='Save model after x seconds since start')
    parser.add_argument('--eval', action="store_true", default=False,
                        dest='eval',
                        help='Evaluate the trained model on test set')                
    arguments = parser.parse_args()

    # Run training using parsed arguments
    model_file_name = train_model(model_type=arguments.model, dataset=arguments.dataset, learning_rate=arguments.lr, lr_decay=arguments.lr_decay, batch_size=arguments.batch_size,
                                epochs=arguments.epochs, accumulate_steps=arguments.accumulate_steps, model_dir=arguments.model_dir, data_dir=arguments.data_dir,
                                hidden_units=arguments.hidden_units, validate=arguments.validate, validate_epochs=arguments.validate_every, graph_type=arguments.graph_type,
                                integrator=arguments.integrator, pre_load_graphs=arguments.pre_load_graphs, data_loader_workers=arguments.data_loader_workers, 
                                smooth_lr_decay=arguments.smooth_lr_decay, target_step=arguments.target_step, cpu=arguments.cpu, experiment_dir=arguments.experiment_dir, 
                                log_dir=arguments.log_dir, resume_checkpoint=arguments.resume_checkpoint, save_after_time=arguments.save_after_time)

    if arguments.eval:
        evaluate_model(model_file=model_file_name, dataset=arguments.dataset, model_dir=arguments.model_dir, data_dir=arguments.data_dir, experiment_dir=arguments.experiment_dir, pre_load_graphs=arguments.pre_load_graphs)
    