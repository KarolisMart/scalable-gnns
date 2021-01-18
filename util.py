import numpy as np
import shutil
import os
import torch
from itertools import accumulate, chain
from collections import abc


def recreate_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
    except OSError:
        print("Directory %s does not exist" % folder_path)
    else:
        print("Successfully deleted the old directory %s" % folder_path)
    try:
        os.makedirs(folder_path)
    except OSError:
        print("Creation of the directory %s failed" % folder_path)
    else:
        print("Successfully created the directory %s" % folder_path)

def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
    except OSError:
        print("Directory %s exists" % folder_path)
    else:
        print("Successfully created the directory %s" % folder_path)

def full_graph_senders_and_recievers(n_particles, batch_size=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    edges_per_node = n_particles-1
    n_edges = n_particles * edges_per_node

    # Create sender and reciever edge lists
    # All possible n*(n-1) edges are modeled. In R_r blocks of (n-1) subsequent edges have the same reciever while senders R_s change from smallest possible ID to largest.
    R_s_idx = torch.zeros(n_edges, dtype=torch.long, device=device)
    R_r_idx = torch.zeros(n_edges, dtype=torch.long, device=device)
    for i in range(n_particles):
        R_s_idx[i*edges_per_node:(i+1)*edges_per_node] = i
        for j in range(i):
            R_r_idx[i*edges_per_node+j] = j
        for j in range(i+1, n_particles):
            R_r_idx[i*edges_per_node+j-1] = j

    # Repeat R_s and R_r for each sample in the batch
    if batch_size:
        R_s_idx = R_s_idx.unsqueeze(0).expand(batch_size, n_edges)
        R_r_idx = R_r_idx.unsqueeze(0).expand(batch_size, n_edges)   

    return R_s_idx, R_r_idx

def nn_graph_senders_and_recievers(current_state, n_neighbours, box_size, device):
    batch_size = current_state.size(0)
    n_particles = current_state.size(1)
    n_edges = n_particles * n_neighbours

    indices = list(range(1, n_particles))
    R_s_idx = torch.zeros(batch_size, n_edges, device=device, dtype=torch.long)
    R_r_idx = torch.zeros(batch_size, n_edges, device=device, dtype=torch.long)

    # Loop over edge recievers
    for i in range(n_particles):
        # Get distance between neighbours
        # Sender features current_state[:, indices,:], Reciever features current_state[:,i,:].unsqueeze(1), Dist = sender_pos - reciever_pos
        dist = pbc_diff(current_state[:, indices, -4:-2], current_state[:,i, -4:-2].unsqueeze(1), box_size=box_size)
        if i < n_particles-1:
            indices[i] -= 1
        
        # Get closest n neigtbour (sender) ids
        neighbour_dist, neighbour_idx = torch.topk(torch.norm(dist, dim=-1, p=2), n_neighbours, largest=False)

        # Fix the sender id (we dropped self connection on reciever) for senders that have true id > reciever id (as currently their id is lower by 1)
        neighbour_idx[neighbour_idx >= i] += 1

        R_s_idx[:, i*n_neighbours:(i+1)*n_neighbours] = neighbour_idx
        R_r_idx[:, i*n_neighbours:(i+1)*n_neighbours] = torch.tensor([i]*n_neighbours, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, n_neighbours)

    return R_s_idx, R_r_idx

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
            cell_mask = x_neg & y_pos
            new_ref_point = [ref_point[0] - ref_point_step, ref_point[1] + ref_point_step]
            cells_1st = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)
            # 2nd cell
            cell_mask = x_pos & y_pos
            new_ref_point = [ref_point[0] + ref_point_step,  ref_point[1] + ref_point_step]
            cells_2nd = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)
            # 3rd cell
            cell_mask = x_neg & y_neg
            new_ref_point = [ref_point[0] - ref_point_step,  ref_point[1] - ref_point_step]
            cells_3th = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)
            # 4th cell
            cell_mask = x_pos & y_neg
            new_ref_point = [ref_point[0] + ref_point_step,  ref_point[1] - ref_point_step]
            cells_4th = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)
            # Reorder cells to be in row major order
            cells += list(chain(*zip(*[cells_1st[i::half_row_len] for i in range(half_row_len)], *[cells_2nd[i::half_row_len] for i in range(half_row_len)])))
            cells += list(chain(*zip(*[cells_3th[i::half_row_len] for i in range(half_row_len)], *[cells_4th[i::half_row_len] for i in range(half_row_len)])))
        else:
            # 1st cell
            cell_mask = x_neg & y_pos
            cells.append(ids[cell_mask])
            # 2nd cell
            cell_mask = x_pos & y_pos
            cells.append(ids[cell_mask])
            # 3rd cell
            cell_mask = x_neg & y_neg
            cells.append(ids[cell_mask])
            # 4th cell
            cell_mask = x_pos & y_neg
            cells.append(ids[cell_mask])
        return cells

def weighted_mean(arr, weights, dim):
        w_sum = torch.sum(arr * weights.unsqueeze(1), dim=dim)
        total_weight = torch.sum(weights, dim=dim, keepdim=True)
        return w_sum / total_weight

def hierarchical_graph_senders_and_recievers(current_state, levels, box_size, device):
    if len(current_state.shape) == 3:
        current_state = current_state[0]
    n_particles = current_state.shape[0]
    n_cells = 4**levels
    row_len = 2**levels

    if levels > 2:
        n_edges_per_super_vertex = 9*4 - 9
    else:
        n_edges_per_super_vertex = 16 - 9

    if levels < 2:
        raise ValueError('Must have at least 2 levels')

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
        return (torch.arange(row_len, device=device).reshape(cluster_row_len, 2).repeat(1, 2) + torch.tensor([0,0,row_len,row_len], device=device, dtype=torch.long).unsqueeze(0)).repeat(cluster_row_len,1) + torch.arange(cluster_row_len, device=device).unsqueeze(1).repeat(1,cluster_row_len).reshape(-1,1) * 2 * row_len

    # cell clusters - higher lever super nodes, each has 4 cell super nodes in it
    n_cell_clusters = n_cells//4
    cluster_row_len = row_len//2
    cell_clusters = build_clusters(row_len, cluster_row_len)

    graph = []
    cell_assignments = []
    super_vertices = []

    super_vertex_edges = torch.zeros((n_cells*n_edges_per_super_vertex, 2), dtype=torch.long, device=device)

    # Get list of indices to pass to cell function
    indices = torch.arange(n_particles, dtype=torch.long, device=device)

    # Split particles into cells for trajectory step
    cells = get_cells(indices, current_state[:, -4:-2], levels, ref_point=[0,0], box_size=box_size)

    # Track cells that have no particles to remove edges from/to them
    non_empty_cells = []

    # Iterate over cells
    for q, cell in enumerate(cells):
        # 9 cell box around and including q
        extended_ids = get_neighboring_ids(q, row_len=row_len, n_cells=n_cells)
        extended_cell = torch.cat([cells[i] for i in extended_ids], dim=-1)
        # Edges between vertices belonging to the same cell and from vertices from nearby cells to vertices in current cell (reciever: cell vertices, sender: extended_cell vertices)
        edges_within = torch.stack(torch.meshgrid(cell,extended_cell)).T.reshape(-1,2)[:,(1,0)]
        # Drop self connections
        edges_within = edges_within[edges_within[:,0] != edges_within[:,1]]
        graph.append(edges_within)

        # Cluster id this cell belongs to
        cluster_id = cluster_row_len * ((q // row_len) // 2) + (q % row_len) // 2
        neighboring_cluster_ids = get_neighboring_ids(cluster_id, row_len=cluster_row_len, n_cells=n_cell_clusters)
        cells_in_neighboring_clusters = cell_clusters[neighboring_cluster_ids].reshape(-1)

        # Edges between current super vertex and other super vertices in neighboring clusters but not in extended_ids
        other_cell_ids = cells_in_neighboring_clusters[~(cells_in_neighboring_clusters[..., None] == torch.tensor(extended_ids, device=device, dtype=torch.long)).any(-1)]
        super_vertex_edges[q*n_edges_per_super_vertex:(q+1)*n_edges_per_super_vertex, :] = torch.stack([other_cell_ids, torch.tensor([q], device=device, dtype=torch.long).repeat(n_edges_per_super_vertex)]).T

        # Edges from vertices in this cell to cell super vertex - super verices have ids starting at 0
        edges_to_super_node = torch.stack([torch.tensor([q], device=device).repeat(cell.size(0)), cell]).T
        cell_assignments.append(edges_to_super_node)

        # Compute super vertex params [total mass, center of mass (x,y), center of mass velocity (x,y)] - 5 params total
        if len(cell) > 0:
            particles_in_cell = current_state[cell]
            super_vertices.append(torch.cat([torch.sum(particles_in_cell[:,0], dim=0, keepdims=True), weighted_mean(particles_in_cell[:,1:], weights=particles_in_cell[:,0], dim=0)], axis=-1))
            non_empty_cells.append(q)

    graph = torch.cat(graph)

    cell_assignments = torch.cat(cell_assignments)
    super_vertices = torch.stack(super_vertices)

    # Generate new ids for non epty cells
    non_empty_cells = torch.tensor(non_empty_cells, dtype=torch.long, device=device)
    new_cell_ids = torch.arange(non_empty_cells.size(0), dtype=torch.long, device=device) 

    # Remove edges that belong to empty super vertices
    super_vertex_edges = super_vertex_edges[(super_vertex_edges[:, :, None] == non_empty_cells).any(-1).all(dim=1)]

    # Re-index all the non empty cells with new ids
    for new_idx, old_idx in enumerate(non_empty_cells):
        cell_assignments[:,0][cell_assignments[:,0] == old_idx] = new_idx
        super_vertex_edges[super_vertex_edges == old_idx] = new_idx

    # Sort assignments w.r.t. vertex ids to use in scatter and gather operations
    cell_assignments = cell_assignments[cell_assignments[:,1].argsort()]

    assignment = [cell_assignments.unsqueeze(0)]
    super_vertices = [super_vertices.unsqueeze(0)]
    super_vertex_edges = [super_vertex_edges.unsqueeze(0)]
    super_vertex_ids = [non_empty_cells.unsqueeze(0)]

    # Build higher level super graphs
    for level in reversed(range(2, levels)):
        
        n_higher_level_clusters = 4**(level-1)
        n_current_level_clusters = 4**level
        higher_level_row_len = 2**(level-1)
        current_level_row_len = 2**level
        lower_level_row_len = 2**(level+1)
        lower_level_super_vertices = super_vertices[-1][0]
        lower_level_super_vertex_ids = super_vertex_ids[-1][0]
                                
        higher_level_clusters = build_clusters(current_level_row_len, higher_level_row_len)
        clusters = build_clusters(lower_level_row_len, current_level_row_len)
        assingments_to_current_level_super_vertices = []
        current_level_super_vertex_features = []
        current_level_super_vertex_edges = []

        non_empty_clusters = []

        for c, cluster in enumerate(clusters):
            # Get all non empty cells from lower level that belong to current cluster
            cluster = torch.arange(len(lower_level_super_vertex_ids), device=device)[(lower_level_super_vertex_ids[..., None] == cluster).any(-1)]

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
                cells_in_neighboring_clusters = cells_in_neighboring_clusters[~(cells_in_neighboring_clusters[..., None] == torch.tensor(neighbour_ids, device=device, dtype=torch.long)).any(-1)]
                current_level_super_vertex_edges.append(torch.stack([cells_in_neighboring_clusters, torch.tensor([c], device=device, dtype=torch.long).repeat(len(cells_in_neighboring_clusters))]).T)

                assingments_to_current_level_super_vertices.append(torch.stack([torch.tensor([c], device=device, dtype=torch.long).repeat(len(cluster)), cluster]).T)

                # Compute super vertex params [total mass, center of mass (x,y), center of mass velocity (x,y)] - 5 params total
                cells_in_cluster = lower_level_super_vertices[cluster]
                current_level_super_vertex_features.append(torch.cat([torch.sum(cells_in_cluster[:,0], dim=0, keepdims=True), weighted_mean(cells_in_cluster[:,1:], weights=cells_in_cluster[:,0], dim=0)], axis=-1))
                non_empty_clusters.append(c)
           
        assingments_to_current_level_super_vertices = torch.cat(assingments_to_current_level_super_vertices)
        current_level_super_vertex_features = torch.stack(current_level_super_vertex_features)
        current_level_super_vertex_edges = torch.cat(current_level_super_vertex_edges)

        # Re-index non-empty higher level super nodes
        non_empty_clusters =  torch.tensor(non_empty_clusters, dtype=torch.long, device=device)
        new_current_level_super_vertex_ids = torch.arange(non_empty_clusters.size(0), dtype=torch.long, device=device)

        # Remove edges that belong to empty clusters
        current_level_super_vertex_edges = current_level_super_vertex_edges[(current_level_super_vertex_edges[:, :, None] == non_empty_clusters).any(-1).all(dim=1)]

        # Re-index all the non empty clusters with new ids
        for new_idx, old_idx in enumerate(non_empty_clusters):
            assingments_to_current_level_super_vertices[:,0][assingments_to_current_level_super_vertices[:,0] == old_idx] = new_idx
            current_level_super_vertex_edges[current_level_super_vertex_edges == old_idx] = new_idx
       
        assingments_to_current_level_super_vertices = assingments_to_current_level_super_vertices[assingments_to_current_level_super_vertices[:,1].argsort()]

        assignment.append(assingments_to_current_level_super_vertices.unsqueeze(0))
        super_vertices.append(current_level_super_vertex_features.unsqueeze(0))
        super_vertex_edges.append(current_level_super_vertex_edges.unsqueeze(0))
        super_vertex_ids.append(non_empty_clusters.unsqueeze(0))
    
    R_s = graph[:,0].unsqueeze(0)
    R_r = graph[:,1].unsqueeze(0)

    return R_s, R_r, assignment, super_vertices, super_vertex_edges

def collate_into_one_graph(batch):
    # Do custom collate for cell and hierarchical graphs - make one big graph for the batch
    # Unsqeeze 1st dim so that code is interchangable with batched graphs
    if isinstance(batch[0], abc.Sequence) and len(batch[0]) > 4:
        # Hierarchical graph
        transposed = list(zip(*batch))
        # sample format: inputs, targets, R_s, R_r, assignment, V_super, targets_super,  R_s_super, R_r_super
        n_particles = transposed[0][0].size(0)
        batch_size = len(transposed[0])
        inputs = torch.cat(transposed[0], 0).unsqueeze(0) 
        targets = torch.cat(transposed[1], 0).unsqueeze(0)
        # Correct vertex ids
        R_s = torch.cat([el + n_particles*i for i,el in enumerate(transposed[2])], 0).unsqueeze(0)
        R_r = torch.cat([el + n_particles*i for i,el in enumerate(transposed[3])], 0).unsqueeze(0)
        assignment = []
        V_super = []
        R_s_super = []
        super_graph = []
        # Build list that tracks particle counts at lower (previous) layer
        n_particles = list(accumulate([0] + [n_particles]*(batch_size-1)))
        if isinstance(transposed[4][0][0], torch.LongTensor):
            for level in range(len(transposed[4][0])):
                n_super_nodes = list(accumulate([0] + [el[level].shape[0] for el in transposed[5]][:-1])) 
                # Correct super vertex ids and vertex ids 
                assignment.append(torch.cat([el[level] + torch.tensor([n_super_nodes[i], n_particles[i]], dtype=torch.long) for i,el in enumerate(transposed[4])], 0).unsqueeze(0))
                V_super.append(torch.cat([el[level] for el in transposed[5]], 0).unsqueeze(0))
                # Correct super vertex ids
                super_graph.append(torch.cat([el[level] + n_super_nodes[i] for i,el in enumerate(transposed[6])], 0).unsqueeze(0))
                # Set particles to previous layer's super nodes
                n_particles = n_super_nodes
        else:
            for level in range(len(transposed[4][0])):
                n_super_nodes = list(accumulate([0] + [el[level].shape[0] for el in transposed[5]][:-1])) 
                # Correct super vertex ids and vertex ids 
                assignment.append(torch.cat([torch.from_numpy(el[level].astype(np.int64)) + torch.tensor([n_super_nodes[i], n_particles[i]], dtype=torch.long) for i,el in enumerate(transposed[4])], 0).unsqueeze(0))
                V_super.append(torch.cat([torch.from_numpy(el[level]).float() for el in transposed[5]], 0).unsqueeze(0))
                # Correct super vertex ids
                super_graph.append(torch.cat([torch.from_numpy(el[level].astype(np.int64)) + n_super_nodes[i] for i,el in enumerate(transposed[6])], 0).unsqueeze(0))
                # Set particles to previous layer's super nodes
                n_particles = n_super_nodes
        return [inputs, targets, R_s, R_r, assignment, V_super, super_graph]
    else:
        # Use default collate in other cases (fully connected and nn graphs)
        return torch.utils.data._utils.collate.default_collate(batch)

def pbc_poss(pos, box_size=6):
    pos[pos >= box_size/2] -= box_size
    pos[pos < -box_size/2] += box_size
    return pos

def pbc_diff(pos1, pos2, box_size=6):
    diff = pos1 - pos2
    # Periodic boundry conditions
    diff[diff > box_size/2] = diff[diff > box_size/2] - box_size 
    diff[diff <= -box_size/2] = diff[diff <= -box_size/2] + box_size
    return diff

def pbc_rms_error(predictions, targets, box_size=6):
    loss = np.sqrt(np.mean(pbc_diff(predictions, targets, box_size=box_size)**2))
    return loss

def pbc_mean_relative_energy_error(predictions, box_size=6, physical_const=2, softening=False, softening_radius=0.001):
    # Mean relative error between first time step and end of trajectory
    total_energy_diff = 0
    for trajectory in predictions:
        true_energy = hamiltonian(trajectory[0], physical_const=physical_const, box_size=box_size, softening=softening, softening_radius=softening_radius)
        final_energy = hamiltonian(trajectory[-1], physical_const=physical_const, box_size=box_size, softening=softening, softening_radius=softening_radius)
        energy_diff = np.absolute((final_energy - true_energy)/true_energy)
        total_energy_diff += energy_diff
    loss = total_energy_diff / predictions.shape[0]
    return loss

def total_linear_momentum(states):
    masses = states[:, 0]
    velocities = states[:, -2:]

    return np.sum(np.tile(masses, (2, 1)).T * velocities, axis=0)

def total_angular_momentum(states):
    masses = states[:, 0]
    positions = states[:, -4:-2]
    velocities = states[:, -2:]

    return np.sum(np.cross(positions, np.tile(masses, (2, 1)).T * velocities))

def kinetic_energy(states):
    momentum = np.multiply(states[:, -2:], states[:, 0][:, np.newaxis])

    kinetic_energy = np.sum(np.divide(np.square(momentum), (2 * states[:, 0][:, np.newaxis])))

    return kinetic_energy

def potential_energy(states, physical_const=2, box_size=6, softening=False, softening_radius=0.001):
    n_particles = states.shape[0]

    has_charge = (states.shape[1] == 6) # check if data is from Coulomb simulation

    if n_particles > 1000:
        # If we have many particles iteration is faster
        potential_energy = np.zeros(1)

        for i in range(n_particles):
            diff = pbc_diff(states[:, -4:-2], states[i, -4:-2], box_size=box_size)
            diff[i] = np.zeros((2)) # make sure current particle is not attracted to itself
            distance = np.linalg.norm(diff, axis=1)
            distance[i] = 1 # avoid division by 0 error
            if has_charge:
                m2 =  - np.multiply(states[:, 1], states[i, 1]) # q * q[i]
            else:
                m2 = np.multiply(states[:, 0], states[i, 0]) # m * m[i]
            m2[i] = np.zeros((1)) # make sure current particle is not attracted to itself
            if softening:
                potential_energy -= np.sum((np.divide((physical_const * m2), np.sqrt(distance**2 + softening_radius**2))))
            else:
                potential_energy -= np.sum((np.divide((physical_const * m2), distance)))
        self_ids = np.arange(n_particles)

        return potential_energy/2
    else:
        self_ids = np.arange(n_particles)

        diff = pbc_diff(np.repeat(states[np.newaxis, :, -4:-2], n_particles, axis=0), states[:, np.newaxis, -4:-2], box_size=box_size)
        distance = np.linalg.norm(diff, axis=2)
        distance[self_ids, self_ids] = 1 # avoid division by 0 error
        if has_charge:
            m2 = - np.multiply(np.repeat(states[np.newaxis, :, 1], n_particles, axis=0), states[:, np.newaxis, 1]) # q * q[i]
        else:
            m2 = np.multiply(np.repeat(states[np.newaxis, :, 0], n_particles, axis=0), states[:, np.newaxis, 0]) # m * m[i]
        m2[self_ids, self_ids] = np.zeros((1)) # make sure particle is not attracted to itself
        if softening:
            potential_energy = -np.sum((np.divide((physical_const * m2), np.sqrt(distance**2 + softening_radius**2))))
        else:
            potential_energy = -np.sum((np.divide((physical_const * m2), distance)))

    return potential_energy/2 # we double count

def hamiltonian(states, physical_const=2, box_size=6, softening=False, softening_radius=0.001):
    # Total energy of the system
    return kinetic_energy(states) + potential_energy(states, physical_const, box_size=box_size, softening=softening, softening_radius=softening_radius)
    
def parameter_counts(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_params)
            else:
                print(name, ':', num_params)
            total_params += num_params
    
    print('number of trainable parameters =', total_params)
