import torch
import torch.nn as nn
from collections import deque


# Ensure x and y stay inside the box and follow PBC
def apply_PBC_to_coordinates(coordinates, box_size=6):
    # Only apply to coordinate columns
    coordinates[:,:,-4:-2][coordinates[:,:,-4:-2] >= box_size/2] -= box_size
    coordinates[:,:,-4:-2][coordinates[:,:,-4:-2] < -box_size/2] += box_size
    return coordinates

def apply_PBC_to_distances(distances, box_size=6):
    # Only apply to postion columns
    distances[:,:,-4:-2][distances[:,:,-4:-2] > box_size/2] -= box_size
    distances[:,:,-4:-2][distances[:,:,-4:-2] <= -box_size/2] += box_size
    return distances

# Custom MSE loss that takes periodic boundry conditions into account
def PBC_MSE_loss(output, target, box_size=6):
    # Get difference
    error = output - target
    # Deal with periodic boundry conditions
    error = apply_PBC_to_distances(error, box_size=box_size)
    # Get MSE
    loss = torch.mean((error)**2)
    return loss


class EdgeModel(torch.nn.Module):
    def __init__(self, input_dim=64, output_dim=64, softplus=False, box_size=6):
        super(EdgeModel, self).__init__()
        self.box_size = box_size
        if softplus:
            self.edge_mlp = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Softplus(), nn.Linear(output_dim, output_dim), nn.Softplus())
        else:
            self.edge_mlp = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim), nn.ReLU())

    def forward(self, V_no_pos, V_pos, R_s, R_r, u=None, different_reciever=None, different_reciever_pos=None):
        # Get edge features (sender mass (+charge) and speed, reciever mass and speed, pbc_distance(sender_poss - reciever_pos))
        if different_reciever is None or different_reciever_pos is None:
            # Edges between levels
            E = torch.cat([V_no_pos.gather(1, R_s.expand(R_s.size(0), R_s.size(1), V_no_pos.size(2))), V_no_pos.gather(1, R_r.expand(R_r.size(0), R_r.size(1), V_no_pos.size(2))), (V_pos.gather(1, R_s.expand(R_s.size(0), R_s.size(1), V_pos.size(2))) - V_pos.gather(1, R_r.expand(R_r.size(0), R_r.size(1), V_pos.size(2))))], dim=-1)
        else:
            # If reciever features are supplied in a different matrix (recievers are different type nodes from senders)
            E = torch.cat([V_no_pos.gather(1, R_s.expand(R_s.size(0), R_s.size(1), V_no_pos.size(2))), different_reciever.gather(1, R_r.expand(R_r.size(0), R_r.size(1), different_reciever.size(2))), (V_pos.gather(1, R_s.expand(R_s.size(0), R_s.size(1), V_pos.size(2))) - different_reciever_pos.gather(1, R_r.expand(R_r.size(0), R_r.size(1), different_reciever_pos.size(2))))], dim=-1)
        # Deal with periodic boundry conditions
        E[:,:,-2:][E[:,:,-2:] > self.box_size/2] -= self.box_size
        E[:,:,-2:][E[:,:,-2:] <= -self.box_size/2] += self.box_size
        if u is not None:
            E = torch.cat([E, u.unsqueeze(2).expand(E.size(0), E.size(1), 1)], dim=-1)
        return self.edge_mlp(E)


class NodeModel(torch.nn.Module):
    def __init__(self, input_dim=64, output_dim=64, softplus=False):
        super(NodeModel, self).__init__()
        if softplus:
            self.node_mlp = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Softplus(), nn.Linear(output_dim, output_dim), nn.Softplus(), nn.Linear(output_dim, output_dim), nn.Softplus())
        else:
            self.node_mlp = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim), nn.ReLU())

    def forward(self, V, E_n, u=None, R_r=None):
        if R_r is None:
            # Aggregate edges for each reciever node using the knwoledge that subsequent n_edges_per_node blocks of rows belong to the same reciever per R_r construction
            out = torch.sum(E_n.view(V.size(0), V.size(1), E_n.size(1) // V.size(1), E_n.size(-1)), dim=2)
        else:
            # If recievers can have a different number of edges
            out = torch.zeros((E_n.size(0), V.size(1), E_n.size(2)), device=E_n.device).scatter_add_(1, R_r.expand(R_r.size(0), R_r.size(1), E_n.size(2)), E_n)
        out = torch.cat([V, out], dim=-1)
        if u is not None:
            # Expand global param u from one per sample in a batch to one per particle
            out = torch.cat([out, u.unsqueeze(2).expand(out.size(0), out.size(1), 1)], dim=-1)
        return self.node_mlp(out)


class GlobalModel(torch.nn.Module):
    def __init__(self, input_dim=64, output_dim=64):
        super(GlobalModel, self).__init__()
        self.global_mlp = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Softplus(), nn.Linear(output_dim, output_dim), nn.Softplus())

    def forward(self, *args, u=None):
        out = torch.cat([torch.sum(arg, axis=1) for arg in args], dim=-1)
        if u is not None:
            # Expand global param u from one per sample in a batch to one per particle
            out = torch.cat([out, u.unsqueeze(2).expand(out.size(0), out.size(1), 1)], dim=-1)
        return self.global_mlp(out)


class BaseIntegratorModel(torch.nn.Module):

    def forward_step(self, mass_charge, V_0, *args):
        raise NotImplementedError

    def euler(self, dt, mass_charge, V_0, *args):
        # Euler method
        dt = dt.unsqueeze(2).expand(V_0.size(0), V_0.size(1), 1)
        k1 = self.forward_step(mass_charge, V_0, *args)
        dy = dt * k1
        return apply_PBC_to_coordinates(V_0 + dy, box_size=self.box_size)

    def rk4(self, dt, mass_charge, V_0, *args):
        # NOTE There is an alternative formulation with a smaller error
        # Expand dt from one per sample in a batch to one per particle
        dt = dt.unsqueeze(2).expand(V_0.size(0), V_0.size(1), 1)
        dt2 = dt / 2.0
        k1 = self.forward_step(mass_charge, V_0, *args)
        k2 = self.forward_step(mass_charge, apply_PBC_to_coordinates(V_0 + k1 * dt2, box_size=self.box_size), *args)
        k3 = self.forward_step(mass_charge, apply_PBC_to_coordinates(V_0 + k2 * dt2, box_size=self.box_size), *args)
        k4 = self.forward_step(mass_charge, apply_PBC_to_coordinates(V_0 + k3 * dt, box_size=self.box_size), *args)
        dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return apply_PBC_to_coordinates(V_0 + dy, box_size=self.box_size)

    def forward(self, state, R_s, R_r, dt):
        raise NotImplementedError


class DeltaGN(torch.nn.Module):
    
    def __init__(self, box_size=6, edge_output_dim=-1, node_output_dim=-1, simulation_type='gravity'):
        super(DeltaGN, self).__init__()

        if edge_output_dim < 1:
            edge_output_dim = 150
        if node_output_dim < 1:
            node_output_dim = 100
        
        self.simulation_type = simulation_type
        if self.simulation_type == 'coulomb':
            node_input_dim = 4
            # Used to drop position from particles/nodes
            self.non_pos_indices = list([0,1,4,5]) # (mass, charge, vx, vy)
        else:
            node_input_dim = 3
            # Used to drop position from particles/nodes
            self.non_pos_indices = list([0,3,4]) # (mass, vx, vy)
        
        self.edge_model = EdgeModel(input_dim=2*node_input_dim+2+1, output_dim=edge_output_dim, box_size=box_size) # input dim: sender and reciever nodes + disntace vector + dt

        self.node_model = NodeModel(input_dim=node_input_dim+edge_output_dim+1, output_dim=node_output_dim) # input dim: node features + embedded edge features + dt

        # Linear layer to transform node embeddings to canonical coordinate change (four features: (x,y,v_x,v_y))
        self.linear = nn.Linear(node_output_dim, 4)

        # Set box size
        self.box_size = box_size


    def forward(self, V, R_s, R_r, dt):

        R_s = R_s.unsqueeze(2)
        R_r = R_r.unsqueeze(2)

        # Edge block
        E_n = self.edge_model(V[:, :, self.non_pos_indices], V[:, :, -4:-2], R_s, R_r, dt)
        
        # Node block
        V_n = self.node_model(V[:, :, self.non_pos_indices], E_n, dt)

        new_coordinates = V[:, :, -4:] + self.linear(V_n)

        # Deal with periodic boundry conditions
        return apply_PBC_to_coordinates(new_coordinates, box_size=self.box_size)


class HOGN(BaseIntegratorModel):
    
    def __init__(self, box_size=6, edge_output_dim=150, node_output_dim=100, global_output_dim=100, integrator='rk4', simulation_type='gravity'):
        super(HOGN, self).__init__()

        if edge_output_dim < 1:
            edge_output_dim = 150
        if node_output_dim < 1:
            node_output_dim = 100 
        if global_output_dim < 1:
            global_output_dim = 100 

        self.simulation_type = simulation_type
        # Set number of node features, excluding the position (x,y)
        if self.simulation_type == 'coulomb':
            node_input_dim = 4 # (mass, charge, px, py)
        else:
            node_input_dim = 3 # (mass, px, py)

        self.edge_model = EdgeModel(input_dim=2*node_input_dim+2, output_dim=edge_output_dim, softplus=True, box_size=box_size) # input dim: sender and reciever node features  + disntace vector

        self.node_model = NodeModel(input_dim=node_input_dim+edge_output_dim, output_dim=node_output_dim, softplus=True) # input dim: input node features + embedded edge features

        self.global_model = GlobalModel(input_dim=node_output_dim+edge_output_dim, output_dim=global_output_dim) # input dim: embedded node features and embedded edge features

        # Linear layer to transform global embeddings to a Hamiltonian
        self.linear = nn.Linear(global_output_dim, 1)

        # Set box size
        self.box_size = box_size

        # Set integrator to use
        self.integrator = integrator

    # Here vertices V are in canonical coordinates [x,y,px,py]
    def forward_step(self, mass_charge, V, R_s, R_r):

        # Drop position from particles/nodes and add mass and charge (if present)
        V_no_pos = torch.cat([mass_charge, V[:,:,2:]], dim=2)

        R_s = R_s.unsqueeze(2)
        R_r = R_r.unsqueeze(2)

        # Edge block
        E_n = self.edge_model(V_no_pos, V[:,:,:2], R_s, R_r)

        # Node block
        V_n = self.node_model(V_no_pos, E_n)

        # Global block
        U_n = self.global_model(V_n, E_n)

        # Hamiltonian
        H = self.linear(U_n)

        # Hamiltonian derivatives w.r.t inputs = dH/dq dH/dp
        partial_derivatives = torch.autograd.grad(H.sum(), V, create_graph=True)[0] #, only_inputs=True

        # Return dq and dp
        return torch.cat([partial_derivatives[:,:,2:], partial_derivatives[:,:,:2] * (-1.0)], dim=2)  # dq=dH/dp, dp=-dH/dq

    def forward(self, state, R_s, R_r, dt):
        # Transform inputs [m, x, y, vx, vy] to canonical coordinates [x,y,px,py]
        mass_charge = state[:,:,:-4] # if no charge = [m]; with charge = [m, c]
        momentum = state[:,:,-2:] * mass_charge[:,:,0].unsqueeze(2)
        V = torch.cat([state[:,:,-4:-2], momentum], dim=2)
        # Require grad to be able to compute partial derivatives
        if not V.requires_grad:
            V.requires_grad = True
        
        # Compute updated canonical coordinates
        if self.integrator == 'rk4':
            new_canonical_coordinates = self.rk4(dt, mass_charge, V, R_s, R_r)
        elif self.integrator == 'euler':
            new_canonical_coordinates = self.euler(dt, mass_charge, V, R_s, R_r)
        else:
            raise Exception
        
        # Convert back to original state format [x, y, vx, vy]
        velocity = torch.div(new_canonical_coordinates[:,:,2:], mass_charge[:,:,0].unsqueeze(2))
        new_state = torch.cat([new_canonical_coordinates[:,:,:2], velocity], dim=2)
        return new_state


class HierarchicalDeltaGN(torch.nn.Module):
    
    def __init__(self, box_size=6, edge_output_dim=-1, node_output_dim=-1, simulation_type='gravity'):
        super(HierarchicalDeltaGN, self).__init__()

        if edge_output_dim < 1:
            edge_output_dim = 150
        if node_output_dim < 1:
            node_output_dim = 100

        self.simulation_type = simulation_type
        # Set number of node features, excluding the position (x,y)
        if self.simulation_type == 'coulomb':
            node_input_dim = 4
            self.non_pos_indices = list([0,1,4,5]) # (mass, charge, vx, vy)
        else:
            node_input_dim = 3
            self.non_pos_indices = list([0,3,4]) # (mass, vx, vy)

        self.edge_to_super_model = EdgeModel(input_dim=2*node_input_dim+2+1, output_dim=node_output_dim, box_size=box_size) # input dim: sender (particle) and reciever (super/cell) nodes + disntace vector + dt
        
        self.edge_to_upper_model = EdgeModel(input_dim=node_output_dim+2*node_input_dim+2+1, output_dim=node_output_dim, box_size=box_size) # input dim: sender and reciever (super) nodes (base node_input_features + features from vertex node embedding) + disntace vector + dt

        self.super_edge_model = EdgeModel(input_dim=2*(node_input_dim+node_output_dim)+2+1, output_dim=edge_output_dim, box_size=box_size) # input dim: sender and reciever nodes + disntace vector + dt

        self.super_node_model = NodeModel(input_dim=node_input_dim+node_output_dim+edge_output_dim+1, output_dim=node_output_dim) # input dim: input node features + updated features + embedded super edge features + dt

        self.edge_from_super_model = EdgeModel(input_dim=2*node_input_dim+node_output_dim+2+1, output_dim=edge_output_dim, box_size=box_size) # input dim: sender (super) and reciever (particle) nodes + disntace vector + dt

        self.edge_from_upper_model = EdgeModel(input_dim=2*(node_input_dim+node_output_dim)+2+1, output_dim=edge_output_dim, box_size=box_size) # input dim: sender (super) and reciever (cell) nodes + disntace vector + dt

        self.edge_model = EdgeModel(input_dim=2*node_input_dim+2+1, output_dim=edge_output_dim, box_size=box_size) # input dim: sender and reciever nodes + disntace vector + dt

        self.node_model = NodeModel(input_dim=node_input_dim+edge_output_dim+1, output_dim=node_output_dim) # input dim: node features  + embedded edge features + dt

        # Linear layer to transform node embeddings to canonical coordinate change (four features: (x,y,v_x,v_y))
        self.linear = nn.Linear(node_output_dim, 4)

        # Set box size
        self.box_size = box_size


    def forward(self, V, R_s, R_r, assignments, V_supers, super_graphs, dt):

        R_s = R_s.unsqueeze(2)
        R_r = R_r.unsqueeze(2)

        R_vertex_to_super_s = assignments[0][:,:,1].unsqueeze(2)
        R_vertex_to_super_r = assignments[0][:,:,0].unsqueeze(2)

        ### Embedding of particles into a super graph

        # Edge block
        V_lower_pos = V_supers[0][:, :, -4:-2]

        E_to_super = self.edge_to_super_model(V[:, :, self.non_pos_indices], V[:, :, -4:-2], R_vertex_to_super_s, R_vertex_to_super_r, dt, V_supers[0][:, :, self.non_pos_indices], V_supers[0][:, :, -4:-2])
        
        # Sum up incomming influences to the node
        V_lower = torch.zeros((E_to_super.size(0), V_supers[0].size(1), E_to_super.size(2)), device=E_to_super.device).scatter_add_(1, R_vertex_to_super_r.expand(R_vertex_to_super_r.size(0), R_vertex_to_super_r.size(1), E_to_super.size(2)), E_to_super)
        V_lower = torch.cat([V_supers[0][:, :, self.non_pos_indices], V_lower], dim=-1)

        embeddings = deque([[V_lower, V_lower_pos]])
        ##### Upward pass
        for assignment, V_super in zip(assignments[1:], V_supers[1:]):
            R_vertex_to_super_s = assignment[:,:,1].unsqueeze(2)
            R_vertex_to_super_r = assignment[:,:,0].unsqueeze(2)

            # Edge block
            E_to_super = self.edge_to_upper_model(V_lower, V_lower_pos, R_vertex_to_super_s, R_vertex_to_super_r, dt, V_super[:, :, self.non_pos_indices], V_super[:, :, -4:-2])
            
            # Sum up incomming influences to the node
            V_lower = torch.zeros((E_to_super.size(0), V_super.size(1), E_to_super.size(2)), device=E_to_super.device).scatter_add_(1, R_vertex_to_super_r.expand(R_vertex_to_super_r.size(0), R_vertex_to_super_r.size(1), E_to_super.size(2)), E_to_super)
            
            V_lower_pos = V_super[:, :, -4:-2]
            V_lower = torch.cat([V_super[:, :, self.non_pos_indices], V_lower], dim=-1)

            embeddings.appendleft([V_lower, V_lower_pos])

        del R_vertex_to_super_s, R_vertex_to_super_r, V_lower_pos, E_to_super, V_lower

        V_current, V_current_pos = embeddings.popleft()
        R_s_super = super_graphs[-1][:,:,0].unsqueeze(2)
        R_r_super = super_graphs[-1][:,:,1].unsqueeze(2)
        R_super_to_vertex_s = assignments[-1][:,:,0].unsqueeze(2)
        R_super_to_vertex_r = assignments[-1][:,:,1].unsqueeze(2)

        E_current_n = self.super_edge_model(V_current, V_current_pos, R_s_super, R_r_super, dt)
    
        # Super node block
        V_upper = self.super_node_model(V_current, E_current_n, dt, R_r=R_r_super)
        V_upper = torch.cat([V_current[:, :, :-V_upper.size(2)], V_upper], dim=-1)
        V_upper_pos = V_current_pos

        ##### Downward pass
        for embedding, super_graph, assignment in zip(embeddings, reversed(super_graphs[:-1]), reversed(assignments[1:])):
            V_current, V_current_pos = embedding
            R_s_super = super_graph[:,:,0].unsqueeze(2)
            R_r_super = super_graph[:,:,1].unsqueeze(2)
            R_super_to_vertex_s = assignment[:,:,0].unsqueeze(2)
            R_super_to_vertex_r = assignment[:,:,1].unsqueeze(2)

            upper_influence = self.edge_from_upper_model(V_upper, V_upper_pos, R_super_to_vertex_s, R_super_to_vertex_r, dt, V_current, V_current_pos)

            E_current_n = self.super_edge_model(V_current, V_current_pos, R_s_super, R_r_super, dt)
            E_current_n = torch.cat([E_current_n, upper_influence], dim=1)
            R_r_super = torch.cat([R_r_super, R_super_to_vertex_r], dim=1)
        
            V_upper = self.super_node_model(V_current, E_current_n, dt, R_r=R_r_super)
            V_upper = torch.cat([V_current[:, :, :-V_upper.size(2)], V_upper], dim=-1)
            V_upper_pos = V_current_pos

        del E_current_n, R_s_super, R_r_super, embeddings, super_graphs

        R_super_to_vertex_s = assignments[0][:,:,0].unsqueeze(2)
        R_super_to_vertex_r = assignments[0][:,:,1].unsqueeze(2)

        ### Cell -> Particle edges
        E_n_s = self.edge_from_super_model(V_upper, V_upper_pos, R_super_to_vertex_s, R_super_to_vertex_r, dt, V[:, :, self.non_pos_indices], V[:, :, -4:-2])

        del assignments, V_supers, V_upper, V_upper_pos, R_super_to_vertex_s

        ### Calculating change of lower node particles
        # Edge block
        E_n = self.edge_model(V[:, :, self.non_pos_indices], V[:, :, -4:-2], R_s, R_r, dt)
        E_n = torch.cat([E_n, E_n_s], dim=1)
        R_r = torch.cat([R_r, R_super_to_vertex_r], dim=1)

        # # Node block
        V_n = self.node_model(V[:, :, self.non_pos_indices], E_n, dt, R_r=R_r)

        new_coordinates = V[:, :, -4:] + self.linear(V_n)

        # Deal with periodic boundry conditions
        return apply_PBC_to_coordinates(new_coordinates, box_size=self.box_size)


class HierarchicalHOGN(BaseIntegratorModel):
    
    def __init__(self, box_size=6, edge_output_dim=-1, node_output_dim=-1, integrator='rk4', simulation_type='gravity'):
        super(HierarchicalHOGN, self).__init__()

        if edge_output_dim < 1:
            edge_output_dim = 150
        if node_output_dim < 1:
            node_output_dim = 100

        self.node_output_dim = node_output_dim

        self.simulation_type = simulation_type
        # Set number of node features, excluding the position (x,y)
        if self.simulation_type == 'coulomb':
            node_input_dim = 4
            self.non_pos_indices = list([0,1,4,5]) # (mass, charge, px, py)
        else:
            node_input_dim = 3
            self.non_pos_indices = list([0,3,4]) # (mass, px, py)

        self.edge_to_super_model = EdgeModel(input_dim=2*node_input_dim+2, output_dim=node_output_dim, box_size=box_size, softplus=True) # input dim: sender (particle) and reciever (cell/super) nodes  + disntace vector
        
        self.edge_to_upper_model = EdgeModel(input_dim=node_output_dim+2*node_input_dim+2, output_dim=node_output_dim, box_size=box_size, softplus=True) # input dim: sender (particle) and reciever (cell/super) nodes (node input features + features from node embedding) + disntace vector

        self.super_edge_model = EdgeModel(input_dim=2*(node_input_dim+node_output_dim)+2, output_dim=edge_output_dim, box_size=box_size, softplus=True) # input dim: sender and reciever nodes + features from lower layer + distance vector

        self.super_node_model = NodeModel(input_dim=node_input_dim+node_output_dim+edge_output_dim, output_dim=node_output_dim, softplus=True)  # input dim: input node features + updated features + embedded super edge features + dt

        self.edge_from_super_model = EdgeModel(input_dim=2*node_input_dim+node_output_dim+2, output_dim=edge_output_dim, box_size=box_size, softplus=True) # input dim: sender (super) and reciever (particle) nodes (node input features + features from super node) + disntace vector

        self.edge_from_upper_model = EdgeModel(input_dim=2*(node_input_dim+node_output_dim)+2, output_dim=edge_output_dim, box_size=box_size, softplus=True) # input dim: sender (super) and reciever (particle) nodes (node input features  + embedded features) + disntace vector

        self.edge_model = EdgeModel(input_dim=2*node_input_dim+2, output_dim=edge_output_dim, box_size=box_size, softplus=True) # input dim: sender and reciever nodes + disntace vector

        self.node_model = NodeModel(input_dim=node_input_dim+edge_output_dim, output_dim=node_output_dim, softplus=True) # input dim: input node features + embedded edge features

        self.global_model = GlobalModel(input_dim=edge_output_dim + node_output_dim, output_dim=node_output_dim) # input dim: embedded node features and embedded edge features

        # Linear layer to transform node embeddings to canonical coordinate change (four features: (x,y,v_x,v_y))
        self.linear = nn.Linear(node_output_dim, 1)

        # Set box size
        self.box_size = box_size

        # Set integrator to use
        self.integrator = integrator

    def get_super_features(self, mass_charge, pos, momentum, R_to_upper_r, upper_count, batch_size=1):
        # Compute cell features from vertex features for gradient flow (appears to not be necessary)
        pos_weighted = pos * mass_charge[:,:,0].unsqueeze(2)
        pos_super = torch.zeros((batch_size, upper_count, pos.size(2)), device=pos_weighted.device).scatter_add_(1, R_to_upper_r.expand(R_to_upper_r.size(0), R_to_upper_r.size(1), pos.size(2)), pos_weighted)
        momentum_super = torch.zeros((batch_size, upper_count, momentum.size(2)), device=momentum.device).scatter_add_(1, R_to_upper_r.expand(R_to_upper_r.size(0), R_to_upper_r.size(1), momentum.size(2)), momentum)
        cell_mass_charge = torch.zeros((mass_charge.size(0), upper_count, mass_charge.size(2)), device=mass_charge.device).scatter_add_(1, R_to_upper_r.expand(R_to_upper_r.size(0), R_to_upper_r.size(1), mass_charge.size(2)), mass_charge)
        pos_super = pos_super / cell_mass_charge[:,:,0].unsqueeze(2)
        return torch.cat([cell_mass_charge, pos_super, momentum_super], axis=-1)

    def forward_step(self, mass_charge, V, R_s, R_r, assignments, V_supers, super_graphs):
        
        batch_size = V.size(0)

        # Drop position from particles/nodes
        V_no_pos = torch.cat([mass_charge, V[:,:,2:]],dim=-1)

        R_s = R_s.unsqueeze(2)
        R_r = R_r.unsqueeze(2)

        R_vertex_to_super_s = assignments[0][:,:,1].unsqueeze(2)
        R_vertex_to_super_r = assignments[0][:,:,0].unsqueeze(2)

        ### Embedding of particles into a super graph

        # Edge block
        upper_count = V_supers[0].size(1)
        V_super = self.get_super_features(mass_charge, V[:,:,:2], V[:,:,2:], R_vertex_to_super_r, upper_count, batch_size=batch_size)
        V_lower_pos = V_supers[0][:,:,-4:-2]
        E_to_super = self.edge_to_super_model(V_no_pos, V[:, :, :2], R_vertex_to_super_s, R_vertex_to_super_r, different_reciever=V_super[:, :, self.non_pos_indices], different_reciever_pos=V_super[:,:,-4:-2])
        
        # Sum up incomming influences to the node
        V_lower = torch.zeros((E_to_super.size(0), V_supers[0].size(1), E_to_super.size(2)), device=E_to_super.device).scatter_add_(1, R_vertex_to_super_r.expand(R_vertex_to_super_r.size(0), R_vertex_to_super_r.size(1), E_to_super.size(2)), E_to_super)
        del E_to_super
        V_lower = torch.cat([V_super[:, :, self.non_pos_indices], V_lower], dim=-1)

        embeddings = deque([[V_lower, V_lower_pos]])

        ##### Upward pass + interactions between super nodes
        for assignment, V_super, super_graph in zip(assignments[1:], V_supers[1:], super_graphs[1:]):
            R_vertex_to_super_s = assignment[:,:,1].unsqueeze(2)
            R_vertex_to_super_r = assignment[:,:,0].unsqueeze(2)

            upper_count = V_super.size(1)
            V_super = self.get_super_features(V_lower[:,:,:-(self.node_output_dim+2)], V_lower_pos, V_lower[:,:,-(self.node_output_dim+2):-self.node_output_dim], R_vertex_to_super_r, upper_count, batch_size=batch_size)

            # Edge block
            E_to_super = self.edge_to_upper_model(V_lower, V_lower_pos, R_vertex_to_super_s, R_vertex_to_super_r, different_reciever=V_super[:, :, self.non_pos_indices], different_reciever_pos=V_super[:,:,-4:-2])
            del R_vertex_to_super_s

            # Sum up incomming influences to the node
            V_lower = torch.zeros((E_to_super.size(0), V_super.size(1), E_to_super.size(2)), device=E_to_super.device).scatter_add_(1, R_vertex_to_super_r.expand(R_vertex_to_super_r.size(0), R_vertex_to_super_r.size(1), E_to_super.size(2)), E_to_super)
            del E_to_super, R_vertex_to_super_r

            # Set values for the next iteration
            V_lower_pos = V_super[:,:,-4:-2]
            V_lower = torch.cat([V_super[:, :, self.non_pos_indices], V_lower], dim=-1)
            del V_super
            embeddings.appendleft([V_lower, V_lower_pos])

        del V_lower_pos, V_lower

        V_current, V_current_pos = embeddings.popleft()
        R_s_super = super_graphs[-1][:,:,0].unsqueeze(2)
        R_r_super = super_graphs[-1][:,:,1].unsqueeze(2)
        R_super_to_vertex_s = assignments[-1][:,:,0].unsqueeze(2)
        R_super_to_vertex_r = assignments[-1][:,:,1].unsqueeze(2)

        E_current_n = self.super_edge_model(V_current, V_current_pos, R_s_super, R_r_super)
    
        # Super node block
        V_upper = self.super_node_model(V_current, E_current_n, R_r=R_r_super)
        V_upper = torch.cat([V_current[:, :, :-V_upper.size(2)], V_upper], dim=-1)
        V_upper_pos = V_current_pos

        ##### Downward pass
        for embedding, super_graph, assignment in zip(embeddings, reversed(super_graphs[:-1]), reversed(assignments[1:])):
            V_current, V_current_pos = embedding
            R_s_super = super_graph[:,:,0].unsqueeze(2)
            R_r_super = super_graph[:,:,1].unsqueeze(2)
            R_super_to_vertex_s = assignment[:,:,0].unsqueeze(2)
            R_super_to_vertex_r = assignment[:,:,1].unsqueeze(2)

            upper_influence = self.edge_from_upper_model(V_upper, V_upper_pos, R_super_to_vertex_s, R_super_to_vertex_r, different_reciever=V_current, different_reciever_pos=V_current_pos)

            E_current_n = self.super_edge_model(V_current, V_current_pos, R_s_super, R_r_super)
            E_current_n = torch.cat([E_current_n, upper_influence], dim=1)
            R_r_super = torch.cat([R_r_super, R_super_to_vertex_r], dim=1)
        
            V_upper = self.super_node_model(V_current, E_current_n, R_r=R_r_super)
            V_upper = torch.cat([V_current[:, :, :-V_upper.size(2)], V_upper], dim=-1)
            V_upper_pos = V_current_pos

        del E_current_n, R_s_super, R_r_super, embeddings, super_graphs

        R_super_to_vertex_s = assignments[0][:,:,0].unsqueeze(2)
        R_super_to_vertex_r = assignments[0][:,:,1].unsqueeze(2)

        ### Cell -> Particle edges
        E_n_s = self.edge_from_super_model(V_upper, V_upper_pos, R_super_to_vertex_s, R_super_to_vertex_r, different_reciever=V_no_pos, different_reciever_pos=V[:, :, :2])

        del assignments, V_supers, V_upper, V_upper_pos, R_super_to_vertex_s

        ### Calculating change of lower node particles
        # Edge block
        E_n = self.edge_model(V_no_pos, V[:, :, :2], R_s, R_r)
        E_n = torch.cat([E_n, E_n_s], dim=1)
        R_r = torch.cat([R_r, R_super_to_vertex_r], dim=1)

        # # Node block
        V_n = self.node_model(V_no_pos, E_n, R_r=R_r)

        # Global block
        U_n = self.global_model(V_n, E_n)
        del V_n, E_n

        # Hamiltonian
        H = self.linear(U_n)

        # Hamiltonian derivatives w.r.t inputs = dH/dq dH/dp
        partial_derivatives = torch.autograd.grad(H.sum(), V, create_graph=True)[0]

        # Return dq and dp
        return torch.cat([partial_derivatives[:,:,2:], partial_derivatives[:,:,:2] * (-1.0)], dim=2)  # dq=dH/dp, dp=-dH/dq

    def forward(self, state, R_s, R_r, assignments, V_supers, super_graphs, dt):
        # Transform inputs [m, x, y, vx, vy] to canonical coordinates [x,y,px,py]
        mass_charge = state[:,:,:-4] # if no charge = [m]; with charge = [m, c]
        momentum = state[:,:,-2:] * mass_charge[:,:,0].unsqueeze(2)
        V = torch.cat([state[:,:,-4:-2], momentum], dim=2)
        # Require grad to be able to compute partial derivatives
        if not V.requires_grad:
            V.requires_grad = True
        
        # Compute updated canonical coordinates
        if self.integrator == 'rk4':
            new_canonical_coordinates = self.rk4(dt, mass_charge, V, R_s, R_r, assignments, V_supers, super_graphs)
        elif self.integrator == 'euler':
            new_canonical_coordinates = self.euler(dt, mass_charge, V, R_s, R_r, assignments, V_supers, super_graphs)
        else:
            raise Exception
        
        # Convert back to original state format [x, y, vx, vy]
        velocity = torch.div(new_canonical_coordinates[:,:,2:], mass_charge[:,:,0].unsqueeze(2))
        new_state = torch.cat([new_canonical_coordinates[:,:,:2], velocity], dim=2)

        return new_state
        
