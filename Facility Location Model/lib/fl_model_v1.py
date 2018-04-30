# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For vincenty:
from geopy.distance import great_circle


# Classes
class demand_class:
    """Demand points such as affected cities or villages"""

    def __init__(self, name, x, y, demand):
        self.name = "DP{}".format(name)
        self.type = "demand"
        self.x = x
        self.y = y
        self.location = (self.x, self.y)
        self.demand = demand


class supply_class:
    """Supply points such as airports or seaports"""

    def __init__(self, name, x, y, supply):
        self.name = "SP{}".format(name)
        self.type = "supply"
        self.x = x
        self.y = y
        self.location = (self.x, self.y)
        self.supply = supply
        self.operational = 1


class facility_class:
    """(Temporary) Distribution centers to distribute supply to demand"""

    def __init__(self, name, x, y):
        self.name = "FL{}".format(name)
        self.type = "facility"
        self.x = x
        self.y = y
        self.location = (self.x, self.y)
        self.operational = 0


# Instantiate case functions
# Instantiate model


# Create airport as a supply point
def create_supply_points(sup_xcors, sup_ycors, supply_at_sp=100):

    #TODO get a list with specific supply values for each point
    supply_at_sps = [supply_at_sp for i in range(len(sup_xcors))]

    supply_points = []
    for i in range(len(sup_xcors)):
        supply_points.append(
            supply_class(
                name=i,
                x=sup_xcors[i],
                y=sup_ycors[i],
                supply=supply_at_sps[i]))
    return supply_points


# Create demand points
def create_demand_points(dp_xcors, dp_ycors, demand_at_dp=10):

    demand_at_dps = [demand_at_dp for i in range(len(dp_xcors))]

    demand_points = []
    for i in range(len(dp_xcors)):
        demand_points.append(
            demand_class(
                name=i, x=dp_xcors[i], y=dp_ycors[i], demand=demand_at_dps[i]))
    return demand_points


# # Create possible facility locations
def create_facility_locations(fl_xcors, fl_ycors):

    facility_locations = []
    for i in range(len(fl_xcors)):
        facility_locations.append(
            facility_class(name=i, x=fl_xcors[i], y=fl_ycors[i]))
    return facility_locations


# Create matrix with all distances
def create_distance_matrix(all_nodes, distance_to_self=999):
    """
    Creates a matrix with distances between all nodes
    Input: list of all nodes (objects)
    Output: Matrix with distances from [i,j]
    i,j = from, to
    Note: Matrix is symmetric: distances[i,j] = distances[j,i]
    Note: Distance to self ([i,i]) is 100 times larger than largest distance in matrix
    """
    distances = np.zeros([len(all_nodes), len(all_nodes)])

    def calculate_distance(x1, y1, x2, y2, method="euclidean"): # TODO: include kwarg in model that chooses method
        
        """
        Lat = Y Long = X
        (lat, lon) is coordinate notation used by geopy
        method: euclidean or great_circle
        great_circle returns length in meters
        """
        if method == "euclidean":
            dx = x1 - x2
            dy = y1 - y2
            return (dx**2 + dy**2)**0.5
        
        if method == "great_circle":
            # Geopy uses location order: Lat, Lon
            return great_circle((y1,x1),(y2,x2)).km


    # calculate distance matrix
    for i in range(len(all_nodes)):  #For every row
        for j in range(len(all_nodes)):  #For every column
            distances[i, j] = calculate_distance(
                all_nodes[i].x, all_nodes[i].y, all_nodes[j].x, all_nodes[j].y)

    # set distance to self to big distance
    for i in range(len(distances)):
        distances[i, i] = distance_to_self

    return distances


def create_disrupted_road_matrix(distances, multipliers, nodes):
    """
    input:XX
    nodes can be both a set of FLs or DP
    Output: XX
    Validated
    """
    def mirror_matrix (matrix, distance_to_self=999):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError ('Matrix is not well shaped. Should have dimensions n,n, where n=n') 
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                if i == j: # put large numbers on the diagonals
                    matrix[i,j] = distance_to_self
                matrix[j,i] = (matrix[i,j])
        return matrix
    
    matrix = distances.copy() # don't change distances
    
    node_id_s = [n.id for n in nodes]
    
    for i, node_id in enumerate(node_id_s):
        matrix[:,node_id] = matrix[:,node_id] * multipliers[i]

    
    matrix = mirror_matrix(matrix)
    
    return matrix

# allocate each demand point to either a supply point or a facility location
def create_allocation_matrix(supply_points, demand_points, facility_locations,
                             distances, FL_range):
    """
    Returns an allocation matrix for [n,h]
    if [n,h] is 1: node n is allocated to supply hub h
    The full size of the matrix is n*n
    indexes are based on node.id
    
    Assumptions:
    All allocations are made based on the minimum distance. NOT disrupted distance, because unknown.
    Supply points are not allocated to other nodes, since they are supplied by upstream logistics, which is out of scope    
    """

    # list of all IDs for operational supply points and facility locations
    operational_hubs_id = [
        a.id for a in supply_points + facility_locations if a.operational == 1
    ]

    all_nodes = supply_points + facility_locations + demand_points
    #allocation matrix of all_nodes x all_nodes
    allocation_matrix = np.zeros([len(all_nodes), len(all_nodes)])

    # for each operational facility location:
    # we skip supply points, because we don't consider upstream logistics
    for fl in [fl for fl in facility_locations if fl.operational == 1]:
        # choose closest supply point (if multiple)           # XX is still necessary? not used here: (distance to iself is made very large while creating distance matrix)
        closest_i = np.argmin([
            distances[sp_id, fl.id]
            for sp_id in [sp.id for sp in supply_points]
        ])
        # allocate me to that supply point
        allocation_matrix[fl.id, operational_hubs_id[closest_i]] = 1

    # For each demand point
    for dp in demand_points:
        # Assumption: Each demand point gets 1 location allocated. If 2 locations have same distance, choose first
        # check which supply point or facility location is closest
        # closest = the index of the minimum distance of the list of distances between this demand point and all supply points
        closest_d = np.min(
            [distances[hub_id, dp.id] for hub_id in operational_hubs_id])
        closest_i = np.argmin(
            [distances[hub_id, dp.id] for hub_id in operational_hubs_id])

        # Set allocation 1 if j is closest to this demand point (or i)
        #i: demand point index, j: supply point index
        if closest_d < FL_range:
            allocation_matrix[dp.id, operational_hubs_id[closest_i]] = 1

    return allocation_matrix


################### Objectives ###################

def calc_costs(facility_locations, demand_points, unit_opening_costs, unit_transport_cost,
               distances, allocation_matrix):
    # calc opening costs of facilities
    # assumption: supply point, i.e. airport, is already opened
    nr_opened_fl = sum([fl.operational for fl in facility_locations])
    total_opening_costs = nr_opened_fl * unit_opening_costs

    # Calc transportation costs
    total_distance = (allocation_matrix * distances).sum()
    
    #TODO: Validate
    #Create array of all DPs and whether they are allocated to a SP or FL
    allocated_DPs = allocation_matrix[[dp.id for dp in demand_points],:].sum(axis=1)
    #Create array of demand for each DP
    demand_DPs = [dp.demand for dp in demand_points]
    # Multiply arrays to get supply for each DP
    supply = covered_demand = np.multiply(allocated_DPs,demand_DPs)
    
    transportation_costs = total_distance * unit_transport_cost * sum(supply)
    

    return total_opening_costs + transportation_costs

#     Minimise total uncovered demand
def calc_tot_uncov_demand(allocation_matrix, demand_points):
    """
    input: allocation matrix and list of demand points
    output: total uncovered demand (float)
    
    Validated
    """
    #Create array of all DPs and whether they are allocated to a SP or FL
    allocated_DPs = allocation_matrix[[dp.id for dp in demand_points],:].sum(axis=1)
    #Create array of demand for each DP
    demand_DPs = [dp.demand for dp in demand_points]
    # Multiply arrays to get supply for each DP
    covered_demand = np.multiply(allocated_DPs,demand_DPs)
    total_demand = sum(demand_DPs)
    return total_demand - sum(covered_demand)

#     Minimise uncovered demand points
def calc_uncov_DPs(allocation_matrix,demand_points):
    """
    Input: allocation matrix and list of demand points
    Output: number of demand points that has not been allocated
    
    Validated
    """ 
    # create array of all DPs and whether they are allocated to a SP or FL
    allocated_DPs = allocation_matrix[[dp.id for dp in demand_points],:].sum(axis=1)
    n_uncov_DPs = len(demand_points) - sum(allocated_DPs)
    return int(n_uncov_DPs)

#     Minimise Total Distribution Times
def calc_tot_distr_time(allocation_matrix, road_distances, speed):
    """
    road_distances = disrupted roads
    """
    total_distances = allocation_matrix * road_distances
    tot_distr_time = total_distances.sum() / speed # distance / speed = time
    return tot_distr_time



# Functions for plotting
def plotting_create_allocation_lines(all_nodes, allocation_matrix):

    allocation_lines = np.zeros([len(all_nodes), 4])
    # creates matrix x1,y1,x2,y2
    for i, line in enumerate(zip(allocation_matrix)):
        if allocation_matrix[i].sum() == 0:
            continue
        allocation_lines[i, 0:2] = list(all_nodes[i].location)
        # Assumption: Only 1 location allocated # if changing: loop over list comprehension: [more than 0 allocated]
        allocation_lines[i, 2:4] = list((all_nodes)[np.argmax(line)].location)

    return allocation_lines


def plotting_plot_map(demand_points,
                      facility_locations,
                      supply_points,
                      allocation_lines=False):
    
    xmax = max([n.x for n in demand_points + facility_locations + supply_points]) + 1
    xmin = min([n.x for n in demand_points + facility_locations + supply_points]) - 1
    ymax = max([n.y for n in demand_points + facility_locations + supply_points]) + 1
    ymin = min([n.y for n in demand_points + facility_locations + supply_points]) - 1
    
    plt.axis([xmin, xmax, ymin, ymax])

    for x in demand_points:
        plt.scatter(x.x, x.y, c="green", marker="<")

    for x in facility_locations:
        if x.operational == 0:
            plt.scatter(x.x, x.y, c="red", marker="x", s=10)
        if x.operational == 1:
            plt.scatter(x.x, x.y, c="blue", marker="x", s=50)
    for x in supply_points:
        plt.scatter(x.x, x.y, c="blue", marker=">")

    # plot allocation lines
    if allocation_lines is not False:
        for line in allocation_lines:
            plt.plot(line[[0, 2]], line[[1, 3]], c="green")



def FL_model(unit_opening_costs,
             unit_transport_cost, # Cost for transporting one unit of supplies
             graphical_representation=False,
             FL_range=2,
             lorry_speed=60, # km/h. Speed is Average speed. Constant, because roads are individually disrupted.
             **kwargs):
    """
    Inputs:
    Returns: Objectives, Constraints
    """

    #unpack kwargs
    keys = sorted(kwargs.keys())
    sp_xcors = [kwargs[x] for x in [k for k in keys if k[:3] == 'SPX']]
    sp_ycors = [kwargs[x] for x in [k for k in keys if k[:3] == 'SPY']]
    dp_xcors = [kwargs[x] for x in [k for k in keys if k[:3] == 'DPX']]
    dp_ycors = [kwargs[x] for x in [k for k in keys if k[:3] == 'DPY']]
    fl_xcors = [kwargs[x] for x in [k for k in keys if k[:3] == 'FLX']]
    fl_ycors = [kwargs[x] for x in [k for k in keys if k[:3] == 'FLY']]

    fl_operational = [kwargs[x] for x in [k for k in keys if k[:3] == 'FLO']]

    disruption_DPs = [kwargs[x] for x in [k for k in keys if k[:5] == 'DSRDP']]
    disruption_FLs = [kwargs[x] for x in [k for k in keys if k[:5] == 'DSRFL']]
    
    #TODO Assign demand to demand points
    dp_demand = [kwargs[x] for x in [k for k in keys if k[:3] == 'DPD']]

    # set up model
    supply_points = create_supply_points(sp_xcors, sp_ycors)
    demand_points = create_demand_points(dp_xcors, dp_ycors)
    facility_locations = create_facility_locations(fl_xcors, fl_ycors)

    # Organise all nodes and create distance matrix
    all_nodes = supply_points + facility_locations + demand_points
    # Give all nodes in model an identifier corresponding to position in matrix
    for i in range(len(all_nodes)):
        all_nodes[i].id = i

    # Check if things are right
    if len(facility_locations) != len(fl_operational):
        print("Length FL arrays not equal, FL_model:", len(facility_locations),
              len(fl_operational))

    # Set operational FLs from levers
    for i, fl in enumerate(facility_locations):
        fl.operational = fl_operational[i]

    distances = create_distance_matrix(all_nodes)
    
    # calculate road disruptions for FLs
    disr_roads = create_disrupted_road_matrix(distances, disruption_FLs, facility_locations)
    # calculate road disruptions for DPs
    disr_roads = create_disrupted_road_matrix(distances, disruption_DPs, demand_points)
    
    
    allocation_matrix = create_allocation_matrix(
        supply_points, demand_points, facility_locations, distances, FL_range)
    # Assumption: allocation based on euclidean distance (because roads and road conditions are unknown)

    # determine objectives

    #     Minimise total costs
    #         Total Opening costs
    #         Total Transportation costs
    total_costs = calc_costs(facility_locations, demand_points, unit_opening_costs,
                             unit_transport_cost, distances, allocation_matrix)

    #     Minimise total uncovered demand
    total_uncovered_demand = calc_tot_uncov_demand(allocation_matrix,
                                                   demand_points)

    #     Minimise uncovered demand points
    nr_uncovered_DPs = calc_uncov_DPs(allocation_matrix, demand_points)

    #     Minimise Total Distribution Times
    total_distr_time = calc_tot_distr_time(allocation_matrix, disr_roads, lorry_speed)

    # give a graphical representation of instantiation and allocation
    if graphical_representation == True:
        allocation_lines = plotting_create_allocation_lines(
            all_nodes, allocation_matrix)

        plotting_plot_map(demand_points, facility_locations, supply_points,
                          allocation_lines)
        plt.show()

    return total_costs, nr_uncovered_DPs, total_uncovered_demand, total_distr_time, sum(
        [fl.operational for fl in facility_locations])

