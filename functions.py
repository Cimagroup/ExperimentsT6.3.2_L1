import numpy as np
import matplotlib.pyplot as plt
from gudhi import RipsComplex
from gudhi import AlphaComplex
from gudhi.representations import DiagramSelector
import gudhi as gd
import plotly.graph_objects as go
from scipy.spatial import distance_matrix
import math
from scipy import sparse
import matplotlib as mpl
from scipy.stats import pearsonr
import pandas as pd
import ripser

## calculating persistence features and diagram
def ComputePersistenceDiagram(ps,moment,dimension,complex="alpha",robotsSelected="all"):
    maximumFiltration = [float(np.max(distance_matrix(X,X))) for X in ps[:,:,:2]]
    if robotsSelected == "all":
        points=ps[moment,:,:2]
    else:
        points=ps[moment,robotsSelected,:2]
    if complex not in ["rips","alpha"]:
        raise ValueError("The selected complex must be rips or alpha")
    elif complex=="alpha":
        alpha_complex = AlphaComplex(points=points) # 0ption 1: Using alpha complex
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=maximumFiltration[moment])
    else:
        rips_complex = RipsComplex(points=points,max_edge_length=maximumFiltration[moment]) # Option 2: Using Vietoris-Rips complex
        simplex_tree = rips_complex.create_simplex_tree()
    persistence_features = simplex_tree.persistence()
    persistence = simplex_tree.persistence_intervals_in_dimension(dimension)
    return persistence

## removing infinity bars or limiting this bars
def limitingDiagram(Diagram,maximumFiltr,remove=False):
    if remove is False:
        infinity_mask = np.isinf(Diagram) #Option 1:  Change infinity by a fixed value
        Diagram[infinity_mask] = maximumFiltr 
    elif remove is True:
        Diagram = DiagramSelector(use=True).fit_transform([Diagram])[0] #Option 2: Remove infinity bars
    return Diagram

## calculating entropy
def EntropyCalculationFromBarcode(persistentBarcode):
    l=[]
    for i in persistentBarcode:
        l.append(i[1]-i[0])
    L = sum(l)
    p=l/L
    entropy=-np.sum(p*np.log(p))
    return round(entropy,4)

# def relative_entropy(persistentBarcode):
#     entropy=EntropyCalculationFromBarcode(persistentBarcode) / len(persistentBarcode)
#     return round(entropy,4)

#lower stair
# function for calculate persistence diagramas using LowerStar filtration.
def calculatePersistenceDiagrams_LowerStar(t,x):
    N = x.shape[0]
    I = np.arange(N-1)
    J = np.arange(1, N)
    V = np.maximum(x[0:-1], x[1::])
    # Add vertex birth times along the diagonal of the distance matrix
    I = np.concatenate((I, np.arange(N)))
    J = np.concatenate((J, np.arange(N)))
    V = np.concatenate((V, x))
    #Create the sparse distance matrix
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    dgms = ripser.ripser(D, maxdim=3, distance_matrix=True)['dgms'] # doesn't matter the maxdim as there is only diagram for dimension 0 in the lowerstar filtration.
    return dgms

# function for obtain persistence diagrama of specific dimension.
def obtainDiagramDimension(Diagrams,dimension):
    dgm=Diagrams[dimension]
    dgm = dgm[dgm[:, 1]-dgm[:, 0] > 1e-3, :]
    return dgm

# function for remove infinity values for persistence diagram.
def limitDiagramLowerStar(Diagram,maximumFiltration):
    infinity_mask = np.isinf(Diagram)
    Diagram[infinity_mask] = maximumFiltration + 1
    return Diagram

# function for compute PE from persistence barcode
def computePersistenceEntropy(persistentBarcode):
    l=[]
    for i in persistentBarcode:
        l.append(i[1]-i[0])
    L = sum(l)
    p=l/L
    entropia=-np.sum(p*np.log(p))
    return round(entropia,4)

# plots
def gen_arrow_head_marker(angle):

    arr = np.array([[.1, .3], [.1, -.3], [1, 0], [.1, .3]])  # arrow shape
    angle
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
        ])
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))
    codes = [mpl.path.Path.MOVETO, mpl.path.Path.LINETO,mpl.path.Path.LINETO, mpl.path.Path.CLOSEPOLY]
    arrow_head_marker = mpl.path.Path(arr, codes)
    return arrow_head_marker, scale

def plotPointCloudMoment(ps,time,length,width,types,robotVision=None,vision_radius=5,field_of_view=np.pi/2,ids=False):
    maximumFiltration = [float(np.max(distance_matrix(X,X))) for X in ps[:,:,:2]]
    moment = ps[time]
    x=moment[:,0]
    y=moment[:,1]
    angle=moment[:,2]

    scatter_handles = [] 
    
    # plt.figure(figsize=(8, 8))
    for (a,b,c,d) in zip(x,y,angle,types):
        if d in ["thymio","wheelchair"]:
            color = "green"
            label = "wheelchair"
        elif d == "human":
            color="yellow"
            label="person"
        else:
            color = "pink"
            label="other"
        marker, scale = gen_arrow_head_marker(c)
        markersize = 25
        plt.scatter(a,b,marker=marker,c=color, s=(markersize*scale)**1.5,label = label)

        if label not in [entry.get_label() for entry in scatter_handles]:
            legend_marker = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=color, markersize=10, linestyle='None', label=label)
            scatter_handles.append(legend_marker)
    if ids is True:
        for i in range(len(x)):
            plt.text(x[i], y[i]+0.1, str(i), fontsize=7, ha='center', va='bottom')
    
    if robotVision is not None: 
        xrobot=x[robotVision]
        yrobot=y[robotVision]
        orientation=angle[robotVision]
        arc_points = [[xrobot, yrobot]]  
        
        num_points = 50  
        for i in range(num_points + 1):
            angles = orientation + field_of_view / 2 - (i / num_points) * field_of_view
            arc_points.append([xrobot + vision_radius * np.cos(angles), yrobot + vision_radius * np.sin(angles)])
        arc_points.append([xrobot, yrobot])  
        arc_points = np.array(arc_points)
        # plt.plot(arc_points[:, 0], arc_points[:, 1], 'b-', alpha=0.3) 
        plt.fill(arc_points[:, 0], arc_points[:, 1], color='blue', alpha=0.1)
    if length==width:
        plt.xlim(-length/1.5, length/1.5)
        plt.ylim(-width/1.5, width/1.5)
    else:
        plt.xlim([-length*0.2,length*1.2])
        plt.ylim([-width*0.2,width*1.2])
        plt.axhline(y=0, color='black')
        plt.axhline(y=width, color='black')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Point cloud in time: {time}')
    plt.legend(handles=scatter_handles)
    
def plotPointCloud2Moments(ps,time1,time2,length,width):
    maximumFiltration = [float(np.max(distance_matrix(X,X))) for X in ps[:,:,:2]]
    moment1 = ps[time1]
    moment2 = ps[time2]
    x1=moment1[:,0]
    y1=moment1[:,1]
    angle1=moment1[:,2]

    x2=moment2[:,0]
    y2=moment2[:,1]
    angle2=moment2[:,2]

    maxX=max(max(x1),max(x2)) + 1
    maxY=max(max(y1),max(y2)) + 1
    minX=min(min(x1),min(x2)) - 1
    minY=min(min(y1),min(y2)) - 1

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
    for (a,b,c) in zip(x1,y1,angle1):
        marker, scale = gen_arrow_head_marker(c)
        markersize = 25
        axs[0].scatter(a,b,marker=marker,c="blue", s=(markersize*scale)**1.5, label=f"Initial time: {time1}")
    axs[0].set_title(f'Initial time: {time1}')  
    if length==width:
        axs[0].set_xlim(-length/1.5, length/1.5)
        axs[0].set_ylim(-width/1.5, width/1.5)
    else:
        axs[0].set_xlim([-length*0.2,length*1.2])
        axs[0].set_ylim([-width*0.2,width*1.2])
        axs[0].axhline(y=0, color='black')
        axs[0].axhline(y=width, color='black')
    for i in range(len(x1)):
        axs[0].text(x1[i], y1[i]+0.1, str(i), fontsize=7, ha='center', va='bottom')

    for (a,b,c) in zip(x2,y2,angle2):
        marker, scale = gen_arrow_head_marker(c)
        markersize = 25
        axs[1].scatter(a,b,marker=marker,c="blue", s=(markersize*scale)**1.5, label=f"End time: {time2}")
    axs[1].set_title(f'End time: {time2}') 
    if length==width:
        axs[1].set_xlim(-length/1.5, length/1.5)
        axs[1].set_ylim(-width/1.5, width/1.5)
    else:
        axs[1].set_xlim([-length*0.2,length*1.2])
        axs[1].set_ylim([-width*0.2,width*1.2])
        axs[1].axhline(y=0, color='black')
        axs[1].axhline(y=width, color='black')
    for i in range(len(x2)):
        axs[1].text(x2[i], y2[i]+0.1, str(i), fontsize=7, ha='center', va='bottom')
    for (a,b,c) in zip(x1,y1,angle1):
        marker, scale = gen_arrow_head_marker(c)
        markersize = 25
        axs[2].scatter(a,b,marker=marker,c="blue", s=10) #, label=f"Initial time: {time1}")
    # Etiqueta para los puntos azules
    axs[2].scatter([], [], c="blue", label=f"Initial time: {time1}")
    axs[2].scatter([], [], c="red", label=f"End time: {time2}")
    for (a,b,c) in zip(x2,y2,angle2):
        marker, scale = gen_arrow_head_marker(c)
        markersize = 25
        axs[2].scatter(a,b,marker=marker,c="red", s=10) #, label=f"End time: {time2}")
    for i in range(len(x1)):
         axs[2].plot([x1[i], x2[i]], [y1[i], y2[i]], color='gray', linestyle='--',linewidth=0.5,alpha=0.5)
    if length==width:
        axs[1].set_xlim(-length/1.5, length/1.5)
        axs[1].set_ylim(-width/1.5, width/1.5)
    else:
        axs[2].set_xlim([-length*0.2,length*1.2])
        axs[2].set_ylim([-width*0.2,width*1.2])
        axs[2].axhline(y=0, color='black')
        axs[2].axhline(y=width, color='black')
    axs[2].legend()
    axs[2].set_title(f'Movements betweent time {time1} and {time2}') 
    plt.tight_layout()

def plotPersistenceDiagram(ps,moment,dimension):
    maximumFiltration = [float(np.max(distance_matrix(X,X))) for X in ps[:,:,:2]]
    persistence = ComputePersistenceDiagram(ps,moment,dimension,"rips")
    gd.plot_persistence_diagram(persistence)
    plt.title(f"Persistent diagram for time {moment}")

def plotPersistenceBarcode(ps,moment,dimension):
    maximumFiltration = [float(np.max(distance_matrix(X,X))) for X in ps[:,:,:2]]
    persistence = ComputePersistenceDiagram(ps,moment,dimension,"rips")
    persistenciaL=limitingDiagram(persistence,maximumFiltration[moment])
    entropy=EntropyCalculationFromBarcode(persistenciaL)
    gd.plot_persistence_barcode(persistenciaL)
    plt.title(f"Persistent barcode for time {moment}. Entropy: {entropy}")
    

def plotEntropyTimeSerie(entropy):
    plt.plot(entropy)
    plt.xlabel('Time')
    plt.ylabel('Entropy')
    plt.title(f'Persistent entropy time series')
    plt.grid(True)

def plotEntropyTimeSerieInteractive(entropy):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(0,len(entropy)), 
            y=entropy,
            mode='lines+markers',
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        xaxis_title='Time',
        yaxis_title='Entropy',
        title=f'Topological entropy time series of persistent diagram'
    )
    fig.show()

#robots in field of vision
def calculate_robots_in_field_vision(ps,time, robot,vision_radius=5,field_of_view=np.pi/2,printing=False):
    maximumFiltration = [float(np.max(distance_matrix(X,X))) for X in ps[:,:,:2]]
    robots_in_field_of_vision = []
    moment = ps[time]
    x=moment[:,0]
    y=moment[:,1]
    angle=moment[:,2]
    xTarget = x[robot]
    yTarget = y[robot]
    angleTarget = angle[robot]
    angle_start = angleTarget - field_of_view / 2
    angle_end = angleTarget + field_of_view / 2
    for i in range(len(x)):
        if i == robot:
            continue
        
        robot_x, robot_y = x[i], y[i]
        distance = calculate_distance(xTarget,yTarget,robot_x,robot_y)
        if distance > vision_radius:
            continue
        
        angle_robot = np.arctan2(robot_y - yTarget, robot_x - xTarget)
        angle_relative = (angle_robot - angleTarget + 2 * np.pi) % (2 * np.pi)
        angle_start_relative = (angle_start - angleTarget + 2 * np.pi) % (2 * np.pi)
        angle_end_relative = (angle_end - angleTarget + 2 * np.pi) % (2 * np.pi)
        if angle_start_relative < angle_end_relative:
            if angle_start_relative <= angle_relative <= angle_end_relative:
                robots_in_field_of_vision.append(i)
        else:  
            if angle_relative >= angle_start_relative or angle_relative <= angle_end_relative:
                robots_in_field_of_vision.append(i)
    if printing is True:
        print(f"Time {time}. Robots in the robot's {robot} field of vision:", robots_in_field_of_vision)
    return robots_in_field_of_vision


# distances and angles
def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculate_angle(x, y, orientation, x2, y2):
    angle_to_point = np.arctan2(y2 - y, x2 - x)
    relative_angle = angle_to_point - orientation
    return relative_angle

def transform_angle(angle):
    while angle < 0:
        angle += 360
    if angle <= 180:
        finalAngle = angle
    else:
        finalAngle = 360 - angle
    return finalAngle

def normangle(angle):
    result = np.mod(angle, 2 * np.pi)
    result[result > np.pi] -= 2 * np.pi
    return result

#count deadlock and extract data from experiment
def count_deadlocks(deadlock_time, final_time): #deadlock_time is a run.deadlocks item
    is_deadlocked = np.logical_and(deadlock_time > 0, deadlock_time < (final_time - 2.0))
    return sum(is_deadlocked)

def count_deadlocks(deadlock_time, final_time):
    is_deadlocked = np.logical_and(deadlock_time > 0, deadlock_time < (final_time - 2.0))
    return sum(is_deadlocked)

def extract_data(experiment, behavior):
    collisions = []
    deadlocks = []
    efficacy = []
    etas=[]
    taus=[]
    sms = []
    bas = []
    seeds = []
    if behavior == "HL":
        for i, run in experiment.runs.items():
            world = run.world
            sm = np.unique([agent.behavior.safety_margin for agent in world.agents])
            tau = np.unique([agent.behavior.tau for agent in world.agents])
            eta = np.unique([agent.behavior.eta for agent in world.agents])
            ba = np.unique([agent.behavior.barrier_angle for agent in world.agents])
            bas += list(ba)
            taus += list(tau)
            etas += list(eta)
            sms += list(sm)
            seeds.append(run.seed)
            final_time = run.world.time
            deadlocks.append(count_deadlocks(run.deadlocks, final_time))
            collisions.append(len(run.collisions))
            efficacy.append(run.efficacy.mean())
    
        df = pd.DataFrame({
            'seeds': seeds,
            'safety_margin': sms,
            'eta': etas,
            'tau': taus,
            'deadlocks': deadlocks,
            'collisions': collisions,
            'barrier_angle': bas,
            'efficacy': efficacy})
        df['safe'] = (df.collisions == 0).astype(int)
        df['fluid'] = (df.deadlocks == 0).astype(int)
        df['ok'] = ((df.deadlocks == 0) & (df.collisions == 0)).astype(int)
    elif behavior == "ORCA":
        for i, run in experiment.runs.items():
            world = run.world
            sm = np.unique([agent.behavior.safety_margin for agent in world.agents])
            sms += list(sm)
            seeds.append(run.seed)
            final_time = run.world.time
            deadlocks.append(count_deadlocks(run.deadlocks, final_time))
            collisions.append(len(run.collisions))
            efficacy.append(run.efficacy.mean())
    
        df = pd.DataFrame({
            'seeds': seeds,
            'safety_margin': sms,
            'deadlocks': deadlocks,
            'collisions': collisions,
            'efficacy': efficacy})
        df['safe'] = (df.collisions == 0).astype(int) # 1 si no colisiones, 0 caso contrario
        df['fluid'] = (df.deadlocks == 0).astype(int) # 1 si no deadlocks, 0 caso contrario
        df['ok'] = ((df.deadlocks == 0) & (df.collisions == 0)).astype(int)
        
    return df
    
## correlation
def show_correlation(feature1,feature2):
    corr_spearman, p_value = pearsonr(feature1, feature2)
    print(f"Pearson's correlation coefficient: {corr_spearman}")
    print(f"P-value: {p_value}")

    if p_value < 0.05:
        print("\033[92mThe correlation is statistically significant.\033[0m")
    else:
        print("There is insufficient evidence to reject the null hypothesis of no correlation.")

