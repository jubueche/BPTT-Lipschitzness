import numpy as np
import concurrent.futures
from collections import OrderedDict
import plotly.graph_objects as go
from architectures import ecg_lsnn
from datajuicer import run, get
import loss_jax
from experiment_utils import get_loader
import jax

def to_array(dic):
    dic = OrderedDict(dic)
    return np.hstack([dic[key].flatten() for key in dic]), OrderedDict((key,dic[key].shape) for key in dic)

def to_dict(array, shapes):
    lengths = [np.prod(shapes[key]) for key in shapes]
    start = [sum(lengths[0:i]) for i in range(len(lengths))] + [sum(lengths)]
    return {key:array[start[i]:start[i+1]].reshape(shapes[key]) for i,key in enumerate(shapes)}

class Visualizer:

    def __init__(self, theta_center, dir_tuple=None, ppsu=64, extent=2, z_scale=0.5):
        self.z_scale=z_scale
        
        if type(extent) is int:
            extent = (extent, extent)
        else:
            extent = extent
        self.z_loss = None
        self.color_loss = None
        self.paths = []

        dir1 = {}
        dir2 = {}
        fix_dir=False
        self.dict_mode = True
        if not type(theta_center) is dict:
            self.dict_mode = False
            if theta_center.shape == (2,):
                fix_dir = True
            theta_center = {'key':theta_center.astype(float)}
        else:
            theta_center = {key:theta_center[key].astype(float) for key in theta_center}
        

        length = int(np.sqrt(ppsu)*extent[0]) + 1
        width = int(np.sqrt(ppsu)*extent[1]) + 1

        if fix_dir:
            dir1 = {'key':np.array([1.0,0.0])}
            dir2 = {'key':np.array([0.0,1.0])}
        else:
            if dir_tuple == None:
                for key in theta_center:
                    dir1[key] = np.random.normal(loc=0.0, scale=0.3*np.abs(theta_center[key]), size=theta_center[key].shape) 
                    dir2[key] = np.random.normal(loc=0.0, scale=0.3*np.abs(theta_center[key]), size=theta_center[key].shape)
            else:
                dir1,dir2 = dir_tuple
        
        self.x_coords = [(i/ np.sqrt(ppsu) - 0.5* extent[0]) for i in range(length)]
        self.y_coords = [(i/ np.sqrt(ppsu) - 0.5* extent[1]) for i in range(width)]
        
        theta_center_array, theta_shapes = to_array(theta_center)
        dir1_arr, _ = to_array(dir1)
        dir2_arr, _ = to_array(dir2)

        self.thetas = []
        for y in self.y_coords:
            row = []
            for x in self.x_coords:
                row.append(to_dict(theta_center_array + x*dir1_arr + y*dir2_arr, theta_shapes))
            self.thetas.append(row)

        arr1, sh1 = to_array(dir1)
        arr2, sh2 = to_array(dir2)
        M = np.vstack([arr1, arr2]).T
        self.inverse = np.linalg.pinv(M)
        self.theta_center = theta_center
        self.extent = extent

    def get_coords(self,theta):
        return np.matmul(self.inverse, to_array(theta)[0] - to_array(self.theta_center)[0])

    def add_color(self, loss):
        if self.dict_mode:
            self.color_loss = loss
        else:
            self.color_loss = lambda d: loss(d["key"])
    
    def add_z(self, loss):
        if self.dict_mode:
            self.z_loss = loss
        else:
            self.z_loss = lambda d: loss(d["key"])
    
    def add_path(self, path):
        if self.dict_mode:
            self.paths.append(path)
        else:
            self.paths.append([{'key':np.array(theta)} for theta in path])
    
    # def add_square(self, center, side_length, resolution=10):
    #     radius = side_length/2.0

    #     squarex = [center[0] - radius] * resolution 
    #     squarex +=  [(i/resolution) * 2 * radius + center[0] - radius for i in range(resolution+1)]
    #     squarex += [center[0] + radius] * resolution
    #     squarex += reversed([(i/resolution) * 2 * radius + center[0] - radius for i in range(resolution+1)])
    #     squarey = [(i/resolution) * 2 * radius + center[1] - radius for i in range(resolution+1)]
    #     squarey += [center[1] + radius] * resolution
    #     squarey += reversed([(i/resolution) * 2 * radius + center[1] -radius for i in range(resolution+1)])
    #     squarey += [center[1] - radius] * resolution
    #     self.add_path(zip(squarex, squarey))

    def make_landscape(self, loss, n_threads):
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [[executor.submit(loss, theta) for theta in row]for row in self.thetas]

        return [[f.result() for f in row] for row in futures]
    
    def visualize(self, n_threads=10):
        if self.z_loss is None:
            raise Exception("call add_z() before calling visualize")
        
        z_landscape = self.make_landscape(self.z_loss, n_threads)
        
        print(np.array(z_landscape).shape)

        if self.color_loss is None:
            surface = go.Surface(x=self.x_coords,y=self.y_coords,z=z_landscape)
        else:
            c_landscape = self.make_landscape(self.color_loss, n_threads)
            surface = go.Surface(x=self.x_coords,y=self.y_coords,z=z_landscape,surfacecolor=c_landscape)
            
        
        def make_scatter(path):
            coord_path = np.array([self.get_coords(theta) for theta in path])
            scatx = coord_path[:,0]
            scaty = coord_path[:,1]
            scatz = [self.z_loss(theta) for theta in path]
            return go.Scatter3d(x=scatx, y=scaty, z = scatz, mode="lines+markers", marker= {"size":7}, line={"width":15})
        
        scatters = [make_scatter(path) for path in self.paths]

        if self.extent[0] > self.extent[1]:
            aspectratio = {"x":1,"y":self.extent[1]/self.extent[0], "z":self.z_scale}
        else:
            aspectratio= {"x":self.extent[0]/self.extent[1],"y":1, "z":self.z_scale}
        
        range0 = [-0.5*self.extent[0], 0.5*self.extent[0]]
        range1 = [-0.5*self.extent[1], 0.5*self.extent[1]]
        layout = go.Layout(scene={"aspectmode":"manual", "aspectratio":aspectratio}, xaxis=dict(range=range0), yaxis=dict(range=range1))
        fig  = go.FigureWidget(data=[surface] + scatters, layout = layout)
        fig.show()

if __name__ == "__main__":
    

    v = Visualizer(theta_start, ppsu=64, extent=4)

    v.add_z(cce_loss)

    # v.add_color(robust_loss)

    def step(theta, treat_as_constant, beta_robustness):
        FLAGS = ecg_lsnn.get_flags({"treat_as_constant":treat_as_constant, "beta_robustness":beta_robustness,"mode":"direct"})
        grads = loss_jax.compute_gradients(loader.X_test, loader.y_test, theta, model["network"], FLAGS, model["network"]._rng_key,0)
        theta_new = {}
        for key in theta:
            theta_new[key]= theta[key] - 0.001 * grads[key]
        model["network"]._rng_key, _ = jax.random.split(model["network"]._rng_key)
        return theta_new

    def make_path(start, length, treat_as_constant, beta_robustness):
        path = [start]
        for _ in range(length):
            path += [step(path[-1], treat_as_constant, beta_robustness)]
        return path

    # v.add_path(make_path(theta_start, 2, False, 0.125))

    # v.add_path(make_path(theta_start, 2, True, 0.125))

    v.visualize()
