import matplotlib.image as mpimg
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import skimage
from skimage import measure
import trimesh


# Camera Calibration for Al's image[1..12].pgm   
calib = np.array([
    [-78.8596, -178.763, -127.597, 300, -230.924, 0, -33.6163, 300,
     -0.525731, 0, -0.85065, 2],
    [0, -221.578, 73.2053, 300, -178.763, -127.597, -78.8596, 300,
     0, -0.85065, -0.525731, 2],
    [ 78.8596, -178.763, -127.597, 300, -73.2053, 0, -221.578, 300,
     0.525731, 0, -0.85065, 2],
    [0, 33.6163, -230.924, 300, -178.763, 127.597, -78.8596, 300,
     0, 0.85065, -0.525731, 2],
    [-78.8596, -178.763, 127.597, 300, 73.2053, 0, 221.578, 300,
     -0.525731, 0, 0.85065, 2],
    [78.8596, -178.763, 127.597, 300, 230.924, 0, 33.6163, 300,
     0.525731, 0, 0.85065, 2],
    [0, -221.578, -73.2053, 300, 178.763, -127.597, 78.8596, 300,
     0, -0.85065, 0.525731, 2],
    [0, 33.6163, 230.924, 300, 178.763, 127.597, 78.8596, 300,
     0, 0.85065, 0.525731, 2],
    [-33.6163, -230.924, 0, 300, -127.597, -78.8596, 178.763, 300,
     -0.85065, -0.525731, 0, 2],
    [-221.578, -73.2053, 0, 300, -127.597, 78.8596, 178.763, 300,
     -0.85065, 0.525731, 0, 2],
    [221.578, -73.2053, 0, 300, 127.597, 78.8596, -178.763, 300,
     0.85065, 0.525731, 0, 2],
    [33.6163, -230.924, 0, 300, 127.597, -78.8596, -178.763, 300,
     0.85065, -0.525731, 0, 2]
])


# Build 3D grids
# 3D Grids are of size: resolution x resolution x resolution/2
resolution = 30

step = 2 / resolution

# Voxel coordinates
X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]

# Voxel occupancy
occupancy = np.ndarray((resolution, resolution, resolution // 2), dtype=int)

print(occupancy)
print(occupancy.shape)

# Voxels are initially occupied then carved with silhouette information
occupancy.fill(1)
 

# ---------- MAIN ----------
if __name__ == "__main__":
    #print(X.shape)
    i = 1
    # read the input silhouettes
    for g in range(1):
        myFile = "image{0}.pgm".format(g)
        print(myFile)
        img = mpimg.imread(myFile)
        if img.dtype == np.float32:  # if not integer
            img = (img * 255).astype(np.uint8)
        
        projec = calib[g].reshape(3,4)
        #print(img.shape)

        for i in range(len(X)):
            for j in range(len(Y)): 
                for k in range(len(Z)//2):
                    p = projec.dot(np.array([X[i][j][k], Y[i][j][k], Z[i][j][k], 1]).reshape(4,-1))
                    #print(p)
                    
                    
                    if((int(p[0]/p[2])<300 and int(p[1]/p[2])<300) and img[int(p[0]/p[2]),int(p[1]/p[2])]==0):
                        occupancy[i][j][k] = 0  

    # Voxel visualization
    print(occupancy)
    print(occupancy.shape)

    # Use the marching cubes algorithm
    verts, faces, normals, values = measure.marching_cubes(occupancy, 0.25)

    # Export in a standard file format
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
    surf_mesh.export('alvoxels.off')
 
