import math as m
import matplotlib.image as mpimg
import numpy as np
import skimage
from skimage import measure
import torch
from torch import nn
import trimesh

import random
import matplotlib.pyplot as plt

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

# Training
MAX_EPOCH = 40
BATCH_SIZE = 100

# Build 3D grids
# 3D Grids are of size resolution x resolution x resolution/2
resolution = 80
step = 2 / resolution
offset = 12
# Voxel coordinates
X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]

# Voxel occupancy
occupancy = np.ndarray((resolution, resolution, resolution // 2), dtype=int)

# Voxels are initially occupied then carved with silhouette information
occupancy.fill(1)

# drawing the accuracy plot
accuracies = []
epochs = []

#print(" old occupancy",occupancy)
print("old occupancy.shape",occupancy.shape)


# MLP class
class MLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 60),
            nn.Tanh(),
            nn.Linear(60, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            #nn.ReLU(),
            #nn.Linear(180, 60),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        """ Forward pass """
        return self.layers(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# MLP Training
def nif_train(data_in, data_out, batch_size):
    # GPU or not GPU
    

    # Initialize the MLP
    mlp = MLP()
    mlp = mlp.float()
    mlp.to(device)

    # Normalize cost between 0 and 1 in the grid
    n_one = (data_out == 1).sum()

    # loss for positives will be multiplied by this factor in the loss function
    p_weight = (data_out.size()[0] - n_one) / n_one
    print("Pos. Weight: ", p_weight)

    # Define the loss function and optimizer
    # loss_function = nn.CrossEntropyLoss()

    # sigmoid included in this loss function
    loss_function = nn.BCEWithLogitsLoss(pos_weight=p_weight)
    optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-2)
    
    #Starting the matplotlib
    plt.figure(figsize=(9, 5))
    plt.title("Accuracy through training")
    

    # Run the training loop
    for epoch in range(0, MAX_EPOCH):

        print(f'Starting epoch {epoch + 1}/{MAX_EPOCH}')

        # Creating batch indices
        permutation = torch.randperm(data_in.size()[0])
        print("data_in.size()",data_in.size())
        # Set current loss value
        current_loss = 0.0
        accuracy = 0

        # Iterate over batches
        for i in range(0, data_in.size()[0], batch_size):

            indices = permutation[i:i + batch_size]
            batch_x, batch_y = data_in[indices], data_out[indices]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Zero the gradient
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(batch_x.float())

            # Compute loss
            loss = loss_function(outputs, batch_y.float())

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print current loss so far
            current_loss += loss.item()
            if (i/batch_size) % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                      ((i/batch_size) + 1, current_loss / (i/batch_size) + 1))

        outputs = torch.sigmoid(mlp(data_in.float()))
        acc = binary_acc(outputs, data_out)
        
        epochs.append(epoch)
        accuracies.append(acc)
        print("Binary accuracy: ", acc, "\n output \n", outputs)

    # Training is complete.
    print("Final output \n", outputs)
    print('MLP trained.')
    
    print("Saving the figure in results/accuracy.png")
    plt.plot(epochs, accuracies)
    plt.savefig("results/accuracy.png")
    return mlp


# IOU evaluation between binary grids
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    accuracy = correct_results_sum / y_test.shape[0]
    accuracy = torch.round(accuracy * 100)
    return accuracy

def generate_occupancy(occupancy):

    #resolution_p = 100
    step = 2 / resolution
    
    print("==============================================")
    print("Starting the construction of the initial guess")

    # Voxel coordinates
    Xp, Yp, Zp = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]
    i = 1
    # read the input silhouettes
    for g in range(12):
        myFile = "image{0}.pgm".format(g)

        img = mpimg.imread(myFile)
        if img.dtype == np.float32:  # if not integer
            img = (img * 255).astype(np.uint8)
        
        projec = calib[g].reshape(3,4)


        for i in range(len(X)):
            for j in range(len(Y)): 
                for k in range(len(Z)//2):
                    p = projec.dot(np.array([Xp[i][j][k], Yp[i][j][k], Zp[i][j][k], 1]).reshape(4,-1))

                    if((int(p[0]/p[2])<300 and int(p[1]/p[2])<300) and img[int(p[0]/p[2]),int(p[1]/p[2])]==0):
                        occupancy[i][j][k] = 0  
    print("==============================================")
    print("Ended the construction of the initial guess")

    

    
    

def main():
    # Voxel occupancy
    occupancy = np.ndarray((resolution, resolution, resolution // 2), dtype=int)

    # Voxels are initially occupied then carved with silhouette information
    occupancy.fill(1)

    # Generate X,Y,Z and occupancy
    generate_occupancy(occupancy)
    	
    # Reduce outside points
    indexes = np.where(occupancy == 1)
    minx = min(indexes[0])
    maxx = max(indexes[0])
    
    miny = min(indexes[1])
    maxy = max(indexes[1])
    
    minz = min(indexes[2])
    maxz = max(indexes[2])
    
	
    offx = minx
    offy = miny
    offz = minz
    
    print(X.shape)
    l,h,w= X.shape

    resolutionx = maxx - minx
    resolutiony = maxy - miny
    resolutionz = maxz - minz
    
    print(resolutionx)
    print(resolutiony)
    print(resolutionz)
    
    print("Generated ocupancy")
    #print(occupancy)
    print("number of insides", np.count_nonzero(occupancy == 1))
    # Format data for PyTorch
    data_in = np.stack((X, Y, Z), axis=-1)
    
    
    data_in = data_in[minx:maxx,miny:maxy,minz:maxz]
    occupancy = occupancy[minx:maxx,miny:maxy,minz:maxz]
    
    
    print("original data")
    print(data_in.shape)
    
    
    resolution_cube = resolutionx * resolutiony * resolutionz
    data_in = np.reshape(data_in, (resolution_cube, 3))
    data_out = np.reshape(occupancy, (resolution_cube, 1))
    
    # Pytorch format
    data_in = torch.from_numpy(data_in).to(device)
    data_out = torch.from_numpy(data_out).to(device)

    # Train mlp
    mlp = nif_train(data_in, data_out, BATCH_SIZE)  # data_out.size()[0])

    # Visualization on training data
    outputs = mlp(data_in.float())
    
    print("**",outputs)
    
    occ = outputs.detach().cpu().numpy()  # from torch format to numpy

    print("occ.shape",occ.shape)
    print("occ",occ)

    # Go back to 3D grid
    newocc = np.reshape(occ, (resolutionx, resolutiony, resolutionz))
    #newocc = np.abs(np.around(newocc))
    
    #print(newocc)
    print("newocc.shape",newocc.shape)
    
    #newocc = np.ndarray((resolution, resolution, resolution // 2), dtype=int)
    #newocc.fill(1)
    
    
    newocc = np.where(newocc<=0., 0, 1)
    #print(newocc)
    print("newocc",newocc)
    print("newocc.shape",newocc.shape)
    
    

    # Marching cubes
    verts, faces, normals, values = measure.marching_cubes(newocc, 0.25)
    # Export in a standard file format
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
    surf_mesh.export('alimplicit.off')


# --------- MAIN ---------
if __name__ == "__main__":
    main()
