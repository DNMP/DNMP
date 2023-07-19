# Train Mesh AutoEncoder

We suggest to use the pretrained weights in `pretrained/mesh_ae/mesh_ae.pth`. 
Nevertheless, we provide a simple guidance for training mesh autoencoder on customize data (comming soon).

# Step 1
Preparing point cloud patches. 

You should generate a large number of local point cloud patches and save them as `.npy` files in a single folder.

# Step 2
Fitting meshes using the point cloud patches.

Please refer to `mesh_fitting.py` for this process. You should change the `data_dir` and `out_dir` in the script.

# Step 3
Training autoencoder.

Please refer to `train_ae.py` for this process. 