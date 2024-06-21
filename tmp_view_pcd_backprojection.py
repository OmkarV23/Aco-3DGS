import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

data = np.load('backprojected_scene.npy')
data = data.reshape(150,150,120)

data = np.abs(data)

def create_voxels(x_min, x_max, y_min, y_max, z_min, z_max, num_x, num_y, num_z, permute_xy=False):
    scene_corners = np.array(([x_min, y_min, z_min],
                              [x_min, y_max, z_min],
                              [x_min, y_max, z_max],
                              [x_min, y_min, z_max],
                              [x_max, y_min, z_min],
                              [x_max, y_max, z_min],
                              [x_max, y_max, z_max],
                              [x_max, y_min, z_max]))

    x_vect = np.linspace(x_min, x_max, num_x, endpoint=True)
    y_vect = np.linspace(y_min, y_max, num_y, endpoint=True)
    z_vect = np.linspace(z_min, z_max, num_z, endpoint=True)

    x_dim = np.abs(x_max - x_min)
    y_dim = np.abs(y_max - y_min)
    z_dim = np.abs(z_max - z_min)

    if permute_xy:
        (x, y, z) = np.meshgrid(y_vect, x_vect, z_vect)
    else:
        (x, y, z) = np.meshgrid(x_vect, y_vect, z_vect)

    voxels = np.hstack((np.reshape(x, (np.size(x), 1)),
                        np.reshape(y, (np.size(y), 1)),
                        np.reshape(z, (np.size(z), 1))
                       ))

    return voxels, scene_corners

voxels, corners = create_voxels(-0.125, 0.125,
                       -0.125, 0.125,
                       0.00, 0.2,
                       150, 150, 120)

mag = np.abs(data)
mag = mag.ravel()

u = mag.mean()
var = mag.std()
# vals = np.arange(0., 10, 0.5)
vals = 7
thresh_vals = u + vals*var

mag[mag[:] < thresh_vals] = None

# for thresh_count in tqdm(range(0, len(thresh_vals)), desc="Saving 3D plots"):
#     mag[mag[:] < thresh_vals[thresh_count]] = None

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.clear()
im = ax.scatter(voxels[:, 0],
            voxels[:, 1],
            voxels[:, 2],
            c=mag, alpha=0.2)
ax.set_xlim3d((corners[:, 0].min(), corners[:, 0].max()))
ax.set_ylim3d((corners[:, 1].min(), corners[:, 1].max()))
ax.set_zlim3d((corners[:, 2].min(), corners[:, 2].max()))
plt.grid(True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.colorbar(im)
# plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'bf_3d_thresh_' + str(thresh_count) + '.png'))
plt.show()
