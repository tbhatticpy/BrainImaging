import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nibabel as nib

data_folder = "NI-edu-master/NI-edu/fMRI-introduction/week_1"
file = os.path.join(data_folder, 'anat.nii.gz')

imgdata = nib.load(file)
imgdata.dataobj
np.set_printoptions(precision=2, suppress=True)

print(imgdata.affine)

header = imgdata.header

print(header)

img = imgdata.get_fdata()

img_size = header.get_data_shape()
data_type = header.get_data_dtype()
total_voxels = img.size
voxel_size = header.get_zooms()

print("Size of image:", img_size)
print("Data type:", data_type)
print("Total number of voxels:", total_voxels)
print("Voxel size:", voxel_size)


x_mid = img.shape[0] // 2
sagittal_view = img[x_mid,:,:].T

y_mid = img.shape[1] // 2
coronal_view = img[:,y_mid,:].T

z_mid = img.shape[2] // 2
axial_view = img[:,:,z_mid].T

plt.figure(figsize=(6, 6))
plt.imshow(sagittal_view, cmap='gray', origin='lower', aspect='auto')
plt.colorbar(label='Intensity')
plt.title("Sagittal View")
plt.xlabel("Y-axis (mm)")
plt.ylabel("Z-axis (mm)")
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(coronal_view, cmap='gray', origin='lower', aspect='auto')
plt.colorbar(label='Intensity')
plt.title("Coronal View")
plt.xlabel("X-axis (mm)")
plt.ylabel("Z-axis (mm)")
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(axial_view, cmap='gray', origin='lower', aspect='auto')
plt.colorbar(label='Intensity')
plt.title("Axial View")
plt.xlabel("X-axis (mm)")
plt.ylabel("Y-axis (mm)")
plt.show()


ig, axes = plt.subplots(1, 3, figsize=(15, 6))

axes[0].imshow(sagittal_view, cmap='gray', origin='lower', aspect='auto')
axes[0].set_title("Sagittal View")
axes[0].set_xlabel("Y-axis (mm)")
axes[0].set_ylabel("Z-axis (mm)")
plt.colorbar(axes[0].imshow(sagittal_view, cmap='gray', origin='lower', aspect='auto'), ax=axes[0], label="Intensity")

axes[1].imshow(coronal_view, cmap='gray', origin='lower', aspect='auto')
axes[1].set_title("Coronal View")
axes[1].set_xlabel("X-axis (mm)")
axes[1].set_ylabel("Z-axis (mm)")
plt.colorbar(axes[1].imshow(coronal_view, cmap='gray', origin='lower', aspect='auto'), ax=axes[1], label="Intensity")

axes[2].imshow(axial_view, cmap='gray', origin='lower', aspect='auto')
axes[2].set_title("Axial View")
axes[2].set_xlabel("X-axis (mm)")
axes[2].set_ylabel("Y-axis (mm)")
plt.colorbar(axes[2].imshow(axial_view, cmap='gray', origin='lower', aspect='auto'), ax=axes[2], label="Intensity")
plt.tight_layout()
plt.show()


y_mid_low = img.shape[1] // 4
coronal_cerebellum_view = img[:,y_mid_low,:].T

z_mid_low = img.shape[2] // 4
axial_cerebellum_view = img[:,:,z_mid_low].T

plt.figure(figsize=(6, 6))
plt.imshow(coronal_cerebellum_view, cmap='gray', origin='lower', aspect='auto')
plt.colorbar(label='Intensity')
plt.title("Coronal View with Cerebellum")
plt.xlabel("Y-axis (mm)")
plt.ylabel("Z-axis (mm)")
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(axial_cerebellum_view, cmap='gray', origin='lower', aspect='auto')
plt.colorbar(label='Intensity')
plt.title("Axial View with Cerebellum")
plt.xlabel("X-axis (mm)")
plt.ylabel("Y-axis (mm)")
plt.show()


sim_brain_data = np.load("sim_brain.npy")
data_shape = sim_brain_data.shape
print("Data shape:", data_shape)

z_index = 100
axial_slice = sim_brain_data[:, :, z_index]

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(axial_slice, cmap="gray", origin="lower", aspect="auto")

lession_points = [
    (75, 110,15, 15), #via visual inspection
    (120, 97, 30, 17),
    (74, 59, 18, 10),
    (131, 70, 8, 8),
]

for idx in lession_points:
    x, y, w, h = idx
    rect = Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor="none")
    ax.add_patch(rect)

ax.set_title("Axial View with Bounding Boxes around Lesions")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
plt.show()


data_folder = "NI-edu-master/NI-edu/fMRI-introduction/week_1"
file = os.path.join(data_folder, 'func.nii.gz')

imgdata = nib.load(file)
imgdata.dataobj
np.set_printoptions(precision=2, suppress=True)

print(imgdata.affine)

header = imgdata.header

print(header)

img = imgdata.get_fdata()

img_size = header.get_data_shape()
data_type = header.get_data_dtype()
total_voxels = img.size
voxel_size = header.get_zooms()

print("Size of image:", img_size)
print("Data type:", data_type)
print("Total number of voxels:", total_voxels)
print("Voxel size:", voxel_size)


z_mid = img.shape[2] // 2
axial_view = img[:,:,z_mid,:].T

time_dimension = img.shape[3]
print("Total time points: ",time_dimension)

time_points = np.linspace(0, time_dimension - 1, 8, dtype=int)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

intensity_min = axial_view.min()
intensity_max = axial_view.max()

for i, t in enumerate(time_points):
    ax = axes.flat[i]
    im = ax.imshow(img[:, :, z_mid, t].T, cmap='viridis', origin='lower', vmin=intensity_min, vmax=intensity_max)
    ax.set_title(f"Time Point: t={t}")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, label="Intensity")

plt.show()


t0_slice = img[:, :, z_mid, 0]
highest_voxel_index = np.unravel_index(np.argmax(t0_slice), t0_slice.shape)
print("Voxel index with the highest value: ", highest_voxel_index)


time_series = img[highest_voxel_index[0], highest_voxel_index[1], z_mid, :]

plt.figure(figsize=(10, 6))
plt.plot(range(time_dimension), time_series, marker='o', label="Voxel Time Series")
plt.title("Time Series for Voxel with the Highest Value at t=0", fontsize=14)
plt.xlabel("Time Point (t)", fontsize=12)
plt.ylabel("Voxel Intensity", fontsize=12)
plt.xticks(range(0, time_dimension, max(1, time_dimension // 10)))
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
