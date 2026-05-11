# copy from nerfstudio and 2DGS
import torch
from matplotlib import cm
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import cv2


def pca_feature(feat):
    C, H, W = feat.shape
    feat = feat.contiguous().view(C, H * W).permute(1, 0).detach().cpu().numpy()
    pca = PCA(n_components=3)
    feat = pca.fit_transform(feat)  # (H*W, 3)
    feat = torch.from_numpy(feat).cuda().permute(1, 0).contiguous().reshape(3, H, W)
    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-3)
    return feat


def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colored_img = colormap(image.cpu().detach().numpy())[..., :3]  # [h, w, 3], range [0, 1]
    colored_img = torch.from_numpy(colored_img).to(image)
    return colored_img

def apply_depth_colormap(
    depth,
    accumulation,
    near_plane = 2.0,
    far_plane = 6.0,
    cmap="turbo",
):
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)

    if accumulation is not None:
        accumulation = accumulation
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image



def save_points(path_save, pts, colors=None, normals=None, BRG2RGB=False):
    """save points to point cloud using open3d"""
    assert len(pts) > 0
    if colors is not None:
        assert colors.shape[1] == 3
    assert pts.shape[1] == 3

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        # Open3D assumes the color values are of float type and in range [0, 1]
        if np.max(colors) > 1:
            colors = colors / np.max(colors)
        if BRG2RGB:
            colors = np.stack([colors[:, 2], colors[:, 1], colors[:, 0]], axis=-1)
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)

    o3d.io.write_point_cloud(path_save, cloud)
    

def colormap(img, cmap='jet'):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data / 255.).float().permute(2,0,1)
    plt.close()
    if img.shape[1:] != (H, W):
        img = torch.nn.functional.interpolate(img[None], (W, H), mode='bilinear', align_corners=False)[0]
    return img

def convert_colmap_to_opengl(normal_map):
    normal_map_opengl = normal_map.clone()
    normal_map_opengl[1] = -normal_map[1]
    normal_map_opengl[2] = -normal_map[2]
    return normal_map_opengl

def apply_normal_colormap(normal, opengl_color=True):
    if opengl_color:
        normal = convert_colmap_to_opengl(normal)
    return (normal + 1.) / 2.

def draw_distribution(x, y, pred, gt, x_label, y_label, title):
    fig, ax = plt.subplots(figsize=(8, 4))

    # ax.bar(x, y, color='blue', edgecolor='black')
    ax.plot(x, y, '-', color='blue')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()
    plt.axvline(pred, color='green', linestyle='--', label=f'pred value: {pred}')
    plt.axvline(gt, color='red', linestyle='-', label=f'gt value: {gt}')
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    image = Image.open(buf).convert("RGB")
    tensor_image = transforms.ToTensor()(image)

    return tensor_image
