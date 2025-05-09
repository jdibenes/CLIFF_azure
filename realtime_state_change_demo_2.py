import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from pytorch3d import transforms
import trimesh.exchange
import trimesh.exchange.obj
import trimesh.visual

from models.smpl import SMPL
from common import constants

from losses import *
from smplify import SMPLify

from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from common.utils import strip_prefix_if_present, cam_crop2full
# from common.mocap_dataset import MocapDataset  # not used in this live demo

# For 3D visualization
import pyrender
import trimesh

import trimesh.transformations as tf

import visualizer

from PIL import Image

from pyrender.constants import TextAlign

# Joint map (for reference)
JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose',
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder',
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee',
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe',
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',
'Right Knee',
'Right Hip',
'Left Hip',
'Left Knee',
'Left Ankle',
'Right Wrist',
'Right Elbow',
'Right Shoulder',
'Left Shoulder',
'Left Elbow',
'Left Wrist',
'Neck (LSP)',
'Top of Head (LSP)',
'Pelvis (MPII)',
'Thorax (MPII)',
'Spine (H36M)',
'Jaw (H36M)',
'Head (H36M)',
'Nose',
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear'
]

def preprocess_frame(frame, target_size=(224, 224)):
    """
    Preprocess the input frame (BGR) to a normalized tensor.
    Returns norm_img as a tensor of shape (1, 3, H, W) in float32.
    """
    resized = cv2.resize(frame, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
    norm_img = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    return norm_img

def create_arrow(start, end, shaft_radius=0.005, head_radius=0.01, head_length=0.02, sections=20):
    """
    Create an arrow mesh from a start point (tail) to an end point (head) using trimesh.
    The arrow is built from a cylinder (shaft) and a cone (head).
    """
    vec = end - start
    total_length = np.linalg.norm(vec)
    if total_length < 1e-6:
        return None
    direction = vec / total_length

    # Reserve space for the arrow head.
    shaft_length = max(total_length - head_length, total_length * 0.8)
    head_length = total_length - shaft_length

    # Create the shaft as a cylinder along the Z-axis.
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_length, sections=sections)
    shaft.apply_translation([0, 0, shaft_length / 2.0])

    # Create the head as a cone along the Z-axis.
    head = trimesh.creation.cone(radius=head_radius, height=head_length, sections=sections)
    head.apply_translation([0, 0, shaft_length + head_length / 2.0])

    # Combine shaft and head.
    arrow = trimesh.util.concatenate([shaft, head])

    # Align the arrow (default along Z) with the desired direction.
    z_axis = np.array([0, 0, 1])
    rot_matrix = trimesh.geometry.align_vectors(z_axis, direction)
    if rot_matrix is None:
        rot_matrix = np.eye(3)
    elif rot_matrix.shape == (4, 4):
        rot_matrix = rot_matrix[:3, :3]
    
    T_rot = np.eye(4)
    T_rot[:3, :3] = rot_matrix
    arrow.apply_transform(T_rot)

    # Translate so that its base (tail) is at the start position.
    arrow.apply_translation(start)
    return arrow

# Define state configurations.
STATE_CONFIG = {
    0: {
        'text': "Check knees visually for redness",
        'targets': lambda joints: [joints[JOINT_NAMES.index("OP LKnee")], joints[JOINT_NAMES.index("OP RKnee")]]  # Left and Right knees
    },
    1: {
        'text': "Check for redness and swelling",
        'targets': lambda joints: [
            (joints[JOINT_NAMES.index("OP LKnee")] + joints[JOINT_NAMES.index("OP LAnkle")]) / 2.0,  # Left calf (midpoint between knee and ankle)
            (joints[JOINT_NAMES.index("OP RKnee")] + joints[JOINT_NAMES.index("OP RAnkle")]) / 2.0   # Right calf (midpoint between knee and ankle)
        ]
    },
    2: {
        'text': "Check between toes",
        'targets': lambda joints: [
            (joints[JOINT_NAMES.index("OP LSmallToe")] + joints[JOINT_NAMES.index("OP LBigToe")]) / 2.0,  # Left toe (midpoint between big and small toe)
            (joints[JOINT_NAMES.index("OP RSmallToe")] + joints[JOINT_NAMES.index("OP RBigToe")]) / 2.0   # Right toe (midpoint between big and small toe)
        ]
    },
    3: {
        'text': "Check for swelling",
        'targets': lambda joints: [joints[JOINT_NAMES.index("OP RHeel")], joints[JOINT_NAMES.index("OP LHeel")]]  # Left and Right heels
    }
}



class demo:
    def run(self, args):
        # Device selection
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {device}')

        # Create colormap
        self._colormap = cv2.applyColorMap(np.arange(0, 256, 1, dtype=np.uint8), cv2.COLORMAP_AUTUMN)
        self._colormap = self._colormap[:, :, ::-1]
        self._colormap = np.dstack((self._colormap, 255 * np.ones(self._colormap.shape[0:2] + (1,), dtype=self._colormap.dtype)))

        self._mesh_colors = np.zeros((6890, 4), dtype=np.uint8)
        self._mesh_colors[:, 0] = 102
        self._mesh_colors[:, 1] = 102
        self._mesh_colors[:, 2] = 102
        self._mesh_colors[:, 3] = 255

        # Parameters
        viewport_width, viewport_height = 1280, 720
        focal_length = torch.tensor([500.0], device=device, dtype=torch.float32)
        camera_yfov = np.pi / 3

        # Load the pretrained CLIFF model.
        cliff = eval("cliff_hr48")
        cliff_model = cliff('./data/smpl_mean_params.npz').to(device)
        state_dict = torch.load('./data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt')['model']
        state_dict = strip_prefix_if_present(state_dict, prefix="module.")
        cliff_model.load_state_dict(state_dict, strict=True)
        cliff_model.eval()

        # Load the SMPL model.
        smpl = SMPL(constants.SMPL_MODEL_DIR, batch_size=1).to(device)

        # Set up the webcam and offscreen pyrender renderer.
        cap = visualizer.camera_opencv()
        cap.open()

        # Model stuff
        frame_data = cap.read()
        if ('depth_intrinsics' in frame_data):
            print(f"depth_intrinsics={frame_data['depth_intrinsics']}")
            print(f"depth_scale={frame_data['depth_scale']}")
        if ('color_intrinsics' in frame_data):
            print(f"color_intrinsics={frame_data['color_intrinsics']}")

        frame = frame_data['color']
        img_h, img_w = frame.shape[:2]
        center = torch.tensor([[img_w / 2.0, img_h / 2.0]], device=device, dtype=torch.float32)
        scale = torch.tensor([1.0], device=device, dtype=torch.float32)
        b = scale * 200
        bbox_info = torch.stack([center[:, 0] - img_w / 2.0, center[:, 1] - img_h / 2.0, b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)
        full_img_shape = torch.stack((torch.tensor([img_h], device=device, dtype=torch.float32), torch.tensor([img_w], device=device, dtype=torch.float32)), dim=-1)

        # Load texture
        texture_image = Image.open("./data/smpl_uv_20200910.png")
        with open('./data/smpl_uv.obj', 'r') as obj_file:
            obj_mesh_a = trimesh.exchange.obj.load_obj(obj_file, maintain_order=True)
        with open('./data/smpl_uv.obj', 'r') as obj_file:
            obj_mesh_b = trimesh.exchange.obj.load_obj(obj_file)
        mesh_vertices_b = obj_mesh_b['geometry']['./data/smpl_uv.obj']['vertices']
        mesh_faces_a = obj_mesh_a['geometry']['./data/smpl_uv.obj']['faces']
        mesh_faces_b = obj_mesh_b['geometry']['./data/smpl_uv.obj']['faces']
        mesh_visuals_b = obj_mesh_b['geometry']['./data/smpl_uv.obj']['visual']
        uv_transform = np.zeros(mesh_vertices_b.shape[0], dtype=np.int64)
        for face_index in range(0, mesh_faces_b.shape[0]):
            for vertex_index in range(0, 3):
                uv_transform[mesh_faces_b[face_index, vertex_index]] = mesh_faces_a[face_index, vertex_index]
        mesh_visuals = trimesh.visual.TextureVisuals(uv=mesh_visuals_b.uv, image=texture_image)        
        
        # Initialize visualization utilities
        joint_solver  = visualizer.solver(camera_yfov, viewport_width, viewport_height)
        scene_control = visualizer.scene_manager(camera_yfov)

        self._cap = cap
        self._device = device
        self._cliff_model = cliff_model
        self._smpl = smpl
        self._bbox_info = bbox_info
        self._joint_solver = joint_solver
        self._scene_control = scene_control
        self._regions = [
            joint_solver.focus_center_whole,
            joint_solver.focus_right_lower_leg,
            joint_solver.focus_left_lower_leg,
        ]
        self._mesh_visuals = mesh_visuals
        self._uv_transform = uv_transform
        self._mesh_faces_b = mesh_faces_b

        # Start with state 1 and focus full body.
        self._current_state = 0
        self._current_region = 0
        self._next_region = 0
        self._pointer_position = [0, 0]
        self._pointer_size = 0.05
        self._use_texture = False

        caption = [dict(
            text     = STATE_CONFIG[self._current_state]['text'],
            location = TextAlign.TOP_CENTER,
            font_name = r"C:\Windows\Fonts\arial.ttf",
            font_pt   = 30,
            color    = (0.,1.,0.,1.),
            scale    = 1.0
            )
        ]

        di = 0.02
        da = (10/180) * np.pi
        dr = 0.01

        key_handlers = {
            '1': (self.advance_state,  []),
            '2': (self.traverse, [di, 0]),
            '3': (self.traverse, [-di, 0]),
            '4': (self.traverse, [0, da]),
            '5': (self.traverse, [0, -da]),
            '6': (self.adjust_pointer, [dr]),
            '7': (self.adjust_pointer, [-dr]),
            '8': (self.advance_target, []),
            '9': (self.toggle_texture, []),
        }

        viewer_flags = {'caption' : caption, 'vsv.hook_on_begin' : (self.animate, [])}

        print("Starting real-time inference.")
        print("Press 'q' to exit.")
        print("Press 1 to change state.")
        print("Press 2/3 to move pointer.")
        print("Press 4/5 to rotate pointer.")
        print("Press 6/7 to adjust pointer size.")
        print("Press 8 to cycle through full body / right leg / left leg.")
        print("Press 9 to cycle between vertex colors / texture.")

        visualizer.viewer(self._scene_control._scene,
                    viewport_size=(viewport_width, viewport_height),
                    viewer_flags=viewer_flags,
                    use_raymond_lighting=True,
                    registered_keys=key_handlers)

        cap.close()

    def advance_state(self, viewer):
        self._current_state = (self._current_state + 1) % len(STATE_CONFIG)
        viewer.viewer_flags['caption'][0]['text'] = STATE_CONFIG[self._current_state]['text']

    def advance_target(self, viewer):
        self._next_region = (self._current_region + 1) % len(self._regions)
        self._pointer_position = [0, 0]

    def traverse(self, viewer, delta_displacement, delta_angle):
        displacement = self._pointer_position[0] + delta_displacement
        angle = self._pointer_position[1] + delta_angle
        if (angle > np.pi):
            angle = angle - (2*np.pi)
        if (angle < -np.pi):
            angle = angle + (2*np.pi)
        self._pointer_position = [displacement, angle]

    def adjust_pointer(self, viewer, delta_radius):
        radius = max([self._pointer_size + delta_radius, 0])
        self._pointer_size = radius

    def toggle_texture(self, viewer):
        self._use_texture = not self._use_texture

    def animate(self, viewer):
        frame_data = self._cap.read()
        frame = frame_data['color']
        if ('depth' in frame_data):
            depth = frame_data['depth']
            cv2.imshow('Depth', cv2.applyColorMap((((depth * frame_data['depth_scale']) / 7.5) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
        cv2.imshow('Video', frame)
        cv2.waitKey(1)

        norm_img = preprocess_frame(frame, target_size=(224, 224)).to(self._device)

        # Run the CLIFF model.
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_cam_crop = self._cliff_model(norm_img, self._bbox_info)
        
        smpl_poses = transforms.matrix_to_axis_angle(pred_rotmat).contiguous().view(-1, 72)

        with torch.no_grad():
            pred_output = self._smpl(betas=pred_betas, body_pose=smpl_poses[:, 3:], global_orient=smpl_poses[:, :3], pose2rot=True)
        
        vertices = pred_output.vertices.cpu().detach().numpy()[0]
        joints = pred_output.joints.cpu().detach().numpy()[0].T
        faces = self._smpl.faces

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # todo: project mesh into image
        if (self._next_region is not None):
            self._current_region = self._next_region
            self._next_region    = None
            update_region        = True
        else:
            update_region        = False
        
        self._joint_solver.set_smpl_mesh(mesh, joints, None)

        camera_pose, focus_center, axis_displacement = self._regions[self._current_region]()

        camera_pose[:3, 3:4] = focus_center
        align_pose = np.linalg.inv(camera_pose)

        mesh.apply_transform(align_pose)
        joints = align_pose[:3, :3] @ joints + align_pose[:3, 3:4]

        self._joint_solver.set_smpl_mesh(mesh, joints, None)

        camera_pose, focus_center, axis_displacement = self._regions[self._current_region]()
        focus_axis = camera_pose[:3, 1]
        forward = camera_pose[:3, 2:3]

        displacement, angle = self._pointer_position

        pose   = camera_pose
        center = focus_center
        up     = pose[:3, 1:2]
        front  = pose[:3, 2:3]

        rotation = cv2.Rodrigues(up.reshape((-1,)) * angle)[0]

        center = center + up * displacement
        front  = rotation @ front

        point, face_index, vertex_index = self._joint_solver.face_solver(center, front)

        if (point is not None):
            distances = self._joint_solver.select_vertices(vertex_index, self._pointer_size, np.Inf)
            selected_indices = distances.keys()
            filtered_faces, filtered_indices = self._joint_solver.select_complete_faces(selected_indices)
            if (len(filtered_indices) > 0):
                selected_indices = filtered_indices
        else:
            distances = None
            selected_indices = None

        colors = self._mesh_colors.copy()
        if ((selected_indices is not None) and (distances is not None)):
            for vertex_index in selected_indices:
                colors[vertex_index, :] = self._colormap[min([int(255*(distances[vertex_index] / max([self._pointer_size, 0.001]))),255]), 0, :]

        vertices_b = mesh.vertices[self._uv_transform, :]
        colors_b = colors[self._uv_transform, :]
        faces_b = self._mesh_faces_b

        if (self._use_texture):
            mesh2 = trimesh.Trimesh(vertices=vertices_b, faces=faces_b, visual=self._mesh_visuals, process=False)
        else:
            mesh2 = trimesh.Trimesh(vertices=vertices_b, faces=faces_b, vertex_colors=colors_b, process=False)

        self._scene_control.set_smpl_mesh(mesh2)
        self._scene_control.clear_group('arrows')

        if (point is not None):
            arrow = create_arrow((point + front*0.13).reshape((-1,)), (point + front*0.03).reshape((-1,)), 0.005, 0.015)
            if (arrow is not None):
                mat   = pyrender.MetallicRoughnessMaterial(baseColorFactor=(1, 0, 0, 1))
                self._scene_control.add('arrows', pyrender.Mesh.from_trimesh(arrow, material=mat, smooth=False))

        if self._current_state in STATE_CONFIG:
            targets = STATE_CONFIG[self._current_state]['targets'](joints.T)
            for target in targets:
                # Define a constant tail offset. Adjust this as necessary.
                offset = np.array([0.0, -0.02, 0.4])
                head_offset = np.array([0.0, 0.0, 0.1])
                target = target +head_offset
                tail = target + offset

                arrow_mesh = create_arrow(tail, target, shaft_radius=0.01,head_radius=0.03, head_length=0.08)
                if arrow_mesh is not None:
                    arrow_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=(1.0, 0.0, 0.0, 1.0))
                    #arrow_mesh.apply_transform(R_flip)
                    arrow_pyrender = pyrender.Mesh.from_trimesh(arrow_mesh, material=arrow_material, smooth=False)

                    self._scene_control.add('arrows', arrow_pyrender)

        camera_pose[:3, 3:4] = (focus_center + 1.2 * axis_displacement * camera_pose[:3, 2:3])
        self._scene_control.set_camera_pose(camera_pose)

        if (update_region):
            viewer.set_camera_target(camera_pose, focus_axis, focus_center)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections (if available)')
    args = parser.parse_args()

    d = demo()
    d.run(args)

if __name__ == '__main__':
    main()
