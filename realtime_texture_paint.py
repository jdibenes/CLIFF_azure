
import cv2
import argparse
import numpy as np
import time
import json
import torch
import mesh_solver
import visualizer

from pytorch3d import transforms
from models.smpl import SMPL
from common import constants
from losses import *
from smplify import SMPLify
from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from common.utils import strip_prefix_if_present, cam_crop2full
# from common.mocap_dataset import MocapDataset  # not used in this live demo


class demo2:
    def run(self, args):
        # Set up the webcam
        #self._cap = visualizer.camera_opencv()
        self._cap = visualizer.camera_null()
        self._cap.open()

        # Create offscreen renderer
        renderer_width = 1280
        renderer_height = 720
        fov_vertical = np.pi / 3
        fxy = mesh_solver.geometry_fov_to_f(fov_vertical, renderer_height)
        cfg_offscreen = mesh_solver.renderer_create_settings_offscreen(renderer_width, renderer_height)
        cfg_scene = mesh_solver.renderer_create_settings_scene()
        cfg_camera = mesh_solver.renderer_create_settings_camera(fxy, fxy, renderer_width // 2, renderer_height // 2)
        cfg_camera_transform = mesh_solver.renderer_create_settings_camera_transform()
        cfg_lamp = mesh_solver.renderer_create_settings_lamp()
        self._offscreen_renderer = mesh_solver.renderer(cfg_offscreen, cfg_scene, cfg_camera, cfg_camera_transform, cfg_lamp)

        # Load textures and UV map
        tex_filename = './data/textures/f_01_alb.002_1k.png'
        #tex_filename = './data/textures/smpl_uv_20200910.png'
        self._texture_array = mesh_solver.texture_load_image(tex_filename)
        self._test_stamp = mesh_solver.texture_load_image('./data/textures/stamp_test.jpg')
        self._offscreen_renderer.smpl_load_uv('./data/smpl_uv.obj', self._texture_array.shape)

        # Create sample text texture
        font = mesh_solver.texture_load_font('arial.ttf', 512)
        self._test_text = mesh_solver.texture_create_multiline_text(['Sample', 'Text'], font, (255, 0, 0, 255), (255, 255, 255, 255), 1, 20)
        self._test_text = mesh_solver.texture_pad(self._test_text, 0.05, 0.1, (255, 255, 255, 255))

        # Load CLIFF and SMPL
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {self._device}')
        self._offscreen_renderer.smpl_load_model(self._device)
        with open('./test_msg.txt', 'r') as json_file:
            self._test_msg = json.load(json_file)
        
        # Create UI elements
        self._cone = mesh_solver.mesh_create_cone(0.015, 0.04, 10)
        self._sphere = mesh_solver.mesh_create_sphere(0.003)
        self._pose_cone =  np.eye(4, dtype=np.float32)
        self._pose_sphere = np.eye(4, dtype=np.float32)

        # Camera translation mode
        self._use_plane = True

        # piecewise cylindrical coordinates increments
        self._regions = ['body_center', 'thigh_left', 'thigh_right', 'lower_leg_left', 'lower_leg_right', 'foot_left', 'foot_right', 'head_center', 'upper_arm_left', 'upper_arm_right', 'lower_arm_left', 'lower_arm_right']
        self._region_index = 0
        self._displacement = 0
        self._angle = 0        
        self._di = 0.02
        self._da = (10/180) * np.pi
        
        # Run inference and painting
        start = time.perf_counter()
        count = 0
        while (self._loop()):
            count += 1
            end = time.perf_counter()
            if (end-start > 2.0):
                print(f'FPS: ', count / (end-start))
                start = end
                count = 0

        # Close camera
        self._cap.close()

    def _loop(self):
        smpl_params = self._test_msg['persons'][0]['smpl_params']
        camera_translation = self._test_msg['persons'][0]['camera_translation']
        smpl_params = { k : torch.tensor([v], dtype=torch.float32, device=self._device) for k, v in smpl_params.items() }
        camera_translation = torch.tensor([camera_translation], dtype=torch.float32, device=self._device)

        vertices, joints, faces = self._offscreen_renderer.smpl_get_mesh(smpl_params)
        pred_vertices_world = (vertices + camera_translation.unsqueeze(1)).detach().cpu().numpy()

        vertices = vertices[0].cpu().numpy()
        joints = joints[0].cpu().numpy()

        joints = mesh_solver.smpl_joints_to_openpose(joints)

        # Create mesh
        mesh = mesh_solver.mesh_create(vertices, faces)

        # Compute pose to set mesh upright
        frame_0 = mesh_solver.smpl_mesh_chart_openpose(mesh, joints).create_frame('body_center')
        smpl_mesh_pose = np.eye(4, dtype=np.float32)
        smpl_mesh_pose[0:1, :3] = frame_0.left
        smpl_mesh_pose[1:2, :3] = frame_0.up
        smpl_mesh_pose[2:3, :3] = frame_0.front
        smpl_mesh_pose[3:4, :3] = frame_0.center
        smpl_mesh_pose = np.linalg.inv(smpl_mesh_pose).T

        # Add SMPL mesh to the main scene
        self._offscreen_renderer.mesh_add_smpl('smpl', 'patient', mesh, joints, self._texture_array, smpl_mesh_pose)

        # Map current cylindrical coordinates to SMPL mesh point and face
        smpl_frame = self._offscreen_renderer.smpl_chart_create_frame('smpl', 'patient', self._regions[self._region_index])
        anchor = self._offscreen_renderer.smpl_chart_from_cylindrical('smpl', 'patient', smpl_frame, self._displacement, self._angle)

        # Set cone position based on cylindrical coordinates
        self._pose_cone[1:2, :3] = mesh_solver.math_transform_bearings(smpl_frame.up, smpl_mesh_pose.T, False)
        self._pose_cone[2:3, :3] = mesh_solver.math_transform_bearings(-anchor.direction, smpl_mesh_pose.T, False)
        self._pose_cone[0:1, :3] = np.cross(self._pose_cone[1:2, :3], self._pose_cone[2:3, :3])
        if (anchor.point is not None):
            self._pose_cone[3:4, :3] = mesh_solver.math_transform_points(anchor.point + 0.04 * anchor.direction, smpl_mesh_pose.T, False)
        else:
            # Coordinates outside mesh
            self._pose_cone[3:4, :3] = mesh_solver.math_transform_points(anchor.position, smpl_mesh_pose.T, False)

        # Poses convert object to world
        #camera_pose = self._offscreen_renderer.camera_get_transform_local()
        #raycast = self._offscreen_renderer.mesh_operation_raycast('smpl', 'patient', camera_pose[:3, 3:4].T, -camera_pose[:3, 2:3].T)
        #if (raycast.point is not None):
        #    self._pose_sphere[3:4, :3] = mesh_solver.math_transform_points(raycast.point, smpl_mesh_pose.T, False)

        # Set sphere position to SMPL mesh point closest to the cone (rotation does not matter for sphere)
        closest = self._offscreen_renderer.mesh_operation_closest('smpl', 'patient', self._pose_cone[3:4, :3])
        if (closest.point is not None):
            self._pose_sphere[3:4, :3] = mesh_solver.math_transform_points(closest.point, smpl_mesh_pose.T, False)

        # Add cone and sphere to the main scene
        self._offscreen_renderer.mesh_add_user('ui', 'cursor', self._cone, self._pose_cone.T)
        self._offscreen_renderer.mesh_add_user('ui', 'closest', self._sphere, self._pose_sphere.T)

        if (anchor.point is not None):        
            # Paint "circle" on mesh
            #self._offscreen_renderer.smpl_paint_brush_solid('smpl', 'patient', anchor, 0.02, np.array([0, 255, 0, 255], dtype=np.uint8), 0.5)
            #self._offscreen_renderer.smpl_paint_brush_gradient('smpl', 'patient', anchor, 0.01, np.array([255, 0, 0, 255], dtype=np.uint8), np.array([255, 255, 0, 255], dtype=np.uint8), 0.33, 0.25)
            
            # Paint decal on mesh
            # decals require an align prior to mantain a consistent orientation        
            align_prior = self._offscreen_renderer.smpl_paint_decal_align_prior('smpl', 'patient', anchor, smpl_frame.up, smpl_frame.front)
            #self._offscreen_renderer.smpl_paint_decal_solid('smpl', 'patient', anchor, self._test_stamp, align_prior, 0, 10000 * 1, False, 0.5)
            self._offscreen_renderer.smpl_paint_decal_solid('smpl', 'patient', anchor, self._test_text, align_prior, 0, 10000 * 2, False, 0.5)

        # Compute painted texture
        self._offscreen_renderer.smpl_paint_flush('smpl', 'patient')

        # Remove painting
        self._offscreen_renderer.smpl_paint_clear('smpl', 'patient')

        # Add meshes to display, update poses and textures
        self._offscreen_renderer.mesh_present_smpl('smpl', 'patient')
        self._offscreen_renderer.mesh_present_user('ui', 'cursor')
        self._offscreen_renderer.mesh_present_user('ui', 'closest')

        world_points = mesh_solver.math_transform_points(smpl_frame.points, smpl_mesh_pose.T, False)
        image_points = self._offscreen_renderer.camera_project_points(world_points)[0]

        # Render
        color, depth = self._offscreen_renderer.scene_render()
        color = color.copy()
        for i in range(0, image_points.shape[0]):
            center = (int(image_points[i, 0]), int(image_points[i, 1]))
            color = cv2.circle(color, center, 3, [255, 0, 255], -1)

        # Show rendered image
        cv2.imshow('offscreen test', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF

        # Controls
        if (key == 49): # 1
            self._region_index = (self._region_index + 1) % len(self._regions)
            smpl_frame = self._offscreen_renderer.smpl_chart_create_frame('smpl', 'patient', self._regions[self._region_index])
            focus_center = mesh_solver.math_transform_points(smpl_frame.center, smpl_mesh_pose.T, False)
            focus_points = mesh_solver.math_transform_points(smpl_frame.points, smpl_mesh_pose.T, False)
            wz = self._offscreen_renderer.camera_solve_fov_z(focus_center, focus_points, False)
            self._offscreen_renderer.camera_adjust_parameters(center=focus_center, distance=wz, relative=False)
            self._displacement = 0
            self._angle = 0
            
        if (key == 50): # 2
            self._displacement += self._di
        if (key == 51): # 3
            self._displacement -= self._di
        if (key == 52): # 4
            self._angle += self._da
        if (key == 53): # 5
            self._angle -= self._da

        if (key == 68 or key == 100): # d
            self._offscreen_renderer.camera_adjust_parameters(yaw=10)
        if (key == 65 or key == 97): # a
            self._offscreen_renderer.camera_adjust_parameters(yaw=-10)
        if (key == 87 or key == 119): # w
            self._offscreen_renderer.camera_adjust_parameters(pitch=10)
        if (key == 83 or key == 115): # s
            self._offscreen_renderer.camera_adjust_parameters(pitch=-10)
        if (key == 82 or key == 114): #r
            self._offscreen_renderer.camera_adjust_parameters(distance=-0.1)
        if (key == 70 or key == 102): #f
            self._offscreen_renderer.camera_adjust_parameters(distance=0.1)
        if (key == 78 or key == 110): #n
            self._offscreen_renderer.camera_move_center([0.1, 0, 0], self._use_plane)
        if (key == 77 or key == 109): #m
            self._offscreen_renderer.camera_move_center([-0.1, 0, 0], self._use_plane)
        if (key == 85 or key == 117): #u
            self._offscreen_renderer.camera_move_center([0, 0.1, 0], self._use_plane)
        if (key == 74 or key == 106): #j
            self._offscreen_renderer.camera_move_center([0, -0.1, 0], self._use_plane)
        if (key == 73 or key == 105): #i
            self._offscreen_renderer.camera_move_center([0, 0, 0.1], self._use_plane)
        if (key == 75 or key == 107): #k
            self._offscreen_renderer.camera_move_center([0, 0, -0.1], self._use_plane)

        if (key == 27): # esc
            return False
        
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections (if available)')
    args = parser.parse_args()

    d = demo2()
    d.run(args)


if __name__ == '__main__':
    main()


#anchor = self._offscreen_renderer.smpl_chart_from_spherical('smpl', 'patient', smpl_frame, self._angle, self._displacement)

#cyl_local = self._offscreen_renderer.smpl_chart_to_cylindrical('smpl', 'patient', smpl_frame, anchor.point) # requires GLOBAL coordinates
#sph_local = self._offscreen_renderer.smpl_chart_to_spherical('smpl', 'patient', smpl_frame, anchor.point) # requires GLOBAL coordinates
#print(smpl_frame)
#print(smpl_frame)
#print([angle,displacement])
#print([cyl_local.p1, cyl_local.p2])
#print([sph_local.p1, sph_local.p2])

#cv2.imshow('Depth', cv2.applyColorMap((((depth * frame_data['depth_scale']) / 7.5) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
#mesh2.visual.material.image.frombytes(self._texture_array.tobytes())

# API
'''
self._offscreen_renderer.smpl_load_uv()

self._offscreen_renderer.camera_get_pose()
self._offscreen_renderer.camera_get_projection_matrix()
self._offscreen_renderer.camera_get_transform_local()
self._offscreen_renderer.camera_get_transform_plane()
self._offscreen_renderer.camera_get_parameters()
self._offscreen_renderer.camera_adjust_parameters() # GLOBAL FRAME
self._offscreen_renderer.camera_move_center() # GLOBAL FRAME
self._offscreen_renderer.camera_solve_fov_z() # GLOBAL FRAME
self._offscreen_renderer.camera_project_points() # GLOBAL FRAME

self._offscreen_renderer.scene_render()

self._offscreen_renderer.mesh_add_smpl()
self._offscreen_renderer.mesh_add_user()
self._offscreen_renderer.mesh_set_pose()
self._offscreen_renderer.mesh_get_pose()
self._offscreen_renderer.mesh_present_smpl()
self._offscreen_renderer.mesh_present_user()
self._offscreen_renderer.mesh_remove_item()
self._offscreen_renderer.mesh_remove_group()

self._offscreen_renderer.mesh_operation_raycast() # GLOBAL FRAME
self._offscreen_renderer.mesh_operation_closest() # GLOBAL FRAME

self._offscreen_renderer.smpl_chart_create_frame() # SMPL FRAME
self._offscreen_renderer.smpl_chart_from_cylindrical() # SMPL FRAME
self._offscreen_renderer.smpl_chart_from_spherical() # SMPL FRAME
self._offscreen_renderer.smpl_chart_to_cylindrical() # GLOBAL FRAME
self._offscreen_renderer.smpl_chart_to_spherical() # GLOBAL FRAME

self._offscreen_renderer.smpl_paint_brush_solid() # SMPL FRAME
self._offscreen_renderer.smpl_paint_brush_gradient() # SMPL FRAME
self._offscreen_renderer.smpl_paint_decal_solid() # SMPL FRAME
self._offscreen_renderer.smpl_paint_decal_align_prior() # SMPL FRAME
self._offscreen_renderer.smpl_paint_clear()
self._offscreen_renderer.smpl_paint_flush()
'''

'''
def create_colormap():
    # Create colormap
    colormap = cv2.applyColorMap(np.arange(0, 256, 1, dtype=np.uint8), cv2.COLORMAP_AUTUMN)
    colormap = colormap[:, :, ::-1]
    colormap = np.dstack((colormap, 255 * np.ones(colormap.shape[0:2] + (1,), dtype=colormap.dtype)))

    mesh_colors = np.zeros((6890, 4), dtype=np.uint8)
    mesh_colors[:, 0] = 102
    mesh_colors[:, 1] = 102
    mesh_colors[:, 2] = 102
    mesh_colors[:, 3] = 255

    return (colormap, mesh_colors)
'''

#def traverse(self, viewer, delta_displacement, delta_angle):
#    displacement = self._pointer_position[0] + delta_displacement
#    angle = self._pointer_position[1] + delta_angle
#    if (angle > np.pi):
#        angle = angle - (2*np.pi)
#    if (angle < -np.pi):
#        angle = angle + (2*np.pi)
#    self._pointer_position = [displacement, angle]
# global transform
#p[:3, 3:4] = (center + wz * z)