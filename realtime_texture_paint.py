
import cv2
import argparse
import numpy as np
import time

import torch

from pytorch3d import transforms

from models.smpl import SMPL
from common import constants

from losses import *
from smplify import SMPLify

from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from common.utils import strip_prefix_if_present, cam_crop2full
# from common.mocap_dataset import MocapDataset  # not used in this live demo

import mesh_solver
import visualizer


def preprocess_frame(frame, target_size=(224, 224)):
    """
    Preprocess the input frame (BGR) to a normalized tensor.
    Returns norm_img as a tensor of shape (1, 3, H, W) in float32.
    """
    resized = cv2.resize(frame, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
    norm_img = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    return norm_img


def create_model(img_shape):
    # Device selection
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    # Load the pretrained CLIFF model.
    cliff = eval("cliff_hr48")
    cliff_model = cliff('./data/smpl_mean_params.npz').to(device)
    state_dict = torch.load('./data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt')['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    cliff_model.load_state_dict(state_dict, strict=True)
    cliff_model.eval()

    img_h, img_w = img_shape[:2]
    focal_length = torch.tensor([500.0], device=device, dtype=torch.float32)
    center = torch.tensor([[img_w / 2.0, img_h / 2.0]], device=device, dtype=torch.float32)
    scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    b = scale * 200
    bbox_info = torch.stack([center[:, 0] - img_w / 2.0, center[:, 1] - img_h / 2.0, b], dim=-1)
    bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8
    bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)
    full_img_shape = torch.stack((torch.tensor([img_h], device=device, dtype=torch.float32), torch.tensor([img_w], device=device, dtype=torch.float32)), dim=-1)

    # Load the SMPL model.
    smpl = SMPL(constants.SMPL_MODEL_DIR, batch_size=1).to(device)

    return (device, cliff_model, bbox_info, smpl)


# mesh_a = 6890 vertices 13776 faces 6890 uvs
# mesh_b = 7576 vertices 13776 faces 7576 uvs
# same faces, duplicated vertices for vertices with multiple uv's

class demo:
    def run(self, args):
        # Set up the webcam and offscreen pyrender renderer.
        #cap = visualizer.camera_opencv()
        cap = visualizer.camera_null()
        cap.open()

        # Model stuff
        frame_data = cap.read()
        if ('depth_intrinsics' in frame_data):
            print(f"depth_intrinsics={frame_data['depth_intrinsics']}")
            print(f"depth_scale={frame_data['depth_scale']}")
        if ('color_intrinsics' in frame_data):
            print(f"color_intrinsics={frame_data['color_intrinsics']}")

        frame = frame_data['color']

        cfg_offscreen = mesh_solver.renderer_create_settings_offscreen(1280, 720)
        cfg_scene = mesh_solver.renderer_create_settings_scene()
        f = 1/(np.tan((np.pi / 3) / 2) / (720 / 2))
        cfg_camera = mesh_solver.renderer_create_settings_camera(f, f, 640, 360)
        cfg_camera_transform = mesh_solver.renderer_create_settings_camera_transform()
        cfg_lamp = mesh_solver.renderer_create_settings_lamp()
        self._offscreen_renderer = mesh_solver.renderer(cfg_offscreen, cfg_scene, cfg_camera, cfg_camera_transform, cfg_lamp)
        self._offscreen_renderer.load_uv('./data/smpl_uv.obj', [1024, 1024])

        tex_filename = './data/textures/f_01_alb.002_1k.png'
        #tex_filename = './data/textures/smpl_uv_20200910.png'
        self._texture_array = mesh_solver.texture_load_image(tex_filename)
        self._test_stamp = mesh_solver.texture_load_image('./data/textures/stamp_test.jpg')
        font = mesh_solver.texture_load_font('arial.ttf', 512)
        self._test_text = mesh_solver.texture_create_multiline_text(['Sample', 'Text'], font, (255, 0, 0, 255), (255, 255, 255, 255), 1, 20)
        self._test_text = mesh_solver.texture_pad(self._test_text, 0.05, 0.1, (255, 255, 255, 255))
        
        self._device, self._cliff_model, self._bbox_info, self._smpl = create_model(frame.shape)

        # Initialize visualization utilities
        viewport_width, viewport_height = 1280, 720
        camera_yfov = np.pi / 3
        joint_solver  = visualizer.solver(camera_yfov, viewport_width, viewport_height)
        scene_control = visualizer.scene_manager(camera_yfov)

        self._cap = cap
        
        self._joint_solver = joint_solver
        self._scene_control = scene_control
        self._regions = [
            #joint_solver.focus_center_whole,
            joint_solver.focus_center_body,
            joint_solver.focus_right_lower_leg,
            joint_solver.focus_left_lower_leg,
        ]        

        # Start with state 1 and focus full body.
        self._current_state = 0
        self._current_region = 0
        self._next_region = 0
        self._pointer_position = [0, 0]
        self._pointer_size = 0.05

        di = 0.02
        da = (10/180) * np.pi
        dr = 0.01

        key_handlers = {
            '2': (self.traverse, [di, 0]),
            '3': (self.traverse, [-di, 0]),
            '4': (self.traverse, [0, da]),
            '5': (self.traverse, [0, -da]),
            '6': (self.adjust_pointer, [dr]),
            '7': (self.adjust_pointer, [-dr]),
            '8': (self.advance_target, []),
        }

        viewer_flags = {'vsv.hook_on_begin' : (self.animate, [])}

        print("Starting real-time inference.")
        print("Press 'q' to exit.")
        print("Press 2/3 to move pointer.")
        print("Press 4/5 to rotate pointer.")
        print("Press 6/7 to adjust pointer size.")
        print("Press 8 to cycle through full body / right leg / left leg.")

        visualizer.viewer(self._scene_control._scene,
                    viewport_size=(viewport_width, viewport_height),
                    viewer_flags=viewer_flags,
                    use_raymond_lighting=True,
                    registered_keys=key_handlers)

        cap.close()

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

        mesh = mesh_solver.mesh_create(vertices, faces)

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
        #print([point, face_index, vertex_index])
        if (point is not None):
            distances = self._joint_solver.select_vertices(vertex_index, np.Inf, 3)
            selected_indices = distances.keys()
            filtered_faces, filtered_indices = self._joint_solver.select_complete_faces(selected_indices)
            if (len(filtered_indices) > 0):
                selected_indices = filtered_indices
        else:
            distances = None
            selected_indices = None
        
        self._scene_control.set_smpl_mesh(mesh)
        self._scene_control.clear_group('arrows')

        camera_pose[:3, 3:4] = (focus_center + 1.2 * axis_displacement * camera_pose[:3, 2:3])
        self._scene_control.set_camera_pose(camera_pose)
        
        use_offscreen = True
        use_plane = True
        di = 0.02
        da = (10/180) * np.pi

        self._offscreen_renderer.smpl_mesh_set('mesh_test', mesh.vertices, joints.T, mesh.faces, self._texture_array)
        cone = mesh_solver.mesh_create_cone(0.015, 0.04, 10)
        sphere = mesh_solver.mesh_create_sphere(0.002)
        I3 = np.eye(3, dtype=np.float32)
        pose_cone =  np.eye(4, dtype=np.float32)
        pose_sphere = np.eye(4, dtype=np.float32)
        
        while (use_offscreen):
            #vertex_index = mesh.faces[face_index][snap_index]
            smpl_frame = self._offscreen_renderer.smpl_chart_create_frame('mesh_test', 'body_center')
            wz = self._offscreen_renderer.camera_solve_fov_z(smpl_frame.center, smpl_frame.points)
            #self._offscreen_renderer.camera_adjust_parameters(center=smpl_frame.center, distance=wz, relative=False)
            anchor = self._offscreen_renderer.smpl_chart_from_cylindrical('mesh_test', smpl_frame, displacement, angle)
            #anchor = self._offscreen_renderer.smpl_chart_from_spherical('mesh_test', smpl_frame, angle, displacement)
            #print(smpl_frame)
            #print(smpl_frame)
            #print([angle,displacement])
            #cyl_local = self._offscreen_renderer.smpl_chart_to_cylindrical('mesh_test', smpl_frame, anchor.point)
            #print([cyl_local.p1, cyl_local.p2])
            #sph_local = self._offscreen_renderer.smpl_chart_to_spherical('mesh_test', smpl_frame, anchor.point)
            #print([sph_local.p1, sph_local.p2])

            arrow_r = mesh_solver.geometry_solve_basis(I3[1:2, :], I3[2:3, :], smpl_frame.up, -anchor.direction)
            pose_cone[:3, :3] = arrow_r.T
            pose_cone[:3, 3:4] = (anchor.point + 0.04 * anchor.direction).T
            

            #print()

            pose_sphere[:3, 3:4] = mesh_solver.mesh_closest(mesh, pose_cone[:3, 3:4].T)[0].T

            self._offscreen_renderer.mesh_add('cursor', cone, pose_cone)
            self._offscreen_renderer.mesh_add('closest', sphere, pose_sphere)

            #self._offscreen_renderer.smpl_paint_brush_solid('mesh_test', anchor, 0.02, np.array([0, 255, 0, 255], dtype=np.uint8))
            #self._offscreen_renderer.smpl_paint_brush_gradient('mesh_test', anchor, 0.01, np.array([255, 0, 0, 255], dtype=np.uint8), np.array([255, 255, 0, 255], dtype=np.uint8), 0.33)
            #self._offscreen_renderer.smpl_paint_decal_solid('mesh_test', anchor, self._test_stamp, 0, 10000 * 2)
            align_prior = self._offscreen_renderer.smpl_paint_decal_align_prior('mesh_test', anchor, smpl_frame.up)
            self._offscreen_renderer.smpl_paint_decal_solid('mesh_test', anchor, self._test_text, align_prior, 0, 10000 * 2, timeout=0.1)
            self._offscreen_renderer.smpl_paint_flush('mesh_test')
            self._offscreen_renderer.smpl_paint_clear('mesh_test')

            color, _ = self._offscreen_renderer.scene_render()

            cv2.imshow('offscreen test', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(0) & 0xFF
            if (key == 50): # 2
                displacement += di
            if (key == 51): # 3
                displacement -= di
            if (key == 52): # 4
                angle += da
            if (key == 53): # 5
                angle -= da

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
            if (key == 85 or key == 117): #u
                self._offscreen_renderer.camera_move_center([0, 0.1, 0], use_plane)
            if (key == 74 or key == 106): #j
                self._offscreen_renderer.camera_move_center([0, -0.1, 0], use_plane)
            if (key == 73 or key == 105): #i
                self._offscreen_renderer.camera_move_center([0, 0, 0.1], use_plane)
            if (key == 75 or key == 107): #k
                self._offscreen_renderer.camera_move_center([0, 0, -0.1], use_plane)
            if (key == 78 or key == 110): #n
                self._offscreen_renderer.camera_move_center([0.1, 0, 0], use_plane)
            if (key == 77 or key == 109): #m
                self._offscreen_renderer.camera_move_center([-0.1, 0, 0], use_plane)
            if (key == 27): # esc
                break

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




#self._offscreen_renderer.group_item_add('smpl_meshes', 'mesh_test', mesh_solver.mesh_to_renderer(mesh2))
#start_time = time.perf_counter()
#mesh2 = mesh_solver.mesh_expand(mesh, self._uv_transform, self._mesh_faces_b, self._mesh_visuals)
#end_time = time.perf_counter()
#print(f'MESH2 time {end_time-start_time}')

#start_time = time.perf_counter()
#self._mesh_painter.brush_create_gradient(0, 0.01, np.array([255, 0, 0, 255], dtype=np.uint8), np.array([255, 255, 0, 255], dtype=np.uint8), 0.33, 0)
#self._mesh_painter.brush_create_solid(1, 0.02, np.array([0, 255, 0, 255], dtype=np.uint8), 0)
#self._mesh_painter.task_create_paint_brush(0, mesh, mesh2, face_index, point.T, [1, 0])
#self._mesh_painter.task_execute(0, 0.033)
#end_time = time.perf_counter()
#print(f'paint solid time {end_time-start_time} proc')

#align_prior = np.array([[0, 1, 0]], dtype=np.float32)
#align_normal = mesh.face_normals[face_index:(face_index+1), :]
#print(align_prior @ align_normal.T)
#align_prior = align_prior - (align_normal @ align_prior.T) * align_normal
#align_prior = align_prior / np.linalg.norm(align_prior)

#start_time = time.perf_counter()
#self._mesh_painter.decal_create_solid(0, align_prior, 0, 10000 * 2, 0, 0, False)
#self._mesh_painter.task_create_paint_decal(0, mesh, mesh2, face_index, point.T, 0)
#self._mesh_painter.task_execute(0, 0.033)
#end_time = time.perf_counter()
#print(f'paint image time {end_time-start_time} proc')

#start_time = time.perf_counter()
#self._mesh_painter.flush()
#self._mesh_painter.layer_clear(0)
#end_time = time.perf_counter()
#print(f'flush image time {end_time-start_time} proc')

#self._uv_transform, self._mesh_faces_b, self._mesh_uv_b = mesh_solver.texture_load_uv('./data/smpl_uv.obj')
#self._bg_array = self._texture_array.copy()
#self._mesh_visuals = mesh_solver.texture_create_visual(self._mesh_uv_b, self._texture_array)
#self._mesh_uvx_b = mesh_solver.texture_uv_to_uvx(self._mesh_uv_b.copy(), self._texture_array.shape)
#f = 700
#self._mesh_painter = mesh_solver.renderer_mesh_paint(self._mesh_uvx_b, self._texture_array, self._uv_transform, self._bg_array)
#self._mesh_painter.texture_attach(0, self._test_stamp)
#self._mesh_painter.layer_create(0)
#self._mesh_painter.layer_enable(0, True)
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
#mesh2.visual.material.image.frombytes(self._texture_array.tobytes())

        #vertices_b = mesh.vertices[self._uv_transform, :]
        #faces_b = self._mesh_faces_b
        #self._overlay_array = np.zeros_like(self._texture_array)

        #mob = mesh_solver.paint_brush_gradient(0.01, np.array([255, 0, 0, 255], dtype=np.uint8), np.array([255, 255, 0, 255], dtype=np.uint8), 0.33, self._overlay_array)
        #mob2 = mesh_solver.paint_brush_solid(0.02, np.array([0, 255, 0, 255], dtype=np.uint8), self._overlay_array)
        #mno = mesh_solver.painter_create_brush(mesh, mesh2, self._mesh_uvx_b, self._uv_transform, face_index, point.T, [mob2.paint, mob.paint])
        #mno.invoke_timeslice(0.010)

        #mob = mesh_solver.paint_decal_solid(align_prior, 0, 10000 * 2, self._test_stamp, self._overlay_array)
        #mno = mesh_solver.painter_create_decal(mesh, mesh2, self._mesh_uvx_b, self._uv_transform, face_index, point.T, mob.paint)
        #mno.invoke_timeslice(0.200)
        #alpha = self._overlay_array[:, :, 3:4] / 255
        #self._texture_array[:, :, :] = (1-alpha) * self._bg_array + alpha * self._overlay_array

        
        