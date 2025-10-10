
import time
import json
import cv2
import numpy as np
import torch
import trimesh
import mesh_solver


def smpl_unpack(person_list, device):
    global_orient = torch.tensor([person['smpl_params']['global_orient'] for person in person_list], dtype=torch.float32, device=device)
    body_pose = torch.tensor([person['smpl_params']['body_pose'] for person in person_list], dtype=torch.float32, device=device)
    betas = torch.tensor([person['smpl_params']['betas'] for person in person_list], dtype=torch.float32, device=device)
    smpl_params = { 'global_orient' : global_orient, 'body_pose' : body_pose, 'betas' : betas }
    camera_translation = torch.tensor([person['camera_translation'] for person in person_list], dtype=torch.float32, device=device)
    return smpl_params, camera_translation


class demo:
    def run(self):
        # Create offscreen renderer
        viewport_width = 1280
        viewport_height = 720
        fov_vertical = np.pi / 3
        fxy = mesh_solver.geometry_fov_to_f(fov_vertical, viewport_height)

        cfg_offscreen = mesh_solver.renderer_create_settings_offscreen(viewport_width, viewport_height)
        cfg_scene = mesh_solver.renderer_create_settings_scene()
        cfg_camera = mesh_solver.renderer_create_settings_camera(fxy, fxy, viewport_width // 2, viewport_height // 2)
        cfg_camera_transform = mesh_solver.renderer_create_settings_camera_transform()
        cfg_lamp = mesh_solver.renderer_create_settings_lamp()
        
        self._offscreen_renderer = mesh_solver.renderer(cfg_offscreen, cfg_scene, cfg_camera, cfg_camera_transform, cfg_lamp)

        # Load textures and UV map
        self._texture_array = mesh_solver.texture_load_image('./data/textures/f_01_alb.002_1k.png')
        self._texture_array_2 = mesh_solver.texture_load_image('./data/textures/smpl_uv_20200910.png', False)
        self._test_stamp = mesh_solver.texture_load_image('./data/textures/stamp_test.jpg')
        self._offscreen_renderer.smpl_load_uv('./data/smpl_uv.obj', self._texture_array.shape)

        # Create sample text texture
        font = mesh_solver.texture_load_font('arial.ttf', 512)
        self._test_text = mesh_solver.texture_create_multiline_text(['Sample', 'Text'], font, (255, 0, 0, 255), (255, 255, 255, 255), 1, 20)
        self._test_text = mesh_solver.texture_pad(self._test_text, 0.05, 0.1, (255, 255, 255, 255))

        # Load SMPL model
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._offscreen_renderer.smpl_load_model('data/smpl/SMPL_NEUTRAL.pkl', 10, self._device)
        print(f'Using device: {self._device}')


        
        
        
        
        # SMPL regions
        self._smpl_regions = ['body_center', 'thigh_left', 'thigh_right', 'lower_leg_left', 'lower_leg_right', 'foot_left', 'foot_right', 'head_center', 'upper_arm_left', 'upper_arm_right', 'lower_arm_left', 'lower_arm_right']
        self._smpl_region_index = 0
        self._smpl_region = self._smpl_regions[self._smpl_region_index]
        self._focus_factor = 1.25
        
        # Create UI elements
        self._cursor_radius = 0.015
        self._cursor_height = 0.04
        self._cursor_offset_increment = 0.02
        self._cursor_angle_increment = np.radians(10)
        self._cursor_mesh = trimesh.creation.cone(radius=self._cursor_radius, height=self._cursor_height)
        self._cursor_pose = np.eye(4, dtype=np.float32)
        self._cursor_offset = 0
        self._cursor_angle = 0

        # Camera translation mode
        self._camera_use_plane = True

        
        

        # Load test CameraHMR message
        with open('./test_msg.txt', 'r') as json_file:
            test_camerahmr_message = json.load(json_file)
        
        # Run inference and painting
        start = time.perf_counter()
        count = 0
        while (self._loop(test_camerahmr_message)):
            count += 1
            end = time.perf_counter()
            if (end-start > 2.0):
                print(f'FPS: ', count / (end-start))
                start = end
                count = 0

    def _loop(self, camerahmr_message):
        # SMPL params to mesh
        person_list = camerahmr_message['persons']
        smpl_params, camera_translation = smpl_unpack(person_list, self._device)
        smpl_result = self._offscreen_renderer.smpl_get_mesh(smpl_params, camera_translation)
        smpl_vertices = smpl_result.vertices[0]
        smpl_joints = smpl_result.joints[0]
        smpl_faces = smpl_result.faces
        smpl_mesh = mesh_solver.mesh_create(smpl_vertices, smpl_faces, visual=None)

        # Compute pose to set mesh upright
        # Poses convert from object to world
        smpl_mesh_pose = np.linalg.inv(mesh_solver.smpl_mesh_chart_openpose(smpl_mesh, smpl_joints).create_frame('body_center').to_pose()).T

        # Add SMPL mesh to the main scene
        smpl_mesh_id = self._offscreen_renderer.mesh_add_smpl('smpl', 'patient', smpl_mesh, smpl_joints, self._texture_array, smpl_mesh_pose)

        # Change focus region when key is pressed
        # Camera orientation is preserved
        # Cursor coordinates are reset
        smpl_next_region = self._smpl_regions[self._smpl_region_index]
        
        if (smpl_next_region != self._smpl_region):
            smpl_frame = self._offscreen_renderer.smpl_chart_create_frame(smpl_mesh_id, smpl_next_region)
            
            focus_center = mesh_solver.math_transform_points(smpl_frame.center, smpl_mesh_pose.T, inverse=False)
            focus_points = mesh_solver.math_transform_points(smpl_frame.points, smpl_mesh_pose.T, inverse=False)
            focus_distance = self._offscreen_renderer.camera_solve_fov_z(focus_center, focus_points)

            self._offscreen_renderer.camera_adjust_parameters(center=focus_center, distance=self._focus_factor * focus_distance, relative=False)
            self._cursor_offset = 0
            self._cursor_angle = 0
            self._smpl_region = smpl_next_region

        # Map current cylindrical coordinates to SMPL mesh point and face
        smpl_frame = self._offscreen_renderer.smpl_chart_create_frame(smpl_mesh_id, self._smpl_region)
        cursor_anchor = self._offscreen_renderer.smpl_chart_from_cylindrical(smpl_mesh_id, smpl_frame, self._cursor_offset, self._cursor_angle)

        # Set cursor position based on cylindrical coordinates
        # smpl_anchor.point is None when outside mesh
        local_cursor_orientation = np.vstack((np.cross(smpl_frame.up, -cursor_anchor.direction), smpl_frame.up, -cursor_anchor.direction))
        local_cursor_position = (cursor_anchor.point + self._cursor_height * cursor_anchor.direction) if (cursor_anchor.point is not None) else cursor_anchor.position
        self._cursor_pose[0:3, :3] = mesh_solver.math_transform_bearings(local_cursor_orientation, smpl_mesh_pose.T, inverse=False)
        self._cursor_pose[3:4, :3] = mesh_solver.math_transform_points(local_cursor_position, smpl_mesh_pose.T, inverse=False)
        
        # Add cursor to the main scene
        cursor_pose = self._cursor_pose.T
        cursor_mesh_id = self._offscreen_renderer.mesh_add_user('ui', 'cursor', self._cursor_mesh, cursor_pose)

        # Perform ray casting from camera to mesh
        # This will be used to paint on the mesh where the camera is looking
        camera_pose = self._offscreen_renderer.camera_get_transform_local()
        camera_position = camera_pose[:3, 3:4].T
        camera_forward = -camera_pose[:3, 2:3].T
        camera_anchor = self._offscreen_renderer.mesh_operation_raycast(smpl_mesh_id, camera_position, camera_forward)

        # If raycast did not intersect the mesh then use the closest mesh point
        if (camera_anchor.point is None):
            camera_anchor = self._offscreen_renderer.mesh_operation_closest(smpl_mesh_id, camera_position)

        # Paint SMPL mesh
        # Paint decal at cursor position
        if (cursor_anchor.point is not None):
            # Fix loose degree of freedom about face normal, required to maintain consistent orientation
            decal_align_prior = self._offscreen_renderer.smpl_paint_decal_align_prior(smpl_mesh_id, cursor_anchor, smpl_frame.up, smpl_frame.front)
            self._offscreen_renderer.smpl_paint_decal_solid(smpl_mesh_id, cursor_anchor, self._test_text, decal_align_prior, 0, 10000 * 2, double_cover_test=False, fill_test=0.25)
        # Paint circular gradient at camera mesh intersection/closest
        if (camera_anchor.point is not None):
            self._offscreen_renderer.smpl_paint_brush_gradient(smpl_mesh_id, camera_anchor, 0.01, np.array([255, 0, 0, 255], dtype=np.uint8), np.array([255, 255, 0, 255], dtype=np.uint8), 0.33, fill_test=0.25)
            # Solid color option
            #self._offscreen_renderer.smpl_paint_brush_solid(smpl_mesh_id, camera_anchor, 0.02, np.array([0, 255, 0, 255], dtype=np.uint8), 0.25)
        
        # Finalize SMPL painting
        # Compute painted texture
        self._offscreen_renderer.smpl_paint_flush(smpl_mesh_id)
        # Remove painting for next frame (comment out to keep paintings across frames)
        self._offscreen_renderer.smpl_paint_clear(smpl_mesh_id)

        # Finalize mesh processing
        self._offscreen_renderer.mesh_present_smpl(smpl_mesh_id)
        self._offscreen_renderer.mesh_present_user(cursor_mesh_id)

        # Render
        color, depth = self._offscreen_renderer.scene_render()

        # Render focused joints
        color = color.copy()
        world_points = mesh_solver.math_transform_points(smpl_frame.points, smpl_mesh_pose.T, inverse=False)
        image_points, local_points, camera_points = self._offscreen_renderer.camera_project_points(world_points, convention=(1, -1, -1))

        for i in range(0, image_points.shape[0]):
            if (local_points[i, 2] > 0):
                center = (int(image_points[i, 0]), int(image_points[i, 1]))
                color = cv2.circle(color, center, 3, [255, 0, 255], -1)

        # Show rendered image
        cv2.imshow('offscreen test', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        # Process keyboard input
        key = cv2.waitKey(1) & 0xFF

        if (key == 49): # 1:
            self._smpl_region_index = (self._smpl_region_index + 1) % len(self._smpl_regions)
 
        if (key == 50): # 2:
            self._cursor_offset += self._cursor_offset_increment
        if (key == 51): # 3
            self._cursor_offset -= self._cursor_offset_increment
        if (key == 52): # 4 
            self._cursor_angle += self._cursor_angle_increment
        if (key == 53): # 5
            self._cursor_angle -= self._cursor_angle_increment

        if (key == 68 or key == 100): # d
            self._offscreen_renderer.camera_adjust_parameters(yaw=10, relative=True)
        if (key == 65 or key == 97): # a
            self._offscreen_renderer.camera_adjust_parameters(yaw=-10, relative=True)
        if (key == 87 or key == 119): # w
            self._offscreen_renderer.camera_adjust_parameters(pitch=-10, relative=True)
        if (key == 83 or key == 115): # s
            self._offscreen_renderer.camera_adjust_parameters(pitch=10, relative=True)
        if (key == 82 or key == 114): #r
            self._offscreen_renderer.camera_adjust_parameters(distance=-0.1, relative=True)
        if (key == 70 or key == 102): #f
            self._offscreen_renderer.camera_adjust_parameters(distance=0.1, relative=True)
        if (key == 78 or key == 110): #n
            self._offscreen_renderer.camera_move_center([-0.1, 0, 0], plane=self._camera_use_plane)
        if (key == 77 or key == 109): #m
            self._offscreen_renderer.camera_move_center([0.1, 0, 0], plane=self._camera_use_plane)
        if (key == 85 or key == 117): #u
            self._offscreen_renderer.camera_move_center([0, 0.1, 0], plane=self._camera_use_plane)
        if (key == 74 or key == 106): #j
            self._offscreen_renderer.camera_move_center([0, -0.1, 0], plane=self._camera_use_plane)
        if (key == 73 or key == 105): #i
            self._offscreen_renderer.camera_move_center([0, 0, -0.1], plane=self._camera_use_plane)
        if (key == 75 or key == 107): #k
            self._offscreen_renderer.camera_move_center([0, 0, 0.1], plane=self._camera_use_plane)

        if (key == 27): # esc
            return False
        
        return True


def main():
    demo().run()


if __name__ == '__main__':
    main()

