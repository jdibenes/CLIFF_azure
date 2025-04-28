
import numpy as np
import pyrender
import trimesh
import collections


class vsv:
    def __init__(self, camera_vertical_fov, viewport_width, viewport_height):
        self._renderer = pyrender.OffscreenRenderer(viewport_width, viewport_height)
        self._scene    = pyrender.Scene()
        self._camera   = pyrender.PerspectiveCamera(yfov=camera_vertical_fov)
        self._light    = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

        self._camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.float32).reshape((4, 4))
        self._camera_fy   = viewport_height / (2 * np.tan(0.5 * camera_vertical_fov))
        self._camera_fx   = self._camera_fy
        self._camera_cy   = viewport_height / 2
        self._camera_cx   = viewport_width / 2
        self._camera_yfov = camera_vertical_fov
        self._camera_xfov = 2*np.arctan(viewport_width / (2*self._camera_fx))
        
        self._node_camera = self._scene.add(self._camera, pose=self._camera_pose)
        self._node_light  = self._scene.add(self._light, pose=self._camera_pose)
        self._node_mesh   = None

    def set_smpl_mesh(self, vertices, faces, joints, segmentation):
        if (self._node_mesh is not None):
            self._scene.remove_node(self._node_mesh)
            self._node_mesh = None

        self._mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        self._mesh_centroid = self._mesh.bounding_box.centroid.copy().reshape((1, 3))
        self._mesh.vertices -= self._mesh_centroid
        self._joints = (joints - self._mesh_centroid).T
        self._segmentation = segmentation

        self._node_mesh = self._scene.add(pyrender.Mesh.from_trimesh(self._mesh))

    def get_mesh_colors(self):
        return self._mesh.visual.vertex_colors
    
    def set_mesh_colors(self, colors):
        self._mesh.visual.vertex_colors = colors

    def reload_mesh(self):
        self._scene.remove_node(self._node_mesh)
        self._node_mesh = self._scene.add(pyrender.Mesh.from_trimesh(self._mesh))

    def render(self):
        self._color, self._depth = self._renderer.render(self._scene)
        self._color = self._color.copy()

        return (self._color, self._depth)
    
    def get_camera_pose(self):
        return self._camera_pose
    
    def set_camera_pose(self, camera_pose):
        self._camera_pose = camera_pose

        self._scene.set_pose(self._node_camera, self._camera_pose)
        self._scene.set_pose(self._node_light, self._camera_pose)

    def project_points(self, points):
        r = self._camera_pose[:3, :3]
        t = self._camera_pose[:3, 3:4]

        camera_points     = r.T @ points - r.T @ t
        normalized_points = camera_points[0:2, :] / -camera_points[2, :]

        image_x =  normalized_points[0, :] * self._camera_fx + self._camera_cx
        image_y = -normalized_points[1, :] * self._camera_fy + self._camera_cy

        return np.vstack((image_x, image_y))

    # 

    def get_joint(self, index):
        return self._joints[:, index].reshape((3, -1))
    
    def cross(self, a, b):
        return np.cross(a.reshape((-1)), b.reshape((-1))).reshape((3, -1))
    
    def normalize(self, a):
        return a / np.linalg.norm(a)

    def focus_solver(self, x, y, z, center, points):
        p = np.eye(4, 4, dtype=np.float32)

        p[:3, 0:1] = x
        p[:3, 1:2] = y
        p[:3, 2:3] = z

        dp = (points - center)
        dx = np.abs(x.T @ dp)
        dy = np.abs(y.T @ dp)
        dz = -z.T @ dp
        wx = (dx / np.tan(self._camera_xfov / 2)) - dz
        wy = (dy / np.tan(self._camera_yfov / 2)) - dz
        wz = np.max(np.hstack((wy, wx)))

        p[:3, 3:4] = (center + wz * z)

        print(f'focus_solver determinant={np.linalg.det(p)}')

        return [p, center, wz]
    
    def face_solver(self, origin, direction):
        point, rid, tid = self._mesh.ray.intersects_location(origin.T, direction.T, multiple_hits=False)
        if (len(rid) <= 0):
            return (None, None, None)
        point = point.T
        face_index = tid[0]
        vertex_indices = self._mesh.faces.view(np.ndarray)[face_index].tolist()
        vertices = self._mesh.vertices.view(np.ndarray)[vertex_indices, :].T
        distances = np.linalg.norm(point - vertices, axis=0)
        snap_index = np.argmin(distances)

        return point, face_index, vertex_indices[snap_index]
    
    def select_vertices(self, origin_vertex_index, radius, level):
        vertices  = self._mesh.vertices.view(np.ndarray)
        neighbors = self._mesh.vertex_neighbors
        buffer    = collections.deque()
        distances = {origin_vertex_index : 0}

        buffer.append((origin_vertex_index, 0, 0))

        while (len(buffer) > 0):
            vertex_index, vertex_distance, vertex_level = buffer.popleft()
            vertex_xyz = vertices[vertex_index, :]

            for neighbor_index in neighbors[vertex_index]:
                neighbor_xyz = vertices[neighbor_index, :]
                neighbor_distance = vertex_distance + np.linalg.norm(neighbor_xyz - vertex_xyz)
                neighbor_level = vertex_level + 1
                if ((neighbor_distance <= radius) and (neighbor_level <= level) and (neighbor_distance < distances.get(neighbor_index, np.Inf))):          
                    buffer.append((neighbor_index, neighbor_distance, neighbor_level))
                    distances[neighbor_index] = neighbor_distance
        
        return distances
    
    def select_complete_faces(self, vertex_indices):
        vertex_faces = self._mesh.vertex_faces
        faces = self._mesh.faces.view(np.ndarray)
        face_indices_complete = set()
        vertex_indices_complete = set()
        vertex_indices_seen = set()

        for vertex_index in vertex_indices:
            if (vertex_index in vertex_indices_seen):
                continue
            for face_index in vertex_faces[vertex_index, :]:
                if (face_index >= 0):
                    face_vertices = faces[face_index].tolist()
                    vertex_indices_seen.update(face_vertices)
                    keep = all([face_vertex in vertex_indices for face_vertex in face_vertices])
                    if (keep):
                        face_indices_complete.add(face_index)
                        vertex_indices_complete.update(face_vertices)
        
        return list(face_indices_complete), list(vertex_indices_complete)

    def focus_foot(self, bigtoe, smalltoe, ankle, heel):
        left  = self.cross(ankle - heel, bigtoe - ankle)
        front = self.cross(left, ankle - smalltoe)
        up    = self.cross(front, left)

        left  = self.normalize(left)
        front = self.normalize(front)
        up    = self.normalize(up)

        center = (ankle + bigtoe) * 0.5
        points = np.hstack((bigtoe, smalltoe, ankle, heel))

        return self.focus_solver(left, up, front, center, points)

    def focus_lower_leg(self, bigtoe, ankle, knee):
        up    = knee - ankle
        left  = self.cross(up, bigtoe - ankle)
        front = self.cross(left, up)

        up    = self.normalize(up)
        left  = self.normalize(left)
        front = self.normalize(front)

        center = (ankle + knee) * 0.5
        points = np.hstack((bigtoe, ankle, knee))

        return self.focus_solver(left, up, front, center, points)
    
    def focus_thigh(self, ankle, knee, hip):
        up    = hip - knee
        left  = self.cross(up, knee - ankle)
        front = self.cross(left, up)

        up    = self.normalize(up)
        left  = self.normalize(left)
        front = self.normalize(front)

        center = (hip + knee) * 0.5
        points = np.hstack((knee, hip))

        return self.focus_solver(left, up, front, center, points)
    
    def focus_body(self, lhip, mhip, rhip, neck):
        left  = lhip - rhip
        front = self.cross(left, neck - mhip)
        up    = self.cross(front, left)

        up    = self.normalize(up)
        left  = self.normalize(left)
        front = self.normalize(front)

        center = (mhip + neck) * 0.5
        points = np.hstack((lhip, mhip, rhip, neck))
                
        return self.focus_solver(left, up, front, center, points)
    
    def focus_head(self, lear, rear, neck, nose):
        left  = lear - rear
        up    = lear - neck
        front = self.cross(left, up)
        up    = self.cross(front, left)

        left  = self.normalize(left)
        up    = self.normalize(up)
        front = self.normalize(front)

        center = (nose + lear + rear) / 3
        points = np.hstack((lear, rear, neck, nose))

        return self.focus_solver(left, up, front, center, points)
    
    def focus_upper_arm(self, wrist, elbow, shoulder):
        up    = shoulder - elbow
        left  = self.cross(up, wrist - elbow)
        front = self.cross(left, up)

        left  = self.normalize(left)
        up    = self.normalize(up)
        front = self.normalize(front)

        center = (elbow + shoulder) * 0.5
        points = np.hstack((shoulder, elbow))        

        return self.focus_solver(left, up, front, center, points)

    def focus_left_foot(self):
        bigtoe   = self.get_joint(19)
        smalltoe = self.get_joint(20)
        ankle    = self.get_joint(14)
        heel     = self.get_joint(21)

        return self.focus_foot(bigtoe, smalltoe, ankle, heel)
    
    def focus_right_foot(self):
        bigtoe   = self.get_joint(22)
        smalltoe = self.get_joint(23)
        ankle    = self.get_joint(11)
        heel     = self.get_joint(24)

        return self.focus_foot(bigtoe, smalltoe, ankle, heel)
    
    def focus_left_lower_leg(self):
        bigtoe = self.get_joint(19)
        ankle  = self.get_joint(14)
        knee   = self.get_joint(13)

        return self.focus_lower_leg(bigtoe, ankle, knee)

    def focus_right_lower_leg(self):
        bigtoe = self.get_joint(22)
        ankle  = self.get_joint(11)
        knee   = self.get_joint(10)

        return self.focus_lower_leg(bigtoe, ankle, knee)
    
    def focus_left_thigh(self):
        ankle = self.get_joint(14)
        knee  = self.get_joint(13)
        hip   = self.get_joint(12)

        return self.focus_thigh(ankle, knee, hip)
    
    def focus_right_thigh(self):
        ankle = self.get_joint(11)
        knee  = self.get_joint(10)
        hip   = self.get_joint(9)

        return self.focus_thigh(ankle, knee, hip)
    
    def focus_center_body(self):
        lhip = self.get_joint(12)
        mhip = self.get_joint(8)
        rhip = self.get_joint(9)
        neck = self.get_joint(1)

        return self.focus_body(lhip, mhip, rhip, neck)
    
    def focus_center_head(self):
        lear = self.get_joint(18)
        rear = self.get_joint(17)
        neck = self.get_joint(1)
        nose = self.get_joint(0)

        return self.focus_head(lear, rear, neck, nose)
    
    def focus_left_upper_arm(self):
        wrist    = self.get_joint(7)
        elbow    = self.get_joint(6)
        shoulder = self.get_joint(5)

        return self.focus_upper_arm(wrist, elbow, shoulder)
    
    def focus_right_upper_arm(self):
        wrist    = self.get_joint(4)
        elbow    = self.get_joint(3)
        shoulder = self.get_joint(2)

        return self.focus_upper_arm(wrist, elbow, shoulder)
    






    def focus_leg(self, index_ankle, index_knee, index_hip, index_bigtoe, margin):
        ankle  = self.get_joint(index_ankle)
        knee   = self.get_joint(index_knee)
        hip    = self.get_joint(index_hip)
        bigtoe = self.get_joint(index_bigtoe)

        up_thigh        = hip - knee
        up_lower_leg    = knee - ankle
        left            = self.cross(up_thigh, up_lower_leg)
        front_thigh     = self.cross(left, up_thigh)
        front_lower_leg = self.cross(left, up_lower_leg)

        left  = self.normalize(left)
        front = self.normalize(self.normalize(front_thigh) + self.normalize(front_lower_leg))
        up    = self.normalize(self.cross(front, left))
        front = self.normalize(self.cross(left, up))

        center = knee
        points = np.hstack((ankle, knee, hip, bigtoe))

        return self.focus_solver(left, up, front, center, points, margin)
    
    def focus_right_leg(self, margin=1.0):
        return self.focus_leg(11, 10, 9, 22, margin)

    def focus_left_leg(self, margin=1.0):
        return self.focus_leg(14, 13, 12, 19, margin)
    
    


    











        



    '''
          REar-REye--------|--------LEye-LEar
                         Nose
          RShoulder------Neck-------LShoulder
             RElbow        |        LElbow
             RWrist        |        LWrist
                           |
               RHip-----MidHip------LHip
              RKnee                 LKnee
             RAnkle                 LAnkle
    RSmallToe RHeel RBigToe LBigToe LHeel LSmallToe
    '''
    '''
    'OP Nose', # 0
    'OP Neck', # 1
    'OP RShoulder', # 2
    'OP RElbow', # 3
    'OP RWrist', # 4
    'OP LShoulder', # 5
    'OP LElbow', # 6
    'OP LWrist', # 7
    'OP MidHip', # 8
    'OP RHip', # 9
    'OP RKnee', # 10
    'OP RAnkle', # 11
    'OP LHip', # 12
    'OP LKnee', # 13
    'OP LAnkle', # 14
    'OP REye', # 15
    'OP LEye', # 16
    'OP REar', # 17
    'OP LEar', # 18
    'OP LBigToe', # 19
    'OP LSmallToe', # 20
    'OP LHeel', # 21
    'OP RBigToe', # 22
    'OP RSmallToe', # 23
    'OP RHeel', # 24
    '''  

# NO: small toe, heel
# foot side in: (heel-ankle) X (bigtoe-ankle) !
# lower leg side in: (bigtoe-ankle) x (knee-ankle)
# thigh side in: (ankle-knee) X (knee-hip)
# body front: (Rhip-Lhip) X (Lhip-neck)
# face front: (LEar-REar) X (REar-neck)
# upper arm side in: (wrist-elbow) X (shoulder-elbow)
# lower arm: no
# hand: no