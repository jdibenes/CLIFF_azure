
import numpy as np
import trimesh.exchange.obj
import math
import cv2


#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

def load_uv_transform(filename_uv):
    with open(filename_uv, 'r') as obj_file:
        obj_mesh_a = trimesh.exchange.obj.load_obj(obj_file, maintain_order=True)
    with open(filename_uv, 'r') as obj_file:
        obj_mesh_b = trimesh.exchange.obj.load_obj(obj_file)
    mesh_vertices_b = obj_mesh_b['geometry'][filename_uv]['vertices']
    mesh_faces_a = obj_mesh_a['geometry'][filename_uv]['faces']
    mesh_faces_b = obj_mesh_b['geometry'][filename_uv]['faces']
    mesh_uv_b = obj_mesh_b['geometry'][filename_uv]['visual'].uv
    uv_transform = np.zeros(mesh_vertices_b.shape[0], dtype=np.int64)
    for face_index in range(0, mesh_faces_b.shape[0]):
        for vertex_index in range(0, 3):
            uv_transform[mesh_faces_b[face_index, vertex_index]] = mesh_faces_a[face_index, vertex_index]
    return (uv_transform, mesh_faces_b, mesh_uv_b)


def barycentric_create(simplex):
    return np.linalg.inv(np.vstack((simplex[0:2, :] - simplex[2:3, :], simplex[2:3, :])))


def barycentric_encode(vertex, transform):
    weights = (vertex @ transform)[0, :]
    weights[2] = 1 - weights[0] - weights[1]
    return weights


def barycentric_decode(weights, simplex):
    return (weights[0] * simplex[0:1, :]) + (weights[1] * simplex[1:2, :]) + (weights[2] * simplex[2:3, :])


def mesh_raycast(mesh, origin, direction):
    point, rid, tid = mesh.ray.intersects_location(origin, direction, multiple_hits=False)
    return (point, tid[0]) if (len(rid) > 0) else (None, None)


def mesh_operation_uv(simplex_uvx, callback, tolerance=0):
    # uvx : [u * (w - 1), (1 - v) * (h - 1), 1]
    bt = barycentric_create(simplex_uvx)
    vertex = np.ones((1, 3), dtype=simplex_uvx.dtype)
    zero = -tolerance

    left = math.floor(np.min(simplex_uvx[:, 0]))
    right = math.ceil(np.max(simplex_uvx[:, 0]))
    top = math.floor(np.min(simplex_uvx[:, 1]))
    bottom = math.ceil(np.max(simplex_uvx[:, 1]))

    for y in range(top, bottom):
        for x in range(left, right):
            vertex[0, 0] = x
            vertex[0, 1] = y
            bw = barycentric_encode(vertex, bt)
            if ((bw[0] >= zero) and (bw[1] >= zero) and (bw[2] >= zero)):
                callback(vertex, bw)


class mesh_neighborhood_builder:
    def __init__(self, mesh):
        self._mesh = mesh
        self._mesh_faces = self._mesh.faces.view(np.ndarray)
        self._mesh_vertex_faces = self._mesh.vertex_faces
        self._seen_face = set()
        self._seen_vertex = set()
        self._iterations = 0

    def fetch(self, faces):
        result = set()
        self._seen_face.update(faces)
        for face_anchor in faces:
            for vertex_index in self._mesh_faces[face_anchor]:
                if (vertex_index not in self._seen_vertex):
                    self._seen_vertex.add(vertex_index)
                    for face_index in self._mesh_vertex_faces[vertex_index]:
                        if (face_index < 0):
                            break
                        if (face_index not in self._seen_face):
                            result.add(face_index)
        self._iterations += 1
        return result
    
    def level(self):
        return self._iterations


class mesh_neighborhood_operator:
    def __init__(self, mesh, faces, callback):
        self._mnb = mesh_neighborhood_builder(mesh)
        self._faces = faces
        self._expand_faces = set()
        self._callback = callback
        self._done = False

    def invoke(self, max_iterations):
        for _ in range(0, max_iterations):
            if (len(self._expand_faces) > 0):
                self._faces = self._mnb.fetch(self._expand_faces)
                self._expand_faces.clear()
            for face_anchor in self._faces:
                if (self._callback(face_anchor, self._mnb.level())):
                    self._expand_faces.add(face_anchor)
            if (len(self._expand_faces) < 1):
                self._done = True
                break
    
    def done(self):
        return self._done


class mesh_operator_paint_solid:
    def __init__(self, mesh_vertices, mesh_faces, mesh_uvx, point, size, color, render_buffer, tolerance=0):
        self._mesh_vertices = mesh_vertices
        self._mesh_faces = mesh_faces
        self._mesh_uvx = mesh_uvx
        self._origin = point
        self._size = size
        self._color = color
        self._render_buffer = render_buffer
        self._tolerance = tolerance
        self._done_faces = set()

    def paint(self, face_index, level):
        if (face_index in self._done_faces):
            return
        self._done_faces.add(face_index)
        vertex_indices = self._mesh_faces[face_index]
        self._simplex_3d = self._mesh_vertices[vertex_indices, :]
        self._pixels_painted = 0
        mesh_operation_uv(self._mesh_uvx[vertex_indices, :], self._paint_uv, self._tolerance)
        return self._pixels_painted > 0
    
    def _paint_uv(self, pixel, weights):
        position = barycentric_decode(weights, self._simplex_3d)
        distance = np.linalg.norm(position - self._origin)
        # TODO: THIS DISTANCE IS NOT GEODESIC
        if (distance > self._size):
            return
        x = int(pixel[0, 0])
        y = int(pixel[0, 1])
        self._render_buffer[y, x, :] = self._color
        self._pixels_painted += 1


class mesh_operator_paint_image:
    def __init__(self, mesh_vertices_a, mesh_faces_a, mesh_face_normals_a, uv_transform, mesh_faces_b, mesh_uvx_b, point, align_prior, angle, scale, image_buffer, render_buffer, tolerance=0):
        self._mesh_vertices_a = mesh_vertices_a
        self._mesh_faces_a = mesh_faces_a
        self._mesh_face_normals_a = mesh_face_normals_a
        self._uv_transform = uv_transform
        self._mesh_faces_b = mesh_faces_b
        self._mesh_uvx_b = mesh_uvx_b
        self._point = point
        self._align_prior = align_prior
        self._angle = angle
        self._scale = scale
        self._image_buffer = image_buffer
        self._image_width = image_buffer.shape[1]
        self._image_height = image_buffer.shape[0]
        self._render_buffer = render_buffer
        self._tolerance = tolerance
        self._out_source = np.array([[0, 0, 1]], dtype=mesh_face_normals_a.dtype)
        self._align_axis_default = np.array([[0, 1, 0]], dtype=mesh_face_normals_a.dtype)
        self._image_center = np.array([[self._image_width // 2, self._image_height // 2, 0]], dtype=mesh_vertices_a.dtype)
        self._done_faces = set()

    def paint(self, face_index, level):
        if (face_index in self._done_faces):
            return

        vertex_indices_a = self._mesh_faces_a[face_index]
        simplex_3d_a = self._mesh_vertices_a[vertex_indices_a, :]
        out_simplex_a = self._mesh_face_normals_a[face_index:(face_index + 1), :]

        if (level == 0):
            vus = self._align_prior
            vfs = out_simplex_a
            vhs = np.cross(vus, vfs)

            vud = self._align_axis_default
            vfd = self._out_source
            vhd = np.cross(vud, vfd)

            align_orientation = np.linalg.inv(np.vstack((vus, vfs, vhs))) @ np.vstack((vud, vfd, vhd))
            align_orientation = align_orientation @ cv2.Rodrigues((self._out_source * (-self._angle)))[0]

            simplex_image_a = (((simplex_3d_a - self._point) @ align_orientation) * self._scale) + self._image_center
            simplex_image_a[:, 2] = 0

            self._image_uvx = {vertex_indices_a[i] : simplex_image_a[i:(i + 1), :] for i in range(0, 3)}

        vut = [vertex_indices_a[i] in self._image_uvx for i in range(0, 3)]
        count = vut[0] + vut[1] + vut[2]

        if (count < 2):
            return False

        if (count == 2):
            # TODO: THIS UNWRAPPING METHOD IS AFFECTED BY THE ORDER IN WHICH FACES ARE PROCESSED
            vip, viq, vix = (1, 2, 0) if (not vut[0]) else (2, 0, 1) if (not vut[1]) else (0, 1, 2)

            vps = simplex_3d_a[vip:(vip + 1), :]
            vqs = simplex_3d_a[viq:(viq + 1), :]
            vxs = simplex_3d_a[vix:(vix + 1), :]
            
            vpd = self._image_uvx[vertex_indices_a[vip]]
            vqd = self._image_uvx[vertex_indices_a[viq]]

            vls = vqs - vps
            vss = np.linalg.norm(vls)
            vls = vls / vss
            vos = out_simplex_a
            vns = np.cross(vls, vos)

            vld = vqd - vpd
            vsd = np.linalg.norm(vld)
            vld = vld / vsd
            vod = self._out_source
            vnd = np.cross(vld, vod)
            
            align_orientation = np.linalg.inv(np.vstack((vls, vos, vns))) @ np.vstack((vld, vod, vnd))
            align_scale = vsd / vss

            vxd = (((vxs - vps) @ align_orientation) * (align_scale)) + vpd
            vxd[:, 2] = 0

            self._image_uvx[vertex_indices_a[vix]] = vxd

        self._done_faces.add(face_index)

        vertex_indices_b = self._mesh_faces_b[face_index]
        self._simplex_3d_b = np.vstack([self._image_uvx[vertex_index_a] for vertex_index_a in self._uv_transform[vertex_indices_b]])
        self._pixels_painted = 0
        mesh_operation_uv(self._mesh_uvx_b[vertex_indices_b, :], self._paint_uv, self._tolerance)
        return self._pixels_painted > 0
    
    def _paint_uv(self, pixel, weights):
        position = barycentric_decode(weights, self._simplex_3d_b)
        u = int(position[0, 0])
        v = int(position[0, 1])
        if ((u < 0) or (v < 0) or (u >= self._image_width) or (v >= self._image_height)):
            return
        x = int(pixel[0, 0])
        y = int(pixel[0, 1])
        self._render_buffer[y, x, :] = self._image_buffer[v, u, :]
        self._pixels_painted += 1


#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
