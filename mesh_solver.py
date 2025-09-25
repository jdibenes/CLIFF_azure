
import numpy as np
import trimesh.exchange.obj
import math
import cv2
import time


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







def geometry_align_basis(vas, vbs, vad, vbd):
    return np.linalg.inv(np.vstack((vas, vbs, np.cross(vas, vbs)))) @ np.vstack((vad, vbd, np.cross(vad, vbd)))








def mesh_raycast(mesh, origin, direction):
    point, rid, tid = mesh.ray.intersects_location(origin, direction, multiple_hits=False)
    return (point, tid[0]) if (len(rid) > 0) else (None, None)







def texture_test_inside(texture, x, y):
    # ignore last row and column to simplify bilinear interpolation
    return (x >= 0) & (y >= 0) & (x < (texture.shape[1] - 1)) & (y < (texture.shape[0] - 1))


def texture_read(texture, x, y):
    xf = np.floor(x)
    yf = np.floor(y)

    a1 = (x - xf)[:, np.newaxis]
    b1 = (y - yf)[:, np.newaxis]
    a0 = 1 - a1
    b0 = 1 - b1

    x0 = xf.astype(np.int32)
    y0 = yf.astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # sqrt space
    return b0 * (a0 * texture[y0, x0, :] + a1 * texture[y0, x1, :]) + b1 * (a0 * texture[y1, x0, :] + a1 * texture[y1, x1, :])


def texture_alpha_blend(texture_1a, texture_a, alpha):
    # linear space
    return np.sqrt((1 - alpha) * np.square(texture_1a / 255) + alpha * np.square(texture_a / 255)) * 255


def texture_alpha_remap(alpha, midpoint):
    ah = alpha >= midpoint
    al = alpha < midpoint
    alpha[ah] = np.interp(alpha[ah], [midpoint, 1], [0.5, 1])
    alpha[al] = np.interp(alpha[al], [0, midpoint], [0, 0.5])
    return alpha


def mesh_uv_processor(simplex_uvx, callback, tolerance=0):
    # uvx : [u * (w - 1), (1 - v) * (h - 1)]
    u = simplex_uvx[:, 0]
    v = simplex_uvx[:, 1]

    left = math.floor(np.min(u))
    right = math.ceil(np.max(u))
    top = math.floor(np.min(v))
    bottom = math.ceil(np.max(v))

    if ((left < right) and (top < bottom)):
        box = np.mgrid[left:right, top:bottom].T.reshape((-1, 2))
        anchor = simplex_uvx[2:3, :]
        ab = (box - anchor) @ np.linalg.inv(simplex_uvx[0:2, :] - anchor)
        abc = np.hstack((ab, 1 - ab[:, 0:1] - ab[:, 1:2]))
        mask = np.logical_and.reduce(abc >= -tolerance, axis=1)
        if (np.any(mask)):
            callback(box[mask, :], abc[mask, :])


class mesh_neighborhood_builder:
    def __init__(self, mesh):
        self._mesh = mesh
        self._mesh_faces = self._mesh.faces.view(np.ndarray)
        self._mesh_vertex_faces = self._mesh.vertex_faces
        self._seen_face = set()
        self._seen_vertex = set()
        self._iterations = 0

    def fetch(self, expand_faces, ignore_faces):
        result = set()
        self._seen_face.update(expand_faces)
        self._seen_face.update(ignore_faces)
        for face_anchor in expand_faces:
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


class mesh_neighborhood_processor_command:
    CONTINUE = 0
    IGNORE = 1
    EXPAND = 2


class mesh_neighborhood_processor:
    def __init__(self, mesh, faces, callback):
        self._mnb = mesh_neighborhood_builder(mesh)
        self._faces = faces
        self._expand_faces = set()
        self._ignore_faces = set()
        self._callback = callback
        self._done = False

    def invoke(self, max_iterations):
        for _ in range(0, max_iterations):
            if (len(self._expand_faces) > 0):
                self._faces = self._mnb.fetch(self._expand_faces, self._ignore_faces)
                self._expand_faces.clear()
                self._ignore_faces.clear()
            for face_anchor in self._faces:
                code = self._callback(face_anchor, self._mnb.level())
                if (code == mesh_neighborhood_processor_command.EXPAND):
                    self._expand_faces.add(face_anchor)
                elif (code == mesh_neighborhood_processor_command.IGNORE):
                    self._ignore_faces.add(face_anchor)
            if (len(self._expand_faces) < 1):
                self._done = True
                break
    
    def done(self):
        return self._done


class mesh_neighborhood_operation_paint_brush:
    def __init__(self, mesh_vertices, mesh_faces, mesh_uvx, origin, targets, tolerance=0):
        self._mesh_vertices = mesh_vertices
        self._mesh_faces = mesh_faces
        self._mesh_uvx = mesh_uvx
        self._origin = origin
        self._targets = targets
        self._tolerance = tolerance

    def paint(self, face_index, level):
        vertex_indices = self._mesh_faces[face_index]
        self._simplex_3d = self._mesh_vertices[vertex_indices, :]
        self._level = level
        mesh_uv_processor(self._mesh_uvx[vertex_indices, :], self._paint_uv, self._tolerance)
        return mesh_neighborhood_processor_command.EXPAND if (self._pixels_painted > 0) else mesh_neighborhood_processor_command.IGNORE
    
    def _paint_uv(self, pixels, weights):
        # TODO: THIS DISTANCE IS NOT GEODESIC
        distances = np.linalg.norm((weights @ self._simplex_3d) - self._origin, axis=1)
        self._pixels_painted = 0
        for target in self._targets:
            self._pixels_painted += target(pixels, distances)


class paint_brush_solid:
    def __init__(self, size, color, render_buffer):
        self._size = size
        self._color = color
        self._render_buffer = render_buffer

    def paint(self, pixels, distances):
        mask = distances < self._size
        pixels_painted = np.count_nonzero(mask)
        if (pixels_painted > 0):
            selection = pixels[mask, :]
            self._render_buffer[selection[:, 1], selection[:, 0], :] = self._color
        return pixels_painted


class paint_brush_gradient:
    def __init__(self, size, color_center, color_edge, hardness, render_buffer):
        self._size = size
        self._color_center = color_center
        self._color_edge = color_edge
        self._hardness = hardness
        self._render_buffer = render_buffer

    def paint(self, pixels, distances):
        mask = distances < self._size
        pixels_painted = np.count_nonzero(mask)
        if (pixels_painted > 0):
            selection = pixels[mask, :]
            self._render_buffer[selection[:, 1], selection[:, 0], :] = texture_alpha_blend(self._color_center, self._color_edge, texture_alpha_remap(distances[mask, np.newaxis] / self._size, self._hardness))
        return pixels_painted







class mesh_neighborhood_operation_paint_image:
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
        self._render_buffer = render_buffer
        self._tolerance = tolerance

    def paint(self, face_index, level):
        vertex_indices_a = self._mesh_faces_a[face_index]
        simplex_3d_a = self._mesh_vertices_a[vertex_indices_a, :]
        face_normal_a = self._mesh_face_normals_a[face_index:(face_index + 1), :]

        if (level == 0):
            self._align_axis = np.array([[0, 1, 0]], dtype=self._mesh_face_normals_a.dtype)
            self._uvx_normal = np.array([[0, 0, 1]], dtype=self._mesh_face_normals_a.dtype)        

            simplex_image_a = ((simplex_3d_a - self._point) @ geometry_align_basis(self._align_prior, face_normal_a, self._align_axis * self._scale, self._uvx_normal) @ cv2.Rodrigues(self._uvx_normal * self._angle)[0].T) + np.array([[self._image_buffer.shape[1] // 2, self._image_buffer.shape[0] // 2, 0]], dtype=self._mesh_vertices_a.dtype)
            simplex_image_a[:, 2] = 0

            self._image_uvx = {vertex_indices_a[i] : simplex_image_a[i:(i + 1), :] for i in range(0, 3)}

        vt = [vertex_indices_a[i] in self._image_uvx for i in range(0, 3)]
        count = vt[0] + vt[1] + vt[2]

        if (count <= 1):
            return mesh_neighborhood_processor_command.CONTINUE

        if (count == 2):
            # TODO: THIS UNWRAPPING METHOD IS AFFECTED BY THE ORDER IN WHICH FACES ARE PROCESSED
            vip, viq, vix = (1, 2, 0) if (not vt[0]) else (2, 0, 1) if (not vt[1]) else (0, 1, 2)

            vps = simplex_3d_a[vip:(vip + 1), :]
            vqs = simplex_3d_a[viq:(viq + 1), :]
            vxs = simplex_3d_a[vix:(vix + 1), :]
            
            vpd = self._image_uvx[vertex_indices_a[vip]]
            vqd = self._image_uvx[vertex_indices_a[viq]]

            vxd = ((vxs - vps) @ geometry_align_basis(vqs - vps, face_normal_a, vqd - vpd, self._uvx_normal)) + vpd
            vxd[:, 2] = 0

            self._image_uvx[vertex_indices_a[vix]] = vxd

        vertex_indices_b = self._mesh_faces_b[face_index]
        self._simplex_3d_b = np.vstack([self._image_uvx[vertex_index_a] for vertex_index_a in self._uv_transform[vertex_indices_b]])
        mesh_uv_processor(self._mesh_uvx_b[vertex_indices_b, :], self._paint_uv, self._tolerance)
        return mesh_neighborhood_processor_command.EXPAND if (self._pixels_painted > 0) else mesh_neighborhood_processor_command.IGNORE

    def _paint_uv(self, pixels_dst, weights):
        pixels_src = weights @ self._simplex_3d_b
        mask = texture_test_inside(self._image_buffer, pixels_src[:, 0], pixels_src[:, 1])
        dst = pixels_dst[mask, :]
        src = pixels_src[mask, :]
        self._pixels_painted = dst.shape[0]
        if (self._pixels_painted > 0):
            self._render_buffer[dst[:, 1], dst[:, 0], :] = texture_read(self._image_buffer, src[:, 0], src[:, 1])


#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------














# "GAUSSIAN" brush




'''
def barycentric_create(simplex):
    # simplex plane must not pass through the origin
    #return np.linalg.inv(np.vstack((simplex[0:2, :] - simplex[2:3, :], simplex[2:3, :])))
    o = simplex[2:3, :]
    d = simplex[0:2, :] - o
    #return (np.linalg.pinv(d), o)
    return (d.T @ np.linalg.inv(d @ d.T), o)



def barycentric_encode(vertex, transform):
    weights = ((vertex - transform[1]) @ transform[0])[0, :]
    #weights = (vertex @ transform)[0, :]
    #weights[2] = 1 - weights[0] - weights[1]
    return np.append(weights, 1 - weights[0] - weights[1])


def barycentric_decode(weights, simplex):
    return (weights[0] * simplex[0:1, :]) + (weights[1] * simplex[1:2, :]) + (weights[2] * simplex[2:3, :])
'''