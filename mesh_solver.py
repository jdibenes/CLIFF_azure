
import time
import numpy as np
import trimesh.exchange.obj
import math
import cv2


#------------------------------------------------------------------------------
# Geometry
#------------------------------------------------------------------------------

def geometry_align_basis(vas, vbs, vad, vbd):
    return np.linalg.inv(np.vstack((vas, vbs, np.cross(vas, vbs)))) @ np.vstack((vad, vbd, np.cross(vad, vbd)))


#------------------------------------------------------------------------------
# Texture Processing
#------------------------------------------------------------------------------

def texture_load_image(filename_image, alpha=255):
    image_array = cv2.cvtColor(cv2.imread(filename_image, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    channels = image_array.shape[2]
    if (channels == 3):
        image_array = np.dstack((image_array, np.ones((image_array.shape[0], image_array.shape[1], 1), dtype=np.uint8) * alpha))
    return image_array


def texture_load_uv(filename_uv):
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


def texture_uv_to_uvx(uv, image_shape):
    uv[:, 0] = uv[:, 0] * (image_shape[1] - 1)
    uv[:, 1] = (1 - uv[:, 1]) * (image_shape[0] - 1)
    return uv


def texture_uvx_invert(uvx, image_shape, axis):
    uvx[:, axis] = (image_shape[1 - axis] - 1) - uvx[:, axis]
    return uvx


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


def texture_alpha_remap(alpha, src, dst):
    ah = alpha >= src[1]
    al = alpha < src[1]
    alpha[ah] = np.interp(alpha[ah], [src[1], src[2]], [dst[1], dst[1]])
    alpha[al] = np.interp(alpha[al], [src[0], src[1]], [dst[0], dst[1]])
    return alpha


def texture_processor(simplex_uvx, callback, tolerance=0):
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


#------------------------------------------------------------------------------
# Mesh Neighborhood Processing
#------------------------------------------------------------------------------

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


#------------------------------------------------------------------------------
# Mesh Painting
#------------------------------------------------------------------------------

class mesh_neighborhood_operation_brush:
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
        texture_processor(self._mesh_uvx[vertex_indices, :], self._paint_uv, self._tolerance)
        return mesh_neighborhood_processor_command.EXPAND if (self._pixels_painted > 0) else mesh_neighborhood_processor_command.IGNORE
    
    def _paint_uv(self, pixels, weights):
        # TODO: THIS DISTANCE IS NOT GEODESIC
        distances = np.linalg.norm((weights @ self._simplex_3d) - self._origin, axis=1)
        self._pixels_painted = 0
        for target in self._targets:
            self._pixels_painted += target(pixels, distances)


class brush_solid:
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


class brush_gradient:
    def __init__(self, size, color_center, color_edge, hardness, render_buffer):
        self._size = size
        self._color_center = color_center
        self._color_edge = color_edge
        self._render_buffer = render_buffer
        self._src = [0, hardness, 1]
        self._dst = [0, 0.5, 1]

    def paint(self, pixels, distances):
        mask = distances < self._size
        pixels_painted = np.count_nonzero(mask)
        if (pixels_painted > 0):
            selection = pixels[mask, :]
            self._render_buffer[selection[:, 1], selection[:, 0], :] = texture_alpha_blend(self._color_center, self._color_edge, texture_alpha_remap(distances[mask, np.newaxis] / self._size, self._src, self._dst))
        return pixels_painted


class mesh_neighborhood_operation_decal:
    def __init__(self, mesh_vertices, mesh_faces, mesh_face_normals, mesh_uvx, uv_transform, origin, align_prior, angle, scale, image_buffer, render_buffer, tolerance=0):
        self._mesh_vertices = mesh_vertices
        self._mesh_faces = mesh_faces
        self._mesh_face_normals = mesh_face_normals
        self._mesh_uvx = mesh_uvx
        self._uv_transform = uv_transform
        self._origin = origin
        self._align_prior = align_prior
        self._angle = angle
        self._scale = scale
        self._image_buffer = image_buffer
        self._render_buffer = render_buffer
        self._tolerance = tolerance

    def paint(self, face_index, level):
        face_normal = self._mesh_face_normals[face_index:(face_index + 1), :]        
        vertex_indices_b = self._mesh_faces[face_index]
        vertex_indices_a = self._uv_transform[vertex_indices_b]
        
        if (level == 0):
            self._align_axis = np.array([[0, 1, 0]], dtype=self._mesh_face_normals.dtype)
            self._uvx_normal = np.array([[0, 0, 1]], dtype=self._mesh_face_normals.dtype)
            self._image_uvx = np.ones_like(self._mesh_vertices)
            
            vps = self._origin
            vxs = self._mesh_vertices[vertex_indices_b, :]
            vpd = np.array([[self._image_buffer.shape[1] // 2, self._image_buffer.shape[0] // 2, 0]], dtype=self._mesh_vertices.dtype)

            align_outward = geometry_align_basis(self._align_prior, face_normal, self._align_axis * self._scale, self._uvx_normal)
            align_simplex = cv2.Rodrigues(self._uvx_normal * self._angle)[0].T
            
            vxd = (((vxs - vps) @ align_outward) @ align_simplex) + vpd
            vxd[:, 2] = 0

            self._image_uvx[vertex_indices_a, :] = vxd

        unwrapped = self._image_uvx[vertex_indices_a, 2] == 0
        unwrapped_count = unwrapped.sum()

        if (unwrapped_count <= 1):
            return mesh_neighborhood_processor_command.CONTINUE

        if (unwrapped_count == 2):
            # TODO: THIS UNWRAPPING METHOD IS AFFECTED BY THE ORDER IN WHICH FACES ARE PROCESSED
            unwrapped_indices = [1, 2, 0] if (not unwrapped[0]) else [2, 0, 1] if (not unwrapped[1]) else [0, 1, 2]

            vips_a, viqs_a, vixs_a = vertex_indices_a[unwrapped_indices]
            vips_b, viqs_b, vixs_b = vertex_indices_b[unwrapped_indices]

            vps = self._mesh_vertices[vips_b:(vips_b + 1), :]
            vqs = self._mesh_vertices[viqs_b:(viqs_b + 1), :]
            vxs = self._mesh_vertices[vixs_b:(vixs_b + 1), :]
            vpd = self._image_uvx[vips_a:(vips_a + 1), :]
            vqd = self._image_uvx[viqs_a:(viqs_a + 1), :]

            align_outward = geometry_align_basis(vqs - vps, face_normal, vqd - vpd, self._uvx_normal)

            vxd = ((vxs - vps) @ align_outward) + vpd
            vxd[:, 2] = 0

            self._image_uvx[vixs_a:(vixs_a + 1), :] = vxd

        self._simplex_3d_b = self._image_uvx[vertex_indices_a, 0:2]
        texture_processor(self._mesh_uvx[vertex_indices_b, :], self._paint_uv, self._tolerance)
        return mesh_neighborhood_processor_command.EXPAND if (self._pixels_painted > 0) else mesh_neighborhood_processor_command.IGNORE

    def _paint_uv(self, pixels_dst, weights):        
        pixels_src = texture_uvx_invert(weights @ self._simplex_3d_b, self._image_buffer.shape, 1)
        mask = texture_test_inside(self._image_buffer, pixels_src[:, 0], pixels_src[:, 1])
        self._pixels_painted = np.count_nonzero(mask)
        if (self._pixels_painted > 0):
            dst = pixels_dst[mask, :]
            src = pixels_src[mask, :]
            self._render_buffer[dst[:, 1], dst[:, 0], :] = texture_read(self._image_buffer, src[:, 0], src[:, 1])


def mesh_create_painter_brush(mesh_a, mesh_b, mesh_uvx, face_index, origin, brushes, tolerance=0):
    mno = mesh_neighborhood_operation_brush(mesh_b.vertices.view(np.ndarray), mesh_b.faces.view(np.ndarray), mesh_uvx, origin, brushes, tolerance)
    mnp = mesh_neighborhood_processor(mesh_a, {face_index}, mno.paint)
    return mnp


def mesh_create_painter_decal(mesh_a, mesh_b, mesh_uvx, uv_transform, face_index, origin, align_prior, angle, scale, image_buffer, render_buffer, tolerance=0):
    mno = mesh_neighborhood_operation_decal(mesh_b.vertices.view(np.ndarray), mesh_b.faces.view(np.ndarray), mesh_b.face_normals, mesh_uvx, uv_transform, origin, align_prior, angle, scale, image_buffer, render_buffer, tolerance)
    mnp = mesh_neighborhood_processor(mesh_a, {face_index}, mno.paint)
    return mnp


def mesh_neighborhood_processor_execute(mnp, timeout, steps=1):
    start = time.perf_counter()
    while (not mnp.done()):
        mnp.invoke(steps)
        if (time.perf_counter() - start >= timeout):
            break
    return mnp.done()

