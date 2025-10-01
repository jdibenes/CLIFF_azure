
import time
import math
import numpy as np
import cv2
import pyrender

import trimesh.visual
import trimesh.exchange.obj

from PIL import Image, ImageFont, ImageDraw


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
        image_array = np.dstack((image_array, np.ones((image_array.shape[0], image_array.shape[1], 1), np.uint8) * alpha))
    return image_array


def texture_load_uv(filename_uv):
    with open(filename_uv, 'r') as obj_file:
        obj_mesh_a = trimesh.exchange.obj.load_obj(file_obj=obj_file, maintain_order=True)
    with open(filename_uv, 'r') as obj_file:
        obj_mesh_b = trimesh.exchange.obj.load_obj(file_obj=obj_file)
    mesh_vertices_b = obj_mesh_b['geometry'][filename_uv]['vertices']
    mesh_faces_a = obj_mesh_a['geometry'][filename_uv]['faces']
    mesh_faces_b = obj_mesh_b['geometry'][filename_uv]['faces']
    mesh_uv_b = obj_mesh_b['geometry'][filename_uv]['visual'].uv
    uv_transform = np.zeros(mesh_vertices_b.shape[0], np.int64)
    for face_index in range(0, mesh_faces_b.shape[0]):
        for vertex_index in range(0, 3):
            uv_transform[mesh_faces_b[face_index, vertex_index]] = mesh_faces_a[face_index, vertex_index]
    return (uv_transform, mesh_faces_b, mesh_uv_b)


def texture_create_text(text_list, font_name, font_size, font_color, bg_color=(255, 255, 255, 255), stroke_width=0, spacing=4, pad_factor=(0.0, 0.0)):
    font = ImageFont.truetype(font_name, font_size)
    arrays = []
    for text in text_list:
        bbox = font.getbbox(text, stroke_width=stroke_width)
        image = Image.new('RGBA', (bbox[2], bbox[3]), bg_color)
        ImageDraw.Draw(image).text((0, 0), text, font_color, font, stroke_width=stroke_width)
        arrays.append(np.array(image.crop(bbox)))
    image_count = len(arrays)
    if (image_count > 1):
        full_w = np.max([array.shape[1] for array in arrays])
        padded_arrays = []
        for i, array in enumerate(arrays):
            w = array.shape[1]
            count = full_w - w
            pad_l = count // 2
            pad_r = count - pad_l
            padded_arrays.append(cv2.copyMakeBorder(array, 0 if (i == 0) else spacing, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=bg_color))
        composite = np.vstack(padded_arrays)
    else:
        composite = arrays[0]
    h, w = composite.shape[0:2]
    pad_x = math.ceil(w * pad_factor[0])
    pad_y = math.ceil(h * pad_factor[1])
    composite = cv2.copyMakeBorder(composite, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=bg_color)
    return composite


def texture_create_visual(uv, texture):
    return trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(texture))


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
# Mesh Processing
#------------------------------------------------------------------------------

def mesh_create(vertices, faces, visual=None):
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)


def mesh_to_renderer(mesh):
    return pyrender.Mesh.from_trimesh(mesh)


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

    def invoke_timeslice(self, timeout, steps=1):
        start = time.perf_counter()
        while (not self.done()):
            self.invoke(steps)
            if ((time.perf_counter() - start) >= timeout):
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
            self._pixels_painted += target(pixels, distances, self._level)


class mesh_neighborhood_operation_decal:
    def __init__(self, mesh_vertices, mesh_faces, mesh_face_normals, mesh_uvx, uv_transform, origin, target, tolerance=0):
        self._mesh_vertices = mesh_vertices
        self._mesh_faces = mesh_faces
        self._mesh_face_normals = mesh_face_normals
        self._mesh_uvx = mesh_uvx
        self._uv_transform = uv_transform
        self._origin = origin
        self._target = target
        self._tolerance = tolerance

    def paint(self, face_index, level):
        self._face_normal = self._mesh_face_normals[face_index:(face_index + 1), :]        
        self._vertex_indices_b = self._mesh_faces[face_index]
        self._vertex_indices_a = self._uv_transform[self._vertex_indices_b]
        self._level = level
        command = self._target(self._mesh_vertices, self._face_normal, self._origin, self._vertex_indices_b, self._vertex_indices_a, None, None, self._level)
        if (command != mesh_neighborhood_processor_command.EXPAND):
            return command
        texture_processor(self._mesh_uvx[self._vertex_indices_b, :], self._paint_uv, self._tolerance)
        return mesh_neighborhood_processor_command.EXPAND if (self._pixels_painted > 0) else mesh_neighborhood_processor_command.IGNORE

    def _paint_uv(self, pixels, weights):
        self._pixels_painted = self._target(self._mesh_vertices, self._face_normal, self._origin, self._vertex_indices_b, self._vertex_indices_a, pixels, weights, self._level)


class paint_brush_solid:
    def __init__(self, size, color, render_buffer):
        self._size = size
        self._color = color
        self._render_buffer = render_buffer

    def paint(self, pixels, distances, level):
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
        self._render_buffer = render_buffer
        self._src = [0, hardness, 1]
        self._dst = [0, 0.5, 1]

    def paint(self, pixels, distances, level):
        mask = distances < self._size
        pixels_painted = np.count_nonzero(mask)
        if (pixels_painted > 0):
            selection = pixels[mask, :]
            self._render_buffer[selection[:, 1], selection[:, 0], :] = texture_alpha_blend(self._color_center, self._color_edge, texture_alpha_remap(distances[mask, np.newaxis] / self._size, self._src, self._dst))
        return pixels_painted


class paint_decal_solid:
    def __init__(self, align_prior, angle, scale, image_buffer, render_buffer, tolerance=0):
        self._align_prior = align_prior
        self._angle = angle
        self._scale = scale
        self._image_buffer = image_buffer
        self._render_buffer = render_buffer
        self._tolerance = tolerance
        self._simplices = []
        self._simplices_map = []

    def _push_simplex(self, simplex):
        self._simplices.append(simplex)
        self._simplices_map.append(None)

    def _test_simplex(self, point, i, tolerance=0):
        simplex = self._simplices[i]
        simplex_map = self._simplices_map[i]
        anchor = simplex[2:3, :]
        if (simplex_map is None):
            simplex_map = np.linalg.inv(simplex[0:2, :] - anchor)
            self._simplices_map[i] = simplex_map
        ab = (point - anchor) @ simplex_map
        abc = np.hstack((ab, 1 - ab[:, 0:1] - ab[:, 1:2]))
        return np.all(abc > -tolerance)

    def _bootstrap(self, mesh_vertices, face_normal, origin, indices_vertices, indices_uvx, pixels_dst, weights_src, level):
        self._align_axis = np.array([[0, 1, 0]], face_normal.dtype)
        self._uvx_normal = np.array([[0, 0, 1]], face_normal.dtype)

        self._image_uvx = np.ones_like(mesh_vertices)

        vps = origin
        vxs = mesh_vertices[indices_vertices, :]

        vpd = np.array([[self._image_buffer.shape[1] // 2, self._image_buffer.shape[0] // 2, 0]], mesh_vertices.dtype)

        align_outward = geometry_align_basis(self._align_prior, face_normal, self._align_axis * self._scale, self._uvx_normal)
        align_simplex = cv2.Rodrigues(self._uvx_normal * self._angle)[0].T
        
        vxd = (((vxs - vps) @ align_outward) @ align_simplex) + vpd
        vxd[:, 2] = 0

        self._image_uvx[indices_uvx, :] = vxd
        self._push_simplex(vxd[:, 0:2])

        return mesh_neighborhood_processor_command.EXPAND

    def _unwrap(self, mesh_vertices, face_normal, origin, indices_vertices, indices_uvx, pixels_dst, weights_src, level):
        # TODO: THIS UNWRAPPING METHOD IS AFFECTED BY THE ORDER IN WHICH FACES ARE PROCESSED
        unwrapped = self._image_uvx[indices_uvx, 2] == 0
        unwrapped_count = unwrapped.sum()

        if (unwrapped_count <= 1):
            return mesh_neighborhood_processor_command.CONTINUE
        
        if (unwrapped_count >= 3):
            return mesh_neighborhood_processor_command.EXPAND

        unwrapped_indices = [1, 2, 0] if (not unwrapped[0]) else [2, 0, 1] if (not unwrapped[1]) else [0, 1, 2]

        vips_a, viqs_a, vixs_a = indices_uvx[unwrapped_indices]
        vips_b, viqs_b, vixs_b = indices_vertices[unwrapped_indices]

        vps = mesh_vertices[vips_b:(vips_b + 1), :]
        vqs = mesh_vertices[viqs_b:(viqs_b + 1), :]
        vxs = mesh_vertices[vixs_b:(vixs_b + 1), :]

        vpd = self._image_uvx[vips_a:(vips_a + 1), :]
        vqd = self._image_uvx[viqs_a:(viqs_a + 1), :]

        align_outward = geometry_align_basis(vqs - vps, face_normal, vqd - vpd, self._uvx_normal)

        vxd = ((vxs - vps) @ align_outward) + vpd
        vxd[:, 2] = 0

        for i in range(0, len(self._simplices)):
            double_cover = self._test_simplex(vxd[:, 0:2], len(self._simplices) - 1 - i, self._tolerance)
            if (double_cover):
                return mesh_neighborhood_processor_command.IGNORE

        self._image_uvx[vixs_a:(vixs_a + 1), :] = vxd
        self._push_simplex(np.vstack((vxd[:, 0:2], vqd[:, 0:2], vpd[:, 0:2])))

        return mesh_neighborhood_processor_command.EXPAND

    def _blit(self, mesh_vertices, face_normal, origin, indices_vertices, indices_uvx, pixels_dst, weights_src, level):
        pixels_src = texture_uvx_invert(weights_src @ self._image_uvx[indices_uvx, 0:2], self._image_buffer.shape, 1)
        mask = texture_test_inside(self._image_buffer, pixels_src[:, 0], pixels_src[:, 1])
        pixels_painted = np.count_nonzero(mask)
        if (pixels_painted > 0):
            dst = pixels_dst[mask, :]
            src = pixels_src[mask, :]
            self._render_buffer[dst[:, 1], dst[:, 0], :] = texture_read(self._image_buffer, src[:, 0], src[:, 1])
        return pixels_painted

    def paint(self, mesh_vertices, face_normal, origin, indices_vertices, indices_uvx, pixels_dst, weights_src, level):
        call = self._blit if ((pixels_dst is not None) and (weights_src is not None)) else self._unwrap if (level > 0) else self._bootstrap
        return call(mesh_vertices, face_normal, origin, indices_vertices, indices_uvx, pixels_dst, weights_src, level)


def painter_create_brush(mesh_a, mesh_b, mesh_uvx, uv_transform, face_index, origin, brush_collection, tolerance=0):
    mno = mesh_neighborhood_operation_brush(mesh_b.vertices.view(np.ndarray), mesh_b.faces.view(np.ndarray), mesh_uvx, origin, brush_collection, tolerance)
    mnp = mesh_neighborhood_processor(mesh_a, {face_index}, mno.paint)
    return mnp


def painter_create_decal(mesh_a, mesh_b, mesh_uvx, uv_transform, face_index, origin, decal_collection, tolerance=0):
    mno = mesh_neighborhood_operation_decal(mesh_b.vertices.view(np.ndarray), mesh_b.faces.view(np.ndarray), mesh_b.face_normals, mesh_uvx, uv_transform, origin, decal_collection, tolerance)
    mnp = mesh_neighborhood_processor(mesh_a, {face_index}, mno.paint)
    return mnp


#------------------------------------------------------------------------------
# Rendering
#------------------------------------------------------------------------------

class renderer:
    def __init__(self, width, height, fx, fy, cx, cy, znear=0.05, zfar=100, point_size=1, bg_color=(1.0, 1.0, 1.0), ambient_light=(0.0, 0.0, 0.0), lamp_color=(1.0, 1.0, 1.0), lamp_intensity=3.0):
        self._renderer = pyrender.OffscreenRenderer(width, height, point_size)
        self._scene = pyrender.Scene(None, bg_color, ambient_light, 'main_scene')
        self._camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear, zfar, 'main_camera')
        self._light = pyrender.DirectionalLight(lamp_color, lamp_intensity, 'main_camera_lamp')
        self._groups = dict()

        self._camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], np.float32).reshape((4, 4))

        self._node_camera = self._scene.add(self._camera, 'internal@main@camera', self._camera_pose)
        self._node_light = self._scene.add(self._light, 'internal@main@lamp', self._camera_pose)

    def set_camera_pose(self, camera_pose):
        self._camera_pose = camera_pose

        self._scene.set_pose(self._node_camera, self._camera_pose)
        self._scene.set_pose(self._node_light, self._camera_pose)

    def get_camera_pose(self):
        return self._camera_pose

    def render(self):
        color, depth = self._renderer.render(self._scene, pyrender.RenderFlags.RGBA)
        return color, depth

    def group_item_add(self, group, name, item, pose=None):
        nodes = self._groups.get(group, dict())
        previous = nodes.get(name, None)
        if (previous is not None):
            self._scene.remove_node(previous)
        if (pose is None):
            pose = np.eye(4)
        nodes[name] = self._scene.add(item, 'external@' + group + '@' + name, pose)
        self._groups[group] = nodes

    def group_item_remove(self, group, name):
        nodes = self._groups.get(group, None)
        if (nodes is not None):
            item = nodes.get(name, None)
            if (item is not None):
                self._scene.remove_node(item)

    def group_item_set_pose(self, group, name, pose):
        nodes = self._groups.get(group, None)
        if (nodes is not None):
            item = nodes.get(name, None)
            if (item is not None):
                self._scene.set_pose(item, pose)

    def group_item_get_pose(self, group, name):
        nodes = self._groups.get(group, None)
        if (nodes is not None):
            item = nodes.get(name, None)
            if (item is not None):
                return self._scene.get_pose(item)
        return None

    def group_clear(self, group):
        nodes = self._groups.pop(group, None)
        if (nodes is not None):
            for name, item in nodes.items():
                self._scene.remove_node(item)










def process_input(self, inputs):
    # TODO: USE KEYBINDS
    # adjust yaw
    # adjust pitch
    # gravity vector?
    # translate?
    pass

        #split_normal = np.cross(v1d / vds, self._uvx_normal)
        #split_distance = split_normal @ vpd.T

        #o = split_normal @ vxd.T - split_distance
        #if (o > 0):
        #    print('IN?')