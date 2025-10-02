
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
    # TODO: error for singular matrix
    return np.linalg.inv(np.vstack((vas, vbs, np.cross(vas, vbs)))) @ np.vstack((vad, vbd, np.cross(vad, vbd)))


#------------------------------------------------------------------------------
# Image Processing
#------------------------------------------------------------------------------

def font_load(font_name, font_size):
    return ImageFont.truetype(font_name, font_size)


#------------------------------------------------------------------------------
# Texture Processing
#------------------------------------------------------------------------------

def texture_load_image(filename_image, load_alpha=True, alpha=255):
    rgb = cv2.imread(filename_image, cv2.IMREAD_COLOR_RGB)
    raw = cv2.imread(filename_image, cv2.IMREAD_UNCHANGED)
    a = raw[:, :, 3] if ((load_alpha) and (raw.shape[2] == 4)) else (np.ones((rgb.shape[0], rgb.shape[1], 1), rgb.dtype) * alpha)
    return np.dstack((rgb, a))


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


def texture_stack(textures, fill_color, spacing, vertical):
    axis_a, axis_b = (0, 1) if (vertical) else (1, 0)
    pad = np.zeros((len(textures), 4), np.int32)
    d = np.array([texture.shape[axis_b] for texture in textures], np.int32)
    fill_d = np.max(d) - d
    pad[0:, 2 * axis_b + 0] = fill_d // 2
    pad[0:, 2 * axis_b + 1] = fill_d - pad[:, 2 * axis_b + 0]
    pad[1:, 2 * axis_a + 0] = spacing
    return np.concatenate([cv2.copyMakeBorder(texture, pad[i, 0], pad[i, 1], pad[i, 2], pad[i, 3], cv2.BORDER_CONSTANT, value=fill_color) for i, texture in enumerate(textures)], axis_a)


def texture_pad(texture, pad_w, pad_h, fill_color):
    h, w = texture.shape[0:2]
    pad_x = math.ceil(w * pad_w)
    pad_y = math.ceil(h * pad_h)
    return cv2.copyMakeBorder(texture, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=fill_color)


def texture_create_text(text, font, font_color, bg_color=(255, 255, 255, 255), stroke_width=0):
    bbox = font.getbbox(text, stroke_width=stroke_width)
    image = Image.new('RGBA', (bbox[2], bbox[3]), bg_color)
    ImageDraw.Draw(image).text((0, 0), text, font_color, font, stroke_width=stroke_width)
    return np.array(image.crop(bbox))


def texture_create_multiline_text(text_list, font, font_color, bg_color=(255, 255, 255, 255), stroke_width=0, spacing=4):
    return texture_stack([texture_create_text(text, font, font_color, bg_color, stroke_width) for text in text_list], bg_color, spacing, True)


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
    # sqrt space
    return (1 - alpha) * texture_1a + alpha * texture_a


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

    def _test_simplex(self, point, i):
        simplex = self._simplices[i]
        simplex_map = self._simplices_map[i]
        anchor = simplex[2:3, :]
        if (simplex_map is None):
            simplex_map = np.linalg.inv(simplex[0:2, :] - anchor)
            self._simplices_map[i] = simplex_map
        ab = (point - anchor) @ simplex_map
        abc = np.hstack((ab, 1 - ab[:, 0:1] - ab[:, 1:2]))
        return np.all(abc > -self._tolerance)

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
            double_cover = self._test_simplex(vxd[:, 0:2], len(self._simplices) - 1 - i)
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


def painter_create_brush(mesh_a, mesh_b, mesh_uvx, uv_transform, face_index, origin, brush, tolerance=0):
    mno = mesh_neighborhood_operation_brush(mesh_b.vertices.view(np.ndarray), mesh_b.faces.view(np.ndarray), mesh_uvx, origin, brush, tolerance)
    mnp = mesh_neighborhood_processor(mesh_a, {face_index}, mno.paint)
    return mnp


def painter_create_decal(mesh_a, mesh_b, mesh_uvx, uv_transform, face_index, origin, decal, tolerance=0):
    mno = mesh_neighborhood_operation_decal(mesh_b.vertices.view(np.ndarray), mesh_b.faces.view(np.ndarray), mesh_b.face_normals, mesh_uvx, uv_transform, origin, decal, tolerance)
    mnp = mesh_neighborhood_processor(mesh_a, {face_index}, mno.paint)
    return mnp


#------------------------------------------------------------------------------
# Rendering
#------------------------------------------------------------------------------

def renderer_create_settings_offscreen(width, height, point_size=1):
    s = dict()
    s['viewport_width'] = width
    s['viewport_height'] = height
    s['point_size'] = point_size
    return s


def renderer_create_settings_scene(bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.0, 0.0, 0.0), name='scene'):
    s = dict()
    s['bg_color'] = bg_color
    s['ambient_light'] = ambient_light
    s['name'] = name
    return s


def renderer_create_settings_camera(fx, fy, cx, cy, znear=0.05, zfar=100, name='camera'):
    s = dict()
    s['fx'] = fx
    s['fy'] = fy
    s['cx'] = cx
    s['cy'] = cy
    s['znear'] = znear
    s['zfar'] = zfar
    s['name'] = name
    return s


def renderer_create_settings_lamp(color=(1.0, 1.0, 1.0), intensity=3.0, name='lamp'):
    s = dict()
    s['color'] = color
    s['intensity'] = intensity
    s['name'] = name
    return s


def renderer_create_settings_camera_transform(center=np.array([0, 0, 0], np.float32), yaw=0, pitch=0, distance=1, min_pitch=-75, max_pitch=75, znear=0.05, zfar=100):
    s = dict()
    s['center'] = center
    s['yaw'] = yaw
    s['pitch'] = pitch
    s['distance'] = distance
    s['min_pitch'] = min_pitch
    s['max_pitch'] = max_pitch
    s['znear'] = znear
    s['zfar'] = zfar
    return s


class renderer_camera_transform:
    def __init__(self, center, yaw, pitch, distance, min_pitch, max_pitch, znear, zfar):
        self._min_pitch = min_pitch
        self._max_pitch = max_pitch
        self._znear = znear
        self._zfar = zfar
        self._tz = np.eye(4, dtype=center.dtype)
        self._tc = np.eye(4, dtype=center.dtype)

        self.set_center(center)
        self.set_yaw(yaw)
        self.set_pitch(pitch)
        self.set_distance(distance)

    def get_yaw(self):
        return self._yaw

    def set_yaw(self, value):
        self._yaw = value
        self._ry = trimesh.transformations.rotation_matrix(np.radians(self._yaw), [0, 1, 0])
        self._dirty = True

    def update_yaw(self, delta):
        self.set_yaw(self._yaw + delta)

    def get_pitch(self):
        return self._pitch

    def set_pitch(self, value):
        self._pitch = np.clip(value, self._min_pitch, self._max_pitch)
        self._rx = trimesh.transformations.rotation_matrix(np.radians(self._pitch), [1, 0, 0])
        self._dirty = True

    def update_pitch(self, delta):
        self.set_pitch(self._pitch + delta)

    def get_distance(self):
        return self._distance

    def set_distance(self, value):
        self._distance = np.clip(value, self._znear, self._zfar)
        self._tz[2, 3] = self._distance
        self._dirty = True

    def update_distance(self, delta):
        self.set_distance(self._distance + delta)

    def get_center(self):
        return self._center
    
    def set_center(self, value):
        self._center = value
        self._tc[:3, 3] = self._center
        self._dirty = True

    def update_center(self, delta):
        self.set_center(self._center + delta)

    def get_matrix_center(self):
        return self._tc

    def get_matrix_yaw(self):
        return self._ry
    
    def get_matrix_pitch(self):
        return self._rx
    
    def get_matrix_distance(self):
        return self._tz

    def transform(self):
        if (self._dirty):
            self._pose = self._tc @ self._ry @ self._rx @ self._tz
            self._dirty = False
        return self._pose


class renderer:
    def __init__(self, settings_offscreen, settings_scene, settings_camera, settings_camera_transform, settings_lamp):
        self._renderer = pyrender.OffscreenRenderer(**settings_offscreen)
        self._scene = pyrender.Scene(**settings_scene)
        self._camera = pyrender.IntrinsicsCamera(**settings_camera)
        self._camera_transform = renderer_camera_transform(**settings_camera_transform)
        self._light = pyrender.DirectionalLight(**settings_lamp)
        self._groups = dict()

        self._camera_pose = self._camera_transform.transform()

        self._node_camera = self._scene.add(self._camera, 'internal@main@camera', self._camera_pose)
        self._node_light = self._scene.add(self._light, 'internal@main@lamp', self._camera_pose)

    def _camera_set_pose(self, camera_pose):
        self._camera_pose = camera_pose

        self._scene.set_pose(self._node_camera, self._camera_pose)
        self._scene.set_pose(self._node_light, self._camera_pose)

    def camera_get_pose(self):
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

    def camera_adjust(self, yaw=None, pitch=None, distance=None, center=None, relative=True):
        if (yaw is not None):
            if (relative):
                self._camera_transform.update_yaw(yaw)
            else:
                self._camera_transform.set_yaw(yaw)
        if (pitch is not None):
            if (relative):
                self._camera_transform.update_pitch(pitch)
            else:
                self._camera_transform.set_pitch(pitch)
        if (distance is not None):
            if (relative):
                self._camera_transform.update_distance(distance)
            else:
                self._camera_transform.set_distance(distance)
        if (center is not None):
            if (relative):
                self._camera_transform.update_center(center)
            else:
                self._camera_transform.set_center(center)
        self._camera_set_pose(self._camera_transform.transform())

    def camera_get_parameters(self):
        yaw = self._camera_transform.get_yaw()
        pitch = self._camera_transform.get_pitch()
        distance = self._camera_transform.get_distance()
        center = self._camera_transform.get_center()
        tc = self._camera_transform.get_matrix_center()
        ry = self._camera_transform.get_matrix_yaw()
        rx = self._camera_transform.get_matrix_pitch()
        tz = self._camera_transform.get_matrix_distance()
        return (yaw, pitch, distance, center, tc, ry, rx, tz)

