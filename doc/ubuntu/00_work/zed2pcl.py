import os, open3d, quaternion, functools, itertools, struct, time, json
import numpy as np
import vectormath as vmath
from easydict import EasyDict
from PIL import Image

verbose = True


def stop_watch(func):
    @functools.wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        if verbose:
            print(f"{func.__name__}() elaplsed: {time.time() - start:.03f} [sec]")
        return result

    return wrapper


def make_pcd(points, colors):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    return pcd


@stop_watch
def calcurate_xyz(color, depth, intr):
    """
  camera. Given depth value d at (u, v) image coordinate, the corresponding 3d point is:
    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
  
  for example, depth value of ZED 2 is float32 and its value indicates meter directly.
  so we need to ignore "depth_scale" like open3d's RGBDImage style.
  """
    # intr = menu.config[menu.config['table'][menu.reso]]
    ws, hs = ([a for a in range(depth.shape[n])] for n in [0, 1])
    points, colors = [], []
    for u, v in itertools.product(ws, hs):
        z = float(depth[u, v])
        if np.isnan(z) or np.isinf(z):
            continue
        x = float((u - intr.cx) * z / intr.fx)
        y = float((v - intr.cy) * z / intr.fy)
        point = np.asarray([x, y, z])
        points.append(point)
        rgb = (color[u, v][:3] / 255).astype(np.float64)
        colors.append(rgb)
        # colors.append((color[u, v][:3]).astype(np.uint8))
    pcd = make_pcd(np.array(points), colors)
    return pcd  # if not verbose else pcd, points, colors


def create_pcd_from_float_depth(menu):
    # ZED2 cam returns depth map with float32,
    #   and open3d's create_from_depth_image() does not support float.
    color, depth = [
        np.load(os.path.join(data_dir, 'camera{}_{}_{}.npy'.format(menu.cam_id, menu.reso, c_or_d))) \
        for c_or_d in ['color', 'depth']]
    window = menu.frames[menu.cam_id]['window']
    color = convert_bgra2rgba(color)
    if menu.option.with_frame and (np.array(window) != None).all():
        color, depth, depth_med = add_frame2img(menu, color, depth)
        if menu.frames != None:
            menu.frames[menu.cam_id].depth_med = depth_med
    pcd = calcurate_xyz(menu, color, depth)
    if menu.option.correction:
        pass
    return pcd


def add_frame2img(color, depth, window_frame):
    h0, w0, h1, w1 = window_frame
    # prepare data
    img = np.zeros(color.shape).astype(color.dtype)
    img[:] = color[:]
    dep = np.zeros(depth.shape).astype(depth.dtype)
    dep[:] = depth[:]
    # write frame line with red
    red = [255, 0, 0, 255]
    img[h0:h1, w0] = red
    img[h0:h1, w1] = red
    img[h0, w0:w1] = red
    img[h1, w0:w1] = red
    # calcurate median of depth value inside of frame line
    target_depth = depth[h0:h1, w0:w1]
    ws, hs = [[a for a in range(n)] for n in target_depth.shape]
    depth_vals = np.array([target_depth[w, h] for w, h in itertools.product(ws, hs) if
                           not (np.isnan(target_depth[w, h]) or np.isinf(target_depth[w, h]))])
    depth_med = np.median(depth_vals)
    print('depth_med:', depth_med)
    # fill depth_med value in frame line
    dep[h0:h1, w0] = depth_med
    dep[h0:h1, w1] = depth_med
    dep[h0, w0:w1] = depth_med
    dep[h1, w0:w1] = depth_med
    return img, dep, depth_med  # depth_med is needed for merge some cam's pcd data.


def convert_bgra2rgba(img):
    # ZED2 numpy color is BGRA.
    rgba = np.zeros(img.shape).astype(np.uint8)
    rgba[:, :, 0] = img[:, :, 2]  # R
    rgba[:, :, 1] = img[:, :, 1]  # G
    rgba[:, :, 2] = img[:, :, 0]  # B
    rgba[:, :, 3] = img[:, :, 3]  # A
    return rgba


def get_quaternion_from_vectors(target=None, source=None):
    """
    参考URL
    https://knowledge.shade3d.jp/knowledgebase/2%E3%81%A4%E3%81%AE%E3%83%99%E3%82%AF%E3%83%88%E3%83%AB%E3%81%8C%E4%BD%9C%E3%82%8B%E5%9B%9E%E8%BB%A2%E8%A1%8C%E5%88%97%E3%82%92%E8%A8%88%E7%AE%97-%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%97%E3%83%88
    ベクトルaをbに向かせるquaternionを求める.
  """
    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    normalize = lambda x: np.array(vmath.Vector3(x).normalize())
    src_vec = normalize(source)
    target_vec = normalize(target)
    cross_vec = np.cross(target_vec, src_vec)
    cross_vec_norm = -np.linalg.norm(cross_vec)
    cross_vec = normalize(cross_vec)
    epsilon = 0.0002
    inner_vec = np.dot(src_vec, target_vec)
    if -epsilon < cross_vec_norm or 1.0 < inner_vec:
        if inner_vec < (epsilon - 1.0):
            trans_axis_src = np.array([-src_vec[1], src_vec[2], src_vec[0]])
            c = normalize(np.cross(trans_axis_src, src_vec))
            qw = 0.0
            qx = c[0]
            qy = c[1]
            qz = c[2]
    else:
        e = cross_vec * math.sqrt(0.5 * (1.0 - inner_vec))
        qw = math.sqrt(0.5 * (1.0 + inner_vec))
        qx = e[0]
        qy = e[1]
        qz = e[2]
    return np.quaternion(qw, qx, qy, qz)


def extract_red_point(pcd):
    ext = EasyDict({})
    ext.p, ext.c = [], []
    points, colors = np.array(pcd.points), np.array(pcd.colors)
    for i in range(points.shape[0]):
        p, c = points[i], colors[i]
        if not (c == np.array([1., 0., 0.]).astype(c.dtype)).all():
            continue
        else:
            ext.p.append(p)
            ext.c.append(c)
    return make_pcd(ext.p, ext.c)


def find_plane_from_pcd(pcd):
    # 参考URL: https://qiita.com/sage-git/items/f64620d18eeff8a11308
    r = np.array(pcd.points)
    c = np.mean(r, axis=0)
    r0 = r - c
    u, s, v = np.linalg.svd(r0)
    nv = v[-1, :]  # 最小のsに対応するベクトル
    ds = np.dot(r, nv)  # サンプル点群の平面と原点との距離
    param = np.r_[nv, -np.mean(ds)]
    return param


def zed_depthfloat_to_abgr(f):
    """
    ZED pcd data format:
      ----------------------------------------------------------
      https://www.stereolabs.com/docs/depth-sensing/using-depth/
      ----------------------------------------------------------
      The point cloud stores its data on 4 channels using 32-bit 
      float for each channel. 
      The last float is used to store color information, where 
      R, G, B, and alpha channels (4 x 8-bit) are concatenated 
      into a single 32-bit float. 
  """
    # https://stackoverflow.com/questions/23624212/how-to-convert-a-float-into-hex/38879403
    if f == 0.:
        return [0, 0, 0, 0]
    else:
        h = hex(struct.unpack('<I', struct.pack('<f', f))[0])
        return [eval('0x' + a + b) for a, b in zip(h[::2], h[1::2]) if a + b != '0x']


# https://takala.tokyo/takala_wp/2018/11/28/736/
np_zed_depthfloat_to_abgr = np.frompyfunc(zed_depthfloat_to_abgr, 1, 1)


@stop_watch
def convert_zed2pcd_to_ply(zed_pcd):
    zed_points = zed_pcd[:, :, :3]
    zed_colors = zed_pcd[:, :, 3]
    points, colors = [], []
    for x, y in itertools.product(
            [a for a in range(zed_colors.shape[0])],
            [a for a in range(zed_colors.shape[1])]):
        if zed_points[x, y].sum() == 0. and zed_points[x, y].max() == 0.:
            continue
        if np.isinf(zed_points[x, y]).any() or np.isnan(zed_points[x, y]).any():
            continue
        # color
        tmp = zed_depthfloat_to_abgr(zed_colors[x, y])
        tmp = [tmp[3], tmp[2], tmp[1]]  # ABGR to RGB
        tmp = np.array(tmp).astype(np.float64) / 255.  # for ply (color is double)
        colors.append(tmp)
        # point
        tmp = np.array(zed_points[x, y]).astype(np.float64)
        points.append(tmp)
    return make_pcd(points, colors)


@stop_watch
def apply_rotation(p):
    a = EasyDict({x.split('.')[0]: f'{p}/{x}' for x in os.listdir(p)})
    assert 'transform' in a.keys()
    assert ('pcd' in a.keys()) or ('image' in a.keys() and 'depth' in a.keys())
    # check original ply
    if not 'pcd_original' in a.keys():
        if 'pcd' in a.keys():
            zed2pcd = np.load(a.pcd)
            # create pcd from zed2's specific format
            pcd = convert_zed2pcd_to_ply(zed2pcd)
        elif 'image' in a.keys() and 'depth' in a.keys():
            color = np.load(a.image)
            color = convert_bgra2rgba(color)
            depth = np.load(a.depth)
            size = f'{color.shape[1]}x{color.shape[0]}'
            res = zed2cam.cam_reso.size2name[size]
            intr = zed2cam.cam_reso[res]
            pcd = calcurate_xyz(color, depth, intr)
        else:
            print(f'there is no pcd or image, please check directory: {p}')
            return
        # write original ply.
        open3d.io.write_point_cloud(f'{p}/pcd_original.ply', pcd)
    # check transformed ply
    if not 'pcd_transformed' in a.keys():
        zed2transform = np.load(f'{p}/transform.npy')
        # rotate pcd points by transform matrix.
        orig_points = np.array(pcd.points)
        one = np.ones((orig_points.shape[0], 1))
        x = zed2transform @ np.concatenate([orig_points, one], axis=1).T
        convert_points = x.T[:, :3]
        # write transformed pcd to ply.
        colors = np.array(pcd.colors)
        transformed_pcd = make_pcd(convert_points, colors)
        open3d.io.write_point_cloud(f'{p}/pcd_transformed.ply', transformed_pcd)


@stop_watch
def split_zed2pcd(p):
    if all([os.path.exists(f'{p}/{x}.npy') for x in ['depth', 'image']]):
        print('no need to split...')
        return
    zed2pcd = np.load(f'{p}/pcd.npy')
    zed_points = zed2pcd[:, :, :3]
    zed_colors = zed2pcd[:, :, 3]
    depth, color = np.zeros(zed_colors.shape), np.zeros(zed_points.shape)
    res = {}
    for x, y in itertools.product(
            [a for a in range(zed_colors.shape[0])],
            [a for a in range(zed_colors.shape[1])]):
        # color
        tmp = zed_depthfloat_to_abgr(zed_colors[x, y])
        color[x, y] = [tmp[3], tmp[2], tmp[1]]  # ABGR to RGB
        depth[x, y] = np.linalg.norm(zed_points[x, y], ord=2)
    np.save(f'{p}/depth_split.npy', depth)
    np.save(f'{p}/image_split.npy', color)


@stop_watch
def save_image_as_png(p, convert=True):
    color = np.load(f'{p}/image.npy')
    color = convert_bgra2rgba(color) if convert else color
    pil_img = Image.fromarray(color.astype(np.uint8))
    pil_img.save(f'{p}/image.png')


@stop_watch
def extract_in_4point_erea(d, is_wh=True):
    def make_ps(p0, p1, p2, p3):
        (a, b) = ('w', 'h') if is_wh == True else ('h', 'w')
        return [EasyDict({a: p[0], b: p[1]}) \
                for p in [p0, p1, p2, p3]]

    def get_center(ps):
        center = EasyDict({})
        center.w = np.array([p.w for p in ps]).mean()
        center.h = np.array([p.h for p in ps]).mean()
        return center

    def get_slope_intercention(p0, p1):
        slope = (p0.h - p1.h) / (p0.w - p1.w)
        intercept = p0.h - p0.w * slope
        return slope, intercept

    def check_in_line(p0, p1, p2):
        slope, intercept = get_slope_intercention(p0, p1)
        return p2.h == int(p2.w * slope + intercept)

    def check_mode(ps):
        mode, ignore, res = 'square', None, []
        for p0, p1, p2 in itertools.combinations(ps, 3):
            if check_in_line(p0, p1, p2):
                px = [p0, p1, p2]
                idx = np.array([p.w for p in px]).argsort()[1]
                ignore = px[idx]
                res.append([True, px, ignore])
            else:
                res.append([False, None, None])
        check = [x[0] for x in res]
        if all(check):
            raise Exception("all in 1 line!!!")
        elif any(check):
            mode = 'triangle'
            ignore = res[check.index(True)][2]
        return mode, ignore

    def check_pos(ps):
        res = EasyDict({})  # {right_up, left_up, left_down, right_down}
        center = get_center(ps)
        mode, ignore = check_mode(ps)
        if mode == 'square':
            w_indices = list(np.array([p.w for p in ps]).argsort())
            lefts = [ps[i] for i in w_indices[:2]]
            rights = [ps[i] for i in w_indices[2:]]
            res.left_up = lefts[lefts[0].h > lefts[1].h]
            res.left_down = lefts[not lefts[0].h > lefts[1].h]
            res.right_up = rights[rights[0].h > rights[1].h]
            res.right_down = rights[not rights[0].h > rights[1].h]
        elif mode == 'triangle':
            px = ps
            w_indices = list(np.array([p.w for p in ps]).argsort())
            res.start = ps[w_indices[0]]
            res.middle = ps[w_indices[1]]
            res.end = ps[w_indices[2]]
        assert (set(res.keys()) & set([f'{x}_{y}'
                                       for x, y in itertools.product(
                ['right', 'left'], ['up', 'down'])]) == set(res.keys())) \
               or (set(res.keys()) == set(['start', 'middle', 'end']))
        for k in res.keys():
            if len(res[k]) == 1:
                res[k] = res[k][0]
        return EasyDict(res), mode

    def get_area_of_pos(pos):
        ws, hs = sorted([pos[k].w for k in pos.keys()]), sorted([pos[k].h for k in pos.keys()])
        area = EasyDict({'w': {'min': ws[0], 'max': ws[-1]},
                         'h': {'min': hs[0], 'max': hs[-1]}})
        return area

    zed2pcd = np.load(f"{d}/pcd.npy")
    p0, p1, p2, p3 = json.load(open(f'{d}/frame.json'))['extract_frame_wh']
    ps = make_ps(p0, p1, p2, p3)
    assert all([len(p) == 2 and
                all([p[x] in [a for a in range(zed2pcd.shape[n])] for x, n in zip(['h', 'w'], [0, 1])])
                for p in ps])
    extracted_area = np.zeros(zed2pcd.shape).astype(zed2pcd.dtype)
    pos, mode = check_pos(ps)
    area = get_area_of_pos(pos)
    # vertical line scan algorythm
    if mode == 'square':
        start_p = 'left_up' if pos.left_up.w <= pos.left_down.w \
            else 'left_down'
        upper_p = 'right_up' if 'up' in start_p else 'left_up'
        lower_p = 'left_down' if 'up' in start_p else 'right_down'
        end_p = 'right_down' if 'right' in upper_p else 'right_up'
        for w in range(area.w.min, area.w.max + 1):
            if w == area.w.min or w == area.w.max:
                p = start_p if w == area.w.min else end_p
                w, h = pos[p].w, pos[p].h
                extracted_area[h, w, :] = zed2pcd[h, w, :]
            else:
                # edges
                upleft = start_p if w < pos[upper_p].w else upper_p
                upright = end_p if w >= pos[upper_p].w else upper_p
                lowleft = start_p if w < pos[lower_p].w else lower_p
                lowright = end_p if w >= pos[lower_p].w else lower_p
                # check slope
                up_slope = (pos[upright].h - pos[upleft].h) / \
                           (pos[upright].w - pos[upleft].w)
                up_h = int(up_slope * (w - pos[upleft].w) + pos[upleft].h)
                low_slope = (pos[lowright].h - pos[lowleft].h) / \
                            (pos[lowright].w - pos[lowleft].w)
                low_h = int(low_slope * (w - pos[lowleft].w) + pos[lowleft].h)
                extracted_area[up_h:low_h, w, :] = zed2pcd[up_h:low_h, w, :]
    else:
        print('*** WARN ***\n \
           Triangle mode is not implemented!!!\n \
           Please check frame.json and correct \n \
           position NOT to be triangle.')
        return
    pcd = convert_zed2pcd_to_ply(extracted_area)
    open3d.io.write_point_cloud(f'{d}/pcd_extracted.ply', pcd)