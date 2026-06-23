import os
import numpy as np
import torch


def float_numpy_to_tensor(x, device=None):
    if device is not None:
        return torch.from_numpy(x).type(torch.float32).to(device)
    else:
        return torch.from_numpy(x).type(torch.float32)


def float_tensor_to_numpy(x):
    return x.detach().cpu().numpy()


def int_numpy_to_tensor(x, device=None):
    if device is not None:
        return torch.from_numpy(x).type(torch.int).to(device)
    else:
        return torch.from_numpy(x).type(torch.int)


def int_tensor_to_numpy(x):
    return x.detach().cpu().numpy()


def readObj_faces(fname, tagID=0):
    if not (os.path.exists(fname)):
        return None
    faceArray = []
    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'f':
            f = [int(x.split('/')[tagID]) - 1 for x in values[1:4]]
            faceArray.append(f)
    file.close()

    faceArray = np.array(faceArray)
    return faceArray


def readObj_vert_feats(fname, flag='v'):
    if not (os.path.exists(fname)):
        print("!! No File !!: ", fname)
        return None
    posArray = []

    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == flag:
            if flag == 'vt':
                v = [float(x) for x in values[1:3]]
                posArray.append([v[0], v[1]])
            else:
                v = [float(x) for x in values[1:4]]
                posArray.append([v[0], v[1], v[2]])
    file.close()

    posArray = np.array(posArray)
    return posArray


def save_obj(filename, vertices, vnormal=None, faces=None, colors=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as fp:
        if colors is not None:
            for v, c in zip(vertices, colors):
                fp.write('v %f %f %f, %f, %f, %f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
        else:
            for v in vertices:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        if vnormal is not None:
            for nv in vnormal:
                fp.write('vn %f %f %f\n' % (nv[0], nv[1], nv[2]))

        if faces is not None:
            for f in (faces + 1):  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

        fp.close()


def save_index(filename, indices):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as fp:
        for i in indices:
            fp.write('%d ' % i)

        fp.close()


def save_segment_obj(filename, seg_head, seg_tail, seg_color=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    numseg = seg_head.shape[0]
    mtl_path = None
    if seg_color is not None:
        base, _ = os.path.splitext(filename)
        mtl_path = base + ".mtl"

        with open(mtl_path, "w", encoding="utf-8") as fm:
            for i in range(numseg):
                r, g, b = seg_color[i]
                fm.write(f"newmtl seg_{i}\n")
                fm.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")  # colors
                fm.write("Ka 0 0 0\nKs 0 0 0\nNs 1.0\nd 1.0\nillum 1\n\n")

    with open(filename, 'w') as fp:
        if mtl_path is not None:
            fp.write(f"mtllib {os.path.basename(mtl_path)}\n")

        for v in seg_head:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for v in seg_tail:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for i in range(numseg):
            if seg_color is not None:
                fp.write(f"usemtl seg_{i}\n")
            fp.write('l %d %d\n' % (i+1, numseg+i+1))
        fp.close()


def savePly(pDir, verts, *, colors=None, alpha=None, faces=None):
    numVerts = verts.shape[0]
    numFace = 0
    if faces is not None:
        numFace = faces.shape[0]
    with open(pDir, 'w') as f:
        f.write("ply\n" + "format ascii 1.0\n")
        f.write("element vertex " + str(numVerts) + "\n")
        f.write("property float x\n" + "property float y\n" + "property float z\n")
        if colors is not None:
            f.write("property uchar red\n" + "property uchar green\n"
                    + "property uchar blue\n" + "property uchar alpha\n")
        if faces is not None:
            f.write("element face " + str(numFace) + "\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for p in range(numVerts):
            v = verts[p]
            if colors is not None:
                c = colors[p]
                if alpha is not None:
                    a = alpha[p, 0]
                else:
                    a = 255
                f.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + " "
                        + str(int(c[0])) + " " + str(int(c[1])) + " " + str(int(c[2])) + " " + str(int(a)) + "\n")
            else:
                f.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")
        if faces is not None:
            for p in range(numFace):
                fds = faces[p]
                f.write("3 " + str(fds[0]) + " " + str(fds[1]) + " " + str(fds[2]) + "\n")
        f.close()


def readVertMapFile(fname):
    if not (os.path.exists(fname)):
        return None

    vertID = []
    file = open(fname, "r")
    for line in file:
        values = line.split()
        cc = int(values[0])
        cvert = []
        for ci in range(cc):
            cvert.append(int(values[1 + ci]))
        vertID.append(cvert)

    return vertID


def load_Geo_Different_Resolution_Sampling(sample_fname, target_g_to_u_fname):
    uv_in_coarse_faces, uv_in_coarse_abc = readFaceSampleFile(sample_fname)
    geo_to_uv_vmap = readVertMapFile(target_g_to_u_fname)
    vmap = []
    for i in range(len(geo_to_uv_vmap)):
        vmap.append(geo_to_uv_vmap[i][0])
    geo_in_coarse_faces = uv_in_coarse_faces[vmap]
    geo_in_coarse_abc = uv_in_coarse_abc[vmap, :]
    return geo_in_coarse_faces, geo_in_coarse_abc


def readFaceSampleFile(fname, device=None):
    if not (os.path.exists(fname)):
        return None, None

    FID = []
    ABC = []
    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        fid = int(values[0])
        ab = [float(values[1]), float(values[2])]

        FID.append(fid)
        ABC.append([1 - ab[0] - ab[1], ab[0], ab[1]])

    FID = np.array(FID)
    ABC = np.array(ABC)

    if device is not None:
        FID = int_numpy_to_tensor(FID, device)
        ABC = float_numpy_to_tensor(ABC, device)

    return FID, ABC

