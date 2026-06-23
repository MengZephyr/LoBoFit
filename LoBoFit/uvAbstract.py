import numpy as np
import os
from data_IO import savePly

def readTextureOBJFile(fname):
    if not (os.path.exists(fname)):
        print("no file exit: ", fname)
        return None, None, None, None
    VerFaceArray = []
    VerArray = []
    txtfaceArray = []
    txtArray = []
    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = [float(x) for x in values[1:4]]
            VerArray.append(v)
        if values[0] == 'vt':
            vt = [float(x) for x in values[1:3]]
            txtArray.append(vt)
        if values[0] == 'f':
            f = [int(x.split('/')[1])-1 for x in values[1:4]]
            txtfaceArray.append(f)
            f = [int(x.split('/')[0])-1 for x in values[1:4]]
            VerFaceArray.append(f)

    txtfaceArray = np.array(txtfaceArray)
    txtArray = np.array(txtArray)
    VerFaceArray = np.array(VerFaceArray)
    VerArray = np.array(VerArray)

    return txtArray, txtfaceArray, VerArray, VerFaceArray

def geo_uv_map(geo_faces, uv_faces, geo_verts, uv_verts, savename):
    assert geo_faces.shape[0] == uv_faces.shape[0]

    num_gv = int(geo_verts.shape[0])
    num_uv = int(uv_verts.shape[0])
    num_f = int(geo_faces.shape[0])

    print('gv: ', num_gv, ' uv: ', num_uv, ' f: ', num_f)
    # ---------- uv -> geo (1-to-1) ----------
    u_to_g = [-1 for _ in range(num_uv)]
    for f in range(num_f):
        uface = uv_faces[f]  # (3,)
        gface = geo_faces[f]  # (3,)
        for i in range(3):
            uvi = int(uface[i])
            gvi = int(gface[i])
            if u_to_g[uvi] < 0:
                u_to_g[uvi] = gvi

    u2g_path = f"{savename}_u_to_g.txt"
    with open(u2g_path, "w", encoding="utf-8") as f:
        for vi in range(num_uv):
            f.write(f"1 {int(u_to_g[vi])}\n")

    # ---------- geo -> uv (1-to-N) ----------
    g_to_u = [[] for _ in range(num_gv)]
    for vi in range(num_uv):
        gid = int(u_to_g[vi])
        if gid < 0:
            print(vi)
        else:
            g_to_u[gid].append(vi)

    g2u_path = f"{savename}_g_to_u.txt"
    with open(g2u_path, "w", encoding="utf-8") as f:
        for gvi in range(num_gv):
            row = g_to_u[gvi]
            count = len(row)
            if count < 1:
                print("invalid g_to_u mapping")
            f.write(str(count))
            for uvi in row:
                f.write(f" {int(uvi)}")
            f.write("\n")

    print("Mapping Done.")
    return


def textureInfoGrab(fileName, saveName, ifmap=True):
    txtArray, txtfaceArray, VerArray, VerFaceArray = readTextureOBJFile(fileName)
    print(txtArray.shape, txtfaceArray.shape)
    print(VerArray.shape, VerFaceArray.shape)
    z_t = np.zeros((txtArray.shape[0], 1))
    z_texts = np.concatenate([txtArray, z_t], axis=-1)

    savePly(saveName + '_uv.ply', z_texts, colors=192 * np.ones_like(z_texts), faces=txtfaceArray)  # face v/vt/vn
    savePly(saveName + '_geo.ply', VerArray, colors=192 * np.ones_like(VerArray), faces=VerFaceArray)

    if ifmap:
        geo_uv_map(geo_faces=VerFaceArray, uv_faces=txtfaceArray, geo_verts=VerArray, uv_verts=txtArray,
                   savename=saveName)


if __name__ == '__main__':
    dress_name = 'Bodysuit'
    pd_name = 'PD10_G'
    prefix = 'D:/MyWork/2nd_Garment_Refitting/data/Manneq/WHip_Hop/' + dress_name + '/' + pd_name + '/A_pose/'
    textureInfoGrab(prefix+'A_pose.obj', prefix+'A_pose')



