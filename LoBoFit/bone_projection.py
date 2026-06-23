import numpy as np
import os
import torch
from data_IO import float_numpy_to_tensor

KeyBonesNames = ["Head", "Neck", "Spine2", "Spine1", "Spine", "Hips",
                 "RightShoulder", "RightArm", "RightForeArm", "RightHand",
                 "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
                 "LeftUpLeg", "LeftLeg", "LeftFoot",
                 "RightUpLeg", "RightLeg", "RightFoot",
                 "LeftCrotch",  # addBones: hips head --> LeftUpLeg head
                 "RightCrotch",  # addBones: hips head --> RightUpLeg head
                 "LeftRib",  # addBones: Spine2 head --> LeftShoulder head
                 "RightRib"]  # addBones: Spine2 head --> LeftShoulder head

BonesID_dict = dict(zip(KeyBonesNames, [i for i in range(len(KeyBonesNames))]))

Spine_Chain = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"]
Left_Crotch_Chain = ["LeftCrotch", "LeftUpLeg", "LeftLeg", "LeftFoot"]
Right_Crotch_Chain = ["RightCrotch", "RightUpLeg", "RightLeg", "RightFoot"]
Left_Rib_Chain = ["LeftRib", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"]
Right_Rib_Chain = ["RightRib", "RightShoulder", "RightArm", "RightForeArm", "RightHand"]

def normalize_vector(v):
    v_len = np.linalg.norm(v)
    v = v / (v_len + 1e-8)
    return v


def norm_len(v):
    v_len = np.linalg.norm(v)
    return v_len


def create_bone_localCoord(head, tail, parent_x, parent_z):
    s = norm_len(tail - head)
    z = (tail - head) / (s + 1e-8)
    px_proj = np.dot(parent_x, z)
    x_tilde = parent_x - px_proj * z
    if np.linalg.norm(x_tilde) < 1.e-6:
        if px_proj > 0:
            x_tilde = -parent_z
        else:
            x_tilde = parent_z
    x = normalize_vector(x_tilde)
    y = normalize_vector(np.cross(z, x))
    x = normalize_vector(np.cross(y, z))
    return x, y, z, s

def Relative_Vector_Projection(pnts: torch.float32, joints: torch.float32, bones_s: torch.float32,
                               bones_x: torch.float32, bones_y: torch.float32, bones_z: torch.float32):
    '''
    :param pnts: [N, 3]
    :param joints: [K, 3]
    :param bones_s: [K, 1], bones_x,y,z: [K, 3]
    :return: vec_r: [N, K, 3], len_r: [N, K], proj_abk: [N, K, 3]
    '''
    k = joints.shape[0]
    pp = pnts.unsqueeze(1).repeat(1, k, 1)  # [N, k, 3]
    vec_r = pp - joints.unsqueeze(0)  # [N, k, 3]
    len_r = torch.norm(vec_r, dim=-1)  # [N, k]

    rr = vec_r.unsqueeze(-2)  # [N, k, 1, 3]
    pa = torch.matmul(rr, bones_x.unsqueeze(-1))  # [N, k, 1, 1]
    pb = torch.matmul(rr, bones_y.unsqueeze(-1))  # [N, k, 1, 1]
    pk = torch.matmul(rr, bones_z.unsqueeze(-1))  # [N, k, 1, 1]

    ss = bones_s.unsqueeze(0)  # [1, k, 1]
    pa = pa.squeeze(-1) / ss
    pb = pb.squeeze(-1) / ss
    pk = pk.squeeze(-1) / ss
    proj_abk = torch.cat([pa, pb, pk], dim=-1)  # [N, k, 3]

    return vec_r, len_r, proj_abk


def relative_vector_projection_batched(
        pnts: torch.Tensor,  # [B,N,3]
        joints: torch.Tensor,  # [B,K,3]
        bones_s: torch.Tensor,  # [B,K,1]
        bones_x: torch.Tensor,  # [B,K,3]
        bones_y: torch.Tensor,  # [B,K,3]
        bones_z: torch.Tensor,  # [B,K,3]
        eps: float = 1e-8,
        if_flatten: bool = False
):
    """
    returns:
      vec_r   : [B,N,K,3]
      len_r   : [B,N,K]
      proj_abk: [B,N,K,3]
    """
    vec_r = pnts.unsqueeze(2) - joints.unsqueeze(1)  # [B,N,K,3]
    len_r = vec_r.norm(dim=-1)

    # local coordinate axes
    A = torch.stack([bones_x, bones_y, bones_z], dim=-1)  # [B,K,3,3]
    # p_local[b,n,k,m] = <p[b,n,:], axis[b,k,:,m]>
    p_local = torch.einsum('bnc,bkcm->bnkm', pnts, A)  # [B,N,K,3]
    j_local = torch.einsum('bkc,bkcm->bkm', joints, A)  # [B,K,3]
    # normalize by bone length
    scale = bones_s.unsqueeze(1)  # [B,1,K,1]
    proj_abk = (p_local - j_local.unsqueeze(1)) / (scale + eps)  # [B,N,K,3]

    if if_flatten:
        vec_r = vec_r.view(vec_r.shape[0], vec_r.shape[1], -1)  # [B,N,3K]
        proj_abk = proj_abk.view(proj_abk.shape[0], proj_abk.shape[1], -1)  # [B,N,3K]

    return vec_r, len_r, proj_abk


def Inverse_Relative_Projection(proj_abk: torch.float32, joints: torch.float32, bones_s: torch.float32,
                                bones_x: torch.float32, bones_y: torch.float32, bones_z: torch.float32,
                                bones_w: torch.float32):
    '''
    :param proj_abk: [n, k, 3]
    :param joints: [k, 3]
    :param bones_s: [k, 1]
    :param bones_x: [k, 3]
    :param bones_y: [k, 3]
    :param bones_z: [k, 3]
    :param bones_w: [n, k]

    :return: pnts : [n, 3]
    '''
    n = proj_abk.shape[0]
    ss = bones_s.unsqueeze(0).repeat(n, 1, 1)  # [n, k, 1]
    p_abk = ss * proj_abk  # [n, k, 3]
    pa = p_abk[:, :, 0].unsqueeze(-1)  # [n, k, 1]
    pb = p_abk[:, :, 1].unsqueeze(-1)
    pk = p_abk[:, :, 2].unsqueeze(-1)

    xx = bones_x.unsqueeze(0).repeat(n, 1, 1)  # [n, k, 3]
    yy = bones_y.unsqueeze(0).repeat(n, 1, 1)
    zz = bones_z.unsqueeze(0).repeat(n, 1, 1)

    vec_r = pa * xx + pb * yy + pk * zz  # [n, k, 3]
    jj = joints.unsqueeze(0).repeat(n, 1, 1)  # [n, k, 3]
    b_pnts = jj + vec_r  # [n, k, 3]
    pnts = torch.matmul(bones_w.unsqueeze(-2), b_pnts)  # [n, 1, 3]
    pnts = pnts.squeeze(-2)  # [n, 3]

    return pnts


def inverse_relative_projection_batched(
        proj_abk: torch.Tensor,  # [B, N, K, 3]
        joints: torch.Tensor,  # [B, K, 3]
        bones_s: torch.Tensor,  # [B, K, 1]
        bones_x: torch.Tensor,  # [B, K, 3]
        bones_y: torch.Tensor,  # [B, K, 3]
        bones_z: torch.Tensor,  # [B, K, 3]
        bones_w: torch.Tensor,  # [B, N, K]   sum_k =1
        eps: float = 1e-8
):
    """
    returns:
      pnts: [B, N, 3]
    """
    scale = bones_s.unsqueeze(1)  # [B, 1, K, 1]
    p_abk = proj_abk * (scale + eps)  # [B, N, K, 3]

    A = torch.stack([bones_x, bones_y, bones_z], dim=-1)  # [B, K, 3, m]

    # local to world
    # A: [B,K,3,3], p_abk: [B,N,K,3]  => v_world: [B,N,K,3]
    v_world = torch.einsum('bkcm,bnkm->bnkc', A, p_abk)

    # joints: [B,K,3] -> [B,1,K,3]
    b_pnts = joints.unsqueeze(1) + v_world  # [B, N, K, 3]

    # pnts[b,n,:] = sum_k w[b,n,k] * b_pnts[b,n,k,:]
    pnts = torch.einsum('bnk,bnkc->bnc', bones_w, b_pnts)  # [B, N, 3]
    return pnts

def readBoneFile(fname, ifaddbones=True):
    if not (os.path.exists(fname)):
        print('!!No File!: ', fname)
        return None, None
    file = open(fname, "r")

    line = file.readline()
    values = line.split()
    numBones = int(values[0])

    headArray = []
    tailArray = []
    for i in range(numBones):
        line = file.readline()
        values = line.split()
        hp = [float(values[0]), float(values[2]), -float(values[1])]  # Blend: forward -z, up y
        tp = [float(values[3]), float(values[5]), -float(values[4])]
        headArray.append(hp)
        tailArray.append(tp)
    file.close()

    if ifaddbones:
        '''leftCrotch'''
        headArray.append(headArray[BonesID_dict["Hips"]])
        tailArray.append(headArray[BonesID_dict["LeftUpLeg"]])
        '''rigthCrotch'''
        headArray.append(headArray[BonesID_dict["Hips"]])
        tailArray.append(headArray[BonesID_dict["RightUpLeg"]])
        '''leftRib'''
        headArray.append(headArray[BonesID_dict["Spine2"]])
        tailArray.append(headArray[BonesID_dict["LeftShoulder"]])
        '''rightRib'''
        headArray.append(headArray[BonesID_dict["Spine2"]])
        tailArray.append(headArray[BonesID_dict["RightShoulder"]])

    headArray = np.array(headArray)
    tailArray = np.array(tailArray)

    return headArray, tailArray

def correct_bone_tails(headArray, tailArray):
    chain_list = [Spine_Chain, Left_Crotch_Chain, Left_Rib_Chain, Right_Crotch_Chain, Right_Rib_Chain]
    for chain in chain_list:
        for bi in range(len(chain) - 1):
            bn = chain[bi]
            next_bn = chain[bi + 1]
            next_head = headArray[BonesID_dict[next_bn]]
            tailArray[BonesID_dict[bn]] = next_head

    return tailArray

class BoneStructure(object):
    def __init__(self):
        self.headArray = None
        self.tailArray = None

        self.lx_coord = None
        self.ly_coord = None
        self.lz_coord = None
        self.bone_s = None

        self.numBones = len(KeyBonesNames)

    def get_bone_id(self, bname):
        return BonesID_dict[bname]

    def load_bone_from_fname(self, fname, if_correct_tail=True):
        self.headArray, self.tailArray = readBoneFile(fname)
        if if_correct_tail:
            self.tailArray = correct_bone_tails(self.headArray, self.tailArray)

    def load_bone_from_array(self, headArray, tailArray, if_correct_tail=True):
        self.headArray = headArray
        self.tailArray = tailArray
        if if_correct_tail:
            self.tailArray = correct_bone_tails(self.headArray, self.tailArray)

    def create_root_coord(self):
        if self.headArray is None:
            return
        '''
        hips(root) bone
        '''
        bone_id = BonesID_dict["Hips"]
        z = normalize_vector(self.tailArray[bone_id] - self.headArray[bone_id])
        bx_id = BonesID_dict["RightCrotch"]
        x = normalize_vector(self.tailArray[bx_id] - self.headArray[bx_id])
        y = normalize_vector(np.cross(z, x))
        x = normalize_vector(np.cross(y, z))

        return x, y, z, self.headArray[bone_id]

    def get_bone_info_by_name(self, name):
        if name not in BonesID_dict:
            return None, None
        if self.headArray is None:
            return None, None
        if self.lx_coord is None:
            return None, None

        bid = BonesID_dict[name]
        return self.headArray[bid], self.lx_coord[bid], self.ly_coord[bid], self.lz_coord[bid], self.bone_s[bid]

    def get_all_bone_info(self):
        if self.headArray is None:
            return None, None, None, None, None
        if self.lx_coord is None:
            return None, None, None, None, None

        return self.headArray, self.lx_coord, self.ly_coord, self.lz_coord, self.bone_s

    def get_all_bone_info_tensor_device(self, device):
        if self.headArray is None:
            return None, None, None, None, None
        if self.lx_coord is None:
            return None, None, None, None, None
        return float_numpy_to_tensor(self.headArray, device), float_numpy_to_tensor(self.lx_coord, device), \
            float_numpy_to_tensor(self.ly_coord, device), float_numpy_to_tensor(self.lz_coord, device), \
            float_numpy_to_tensor(self.bone_s, device)

    def create_local_coordArray(self):
        if self.headArray is None:
            return
        self.lx_coord = np.zeros((self.numBones, 3))
        self.ly_coord = np.zeros((self.numBones, 3))
        self.lz_coord = np.zeros((self.numBones, 3))
        self.bone_s = np.zeros((self.numBones, 1))

        '''
        hips(root) bone
        '''
        bone_id = BonesID_dict["Hips"]
        z = normalize_vector(self.tailArray[bone_id] - self.headArray[bone_id])
        bx_id = BonesID_dict["RightCrotch"]
        x = normalize_vector(self.tailArray[bx_id] - self.headArray[bx_id])
        y = normalize_vector(np.cross(z, x))
        x = normalize_vector(np.cross(y, z))

        self.lx_coord[bone_id] = x
        self.ly_coord[bone_id] = y
        self.lz_coord[bone_id] = z
        self.bone_s[bone_id] = norm_len(self.tailArray[bone_id] - self.headArray[bone_id])

        parent_x = x
        parent_z = z
        '''
        Spine_chain
        '''
        for i in range(1, len(Spine_Chain)):
            bone_id = BonesID_dict[Spine_Chain[i]]
            x, y, z, s = create_bone_localCoord(self.headArray[bone_id], self.tailArray[bone_id], parent_x, parent_z)
            self.lx_coord[bone_id] = x
            self.ly_coord[bone_id] = y
            self.lz_coord[bone_id] = z
            self.bone_s[bone_id] = s
            parent_x = x
            parent_z = z
        '''
        Left_Crotch_Chain
        '''
        parent_x = self.lx_coord[BonesID_dict["Hips"]]
        parent_z = self.lz_coord[BonesID_dict["Hips"]]
        for i in range(len(Left_Crotch_Chain)):
            bone_id = BonesID_dict[Left_Crotch_Chain[i]]
            x, y, z, s = create_bone_localCoord(self.headArray[bone_id], self.tailArray[bone_id], parent_x, parent_z)
            self.lx_coord[bone_id] = x
            self.ly_coord[bone_id] = y
            self.lz_coord[bone_id] = z
            self.bone_s[bone_id] = s
            parent_x = x
            parent_z = z
        '''
        Right_Crotch_Chain
        '''
        parent_x = self.lx_coord[BonesID_dict["Hips"]]
        parent_z = self.lz_coord[BonesID_dict["Hips"]]
        for i in range(len(Right_Crotch_Chain)):
            bone_id = BonesID_dict[Right_Crotch_Chain[i]]
            x, y, z, s = create_bone_localCoord(self.headArray[bone_id], self.tailArray[bone_id], parent_x, parent_z)
            self.lx_coord[bone_id] = x
            self.ly_coord[bone_id] = y
            self.lz_coord[bone_id] = z
            self.bone_s[bone_id] = s
            parent_x = x
            parent_z = z
        '''
        Left_Rib_Chain
        '''
        parent_x = self.lx_coord[BonesID_dict["Spine1"]]
        parent_z = self.lz_coord[BonesID_dict["Spine1"]]
        for i in range(len(Left_Rib_Chain)):
            bone_id = BonesID_dict[Left_Rib_Chain[i]]
            x, y, z, s = create_bone_localCoord(self.headArray[bone_id], self.tailArray[bone_id], parent_x, parent_z)
            self.lx_coord[bone_id] = x
            self.ly_coord[bone_id] = y
            self.lz_coord[bone_id] = z
            self.bone_s[bone_id] = s
            parent_x = x
            parent_z = z
        '''
        Right_Rib_Chain
        '''
        parent_x = self.lx_coord[BonesID_dict["Spine1"]]
        parent_z = self.lz_coord[BonesID_dict["Spine1"]]
        for i in range(len(Right_Rib_Chain)):
            bone_id = BonesID_dict[Right_Rib_Chain[i]]
            x, y, z, s = create_bone_localCoord(self.headArray[bone_id], self.tailArray[bone_id], parent_x, parent_z)
            self.lx_coord[bone_id] = x
            self.ly_coord[bone_id] = y
            self.lz_coord[bone_id] = z
            self.bone_s[bone_id] = s
            parent_x = x
            parent_z = z


Right_fingers = ['RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3',
                 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3',
                 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3',
                 'RightHandRing1', 'RightHandRing2', 'RightHandRing3',
                 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3']
Left_fingers = [ 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3',
                 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3',
                 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3',
                 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3',
                 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3']
Right_toes = ['RightToeBase']
Left_toes = ['LeftToeBase']

def expand_BoneID_dict():
    Expand_Dict = BonesID_dict.copy()
    # Exp_BoneID_dict and BonesID_dict
    for rf in Right_fingers:
        Expand_Dict[rf] = BonesID_dict["RightHand"]
    for lf in Left_fingers:
        Expand_Dict[lf] = BonesID_dict["LeftHand"]
    for rt in Right_toes:
        Expand_Dict[rt] = BonesID_dict["RightFoot"]
    for lt in Left_toes:
        Expand_Dict[lt] = BonesID_dict["LeftFoot"]
    return Expand_Dict


def create_body_bone_weights(bweiName):
    wei_info = np.load(bweiName)
    raw_wei = wei_info["verts_bweight"].astype(np.float32)
    wgroups = wei_info["bone_names"]
    wgroups = [n.rsplit(':', 1)[-1] for n in list(wgroups)]
    numB = len(BonesID_dict)
    numV = raw_wei.shape[0]
    weights = np.zeros((numV, numB), dtype=np.float32)

    Exp_Bone_ID = expand_BoneID_dict()
    raw_wei_valIDs = np.argwhere(raw_wei > 0)
    for p in list(raw_wei_valIDs):
        vid = p[0]
        r_bid = p[1]
        bname = wgroups[p[1]]
        bid = Exp_Bone_ID[bname]
        weights[vid, bid] += raw_wei[vid, r_bid]

    for v in range(numV):
        sw = np.sum(weights[v,:])
        if sw > 0.:
            weights[v, :] = weights[v, :] / sw

    return weights

def select_most_related_bones_by_abkw(abk: torch.Tensor,       # [N, J, 3]
                                      w: torch.Tensor,         # [N, J]
                                      weight_thresh: float = 4e-1, prefer_heavier: bool = True):
    k = abk[:, :, -1]
    k_clamp = k.clamp(0.0, 1.0)  # [N, J]
    ab = abk[:, :, :2]  # [N, J, 2]
    dist = (ab*ab).sum(dim=-1)
    extra = (k-k_clamp)**2
    dist = dist+extra
    mask = (w < weight_thresh)  # [N, J]
    dist = dist.masked_fill(mask, float(100))
    if prefer_heavier:
        eps = 1.e-8
        dist = dist/(w+eps)
    pick_bone = dist.argmin(dim=-1, keepdim=True)  # [N, 1]
    return pick_bone

def bone_related_cast_ray(abk: torch.Tensor,      # [N, J, 3]
                          pk_boneID: torch.Tensor,  # [N, 1]
                          joints: torch.Tensor,   # [J, 3]
                          bx_coor: torch.Tensor,
                          by_coor: torch.Tensor,
                          bz_coor: torch.Tensor,  # [J, 3]
                          blen: torch.Tensor):    # [J, 1]
    sq_bid = pk_boneID.squeeze(-1)
    p_abk = abk[torch.arange(abk.shape[0]).to(abk.device), sq_bid, :]   # [N, 3]
    p_a = p_abk[:, 0].unsqueeze(dim=-1)  # [N, 1]
    p_b = p_abk[:, 1].unsqueeze(dim=-1)  # [N, 1]
    p_k = p_abk[:, 2].unsqueeze(dim=-1)  # [N,1]

    p_joints = joints[sq_bid, :]  # [N, 3]
    p_bx = bx_coor[sq_bid, :]
    p_by = by_coor[sq_bid, :]
    p_bz = bz_coor[sq_bid, :]  # [N, 3]
    p_blen = blen[sq_bid, :]   # [N, 1]

    p_orig = p_joints + p_k*p_blen*p_bz  # [N, 3]
    p_dir = p_a*p_bx + p_b*p_by
    p_dir = torch.nn.functional.normalize(p_dir, dim=-1, eps=1.e-6)  # [N, 3]

    return p_orig, p_dir