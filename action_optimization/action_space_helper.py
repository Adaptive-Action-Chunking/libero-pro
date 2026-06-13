from scipy.spatial.transform import Rotation as R_sci
import numpy as np
from action_optimization.action_entropy_pi05 import action_entropy_gausssian_bernoulli
import json
import math

# ==========================================
# 基于 SciPy 的 Agent 坐标系转换
# ==========================================
def transform_state_to_agent_frame(state_base, R_base2cam, T_base2cam, target_frame="camera"):
    pos_base = state_base[:3]
    rotvec_base = state_base[3:6]
    gripper_states = state_base[6:] 
    
    pos_cam = np.dot(pos_base, R_base2cam.T) + T_base2cam
    mat_base = R_sci.from_rotvec(rotvec_base).as_matrix()
    mat_cam = np.matmul(R_base2cam, mat_base)
    rotvec_cam = R_sci.from_matrix(mat_cam).as_rotvec()
    
    if target_frame == "image":
        pos_cam[1:3] *= -1.0
        rotvec_cam[1:3] *= -1.0
        
    return np.concatenate([pos_cam, rotvec_cam, gripper_states]).astype(np.float32)

def transform_action_to_base_frame_from_agent(actions_target, R_base2cam, source_frame="camera"):
    delta_pos_cam = actions_target[:, :3].copy()
    delta_rotvec_cam = actions_target[:, 3:6].copy()
    
    if source_frame == "image":
        delta_pos_cam[:, 1:3] *= -1.0
        delta_rotvec_cam[:, 1:3] *= -1.0
        
    delta_pos_base = np.dot(delta_pos_cam, R_base2cam)
    delta_rotvec_base = np.dot(delta_rotvec_cam, R_base2cam)
    
    return np.hstack([delta_pos_base, delta_rotvec_base]).astype(np.float32)

# ==========================================
# 基于 SciPy 的 Wrist 动态坐标系转换
# ==========================================
def transform_state_to_wrist_frame(state_base, T_ee_cam, target_frame="camera"):
    pos_base_ee = state_base[:3]
    rotvec_base_ee = state_base[3:6]
    gripper_states = state_base[6:] 

    R_ee2cam = T_ee_cam[:3, :3]
    t_ee2cam = T_ee_cam[:3, 3]

    R_base2ee = R_sci.from_rotvec(rotvec_base_ee).as_matrix()
    R_base2cam = np.matmul(R_base2ee, R_ee2cam)

    pos_cam_abs = np.dot(R_base2ee, t_ee2cam) + pos_base_ee
    rotvec_cam_abs = R_sci.from_matrix(R_base2cam).as_rotvec()

    if target_frame == "image":
        pos_cam_abs[1:3] *= -1.0
        rotvec_cam_abs[1:3] *= -1.0

    return np.concatenate([pos_cam_abs, rotvec_cam_abs, gripper_states]).astype(np.float32)

def transform_action_to_base_frame_from_wrist(actions_target, current_state_base, T_ee_cam, source_frame="camera"):
    delta_pos_cam = actions_target[:, :3].copy()
    delta_rotvec_cam = actions_target[:, 3:6].copy()
    
    if source_frame == "image":
        delta_pos_cam[:, 1:3] *= -1.0
        delta_rotvec_cam[:, 1:3] *= -1.0

    rotvec_base_ee = current_state_base[3:6]
    R_base2ee = R_sci.from_rotvec(rotvec_base_ee).as_matrix()
    R_ee2cam = T_ee_cam[:3, :3]
    R_base2cam = np.matmul(R_base2ee, R_ee2cam)
        
    delta_pos_base = np.dot(delta_pos_cam, R_base2cam.T)
    delta_rotvec_base = np.dot(delta_rotvec_cam, R_base2cam.T)
    
    return np.hstack([delta_pos_base, delta_rotvec_base]).astype(np.float32)


# ==========================================
# 动作评估相关辅助函数
# ==========================================
def compute_consistency(pred_action_dict, normalize_action=True):
    """
    通过计算 20 个采样轨迹的一致性（方差）来衡量置信度。
    方差越小，说明模型对其预测越笃定，置信度越高。
    """
    # 形状期望为 (B=20, T, D)
    if normalize_action:
        pos = pred_action_dict['normalized_action'][:, :, :3]
        rot = pred_action_dict['normalized_action'][:, :, 3:6]
    else:
        pos = pred_action_dict["action.end_effector_position"]
        rot = pred_action_dict["action.end_effector_rotation"]
    
    # 沿 Batch 维度求方差，然后再所有维度求均值，得到标量分数
    var_pos = np.mean(np.var(pos, axis=0))
    var_rot = np.mean(np.var(rot, axis=0))
    
    return var_pos + var_rot


def select_view(pred_action_dict_agent, pred_action_dict_wrist, chunk_size):
    entropy_agent  = action_entropy_gausssian_bernoulli(pred_action_dict_agent)["chunk_mean"][chunk_size-1]
    entropy_wrist  = action_entropy_gausssian_bernoulli(pred_action_dict_wrist)["chunk_mean"][chunk_size-1]
    if entropy_agent < entropy_wrist:
        view = "agent"
    else:
        view = "wrist"  
    return view

"""提取转换所需的全部外参矩阵 (Agent和Wrist同时加载)"""
def get_camera_extrinsics(extrinsics_path):
    lang_to_ext_dict = {}
    with open(extrinsics_path, "r", encoding="utf-8") as f:
        extrinsics_dict = json.load(f)
    for suite, tasks in extrinsics_dict.items():
        for task_desc, matrices in tasks.items():
            lang_to_ext_dict[task_desc] = {
                "R_agent": np.array(matrices["R_base2cam"]),
                "T_agent": np.array(matrices["T_base2cam"]),
                "T_wrist": np.array(matrices["T_ee_cam"])
            }
    return lang_to_ext_dict


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def collect_obs(action_space, obs, task_description, lang_to_ext_dict):
    if action_space == "base" or action_space == "dual_train":
        obs_dict = {}
        obs_dict['video.wrist_view'] = np.copy( obs["robot0_eye_in_hand_image"][::-1, ::-1, :])[None, ...]
        obs_dict['video.front_view'] = np.copy( obs["agentview_image"][::-1, ::-1, :]) [None, ...]
        obs_dict["state.end_effector_position"] = obs["robot0_eef_pos"][None, ...]
        obs_dict["state.end_effector_rotation"] = _quat2axisangle(obs["robot0_eef_quat"])[None, ...]
        obs_dict["state.gripper_qpos"] = obs["robot0_gripper_qpos"][None, ...]
        obs_dict["annotation.human.action.task_description"] = [task_description]
        return obs_dict
    
    # 加载当前任务的双路坐标系外参
    R_base2cam, T_base2cam, T_ee_cam = None, None, None
    if task_description not in lang_to_ext_dict:
        raise ValueError(f"Extrinsics not found for task: '{task_description}'")
    R_base2cam = lang_to_ext_dict[task_description]["R_agent"]
    T_base2cam = lang_to_ext_dict[task_description]["T_agent"]
    T_ee_cam = lang_to_ext_dict[task_description]["T_wrist"]
    current_state_base = np.concatenate((
                    obs["robot0_eef_pos"], 
                    _quat2axisangle(obs["robot0_eef_quat"]), 
                    obs["robot0_gripper_qpos"]
                ))
    
    if action_space == "agent":
        state_agent = transform_state_to_agent_frame(current_state_base, R_base2cam, T_base2cam, "image")
        obs_dict_agent = {}
        obs_dict_agent['video.wrist_view'] = np.copy(obs["robot0_eye_in_hand_image"][::-1, ::-1, :])[None, ...]
        obs_dict_agent['video.front_view'] = np.copy(obs["agentview_image"][::-1, ::-1, :]) [None, ...]
        obs_dict_agent["state.end_effector_position"] = state_agent[:3][None, ...]
        obs_dict_agent["state.end_effector_rotation"] = state_agent[3:6][None, ...]
        obs_dict_agent["state.gripper_qpos"] = state_agent[6:][None, ...]
        obs_dict_agent["annotation.human.action.task_description"] = [task_description]
        return obs_dict_agent
    elif action_space == "wrist":
        # 构造 Wrist 模型的输入
        state_wrist = transform_state_to_wrist_frame(current_state_base, T_ee_cam, "image")
        obs_dict_wrist = {}
        obs_dict_wrist['video.wrist_view'] = np.copy(obs["robot0_eye_in_hand_image"][::-1, ::-1, :])[None, ...]
        obs_dict_wrist['video.front_view'] = np.copy(obs["agentview_image"][::-1, ::-1, :]) [None, ...]
        obs_dict_wrist["state.end_effector_position"] = state_wrist[:3][None, ...]
        obs_dict_wrist["state.end_effector_rotation"] = state_wrist[3:6][None, ...]
        obs_dict_wrist["state.gripper_qpos"] = state_wrist[6:][None, ...]
        obs_dict_wrist["annotation.human.action.task_description"] = [task_description]
        return obs_dict_wrist
    
def select_view(pred_action_dict_agent, pred_action_dict_wrist, chunk_size):
    entropy_agent  = action_entropy_gausssian_bernoulli(pred_action_dict_agent)["chunk_mean"][chunk_size-1]
    entropy_wrist  = action_entropy_gausssian_bernoulli(pred_action_dict_wrist)["chunk_mean"][chunk_size-1]
    if entropy_agent < entropy_wrist:
        view = "agent"
    else:
        view = "wrist"  
    return view