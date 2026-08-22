# import numpy as np
# import torch
# import dill
# import os, sys
# import inspect
# current_file_path = os.path.abspath(__file__)
# parent_directory = os.path.dirname(current_file_path)
# sys.path.append(parent_directory)

# from pi_model import *
# def inspect_env_data(env, obs):
#     print("\n" + "="*50)
#     print("🔍 [DEBUG] 环境与数据探查模式")
#     print("="*50)

#     # 1. 检查 Observation 结构
#     print(f"\n1. Observation Keys: {list(obs.keys())}")
#     if "observation" in obs:
#         print(f"   observation['observation'] Keys: {list(obs['observation'].keys())}")
    
#     if "joint_action" in obs:
#         print(f"   observation['joint_action'] Keys: {list(obs['joint_action'].keys())}")
#         if "vector" in obs["joint_action"]:
#             print(f"   Current Joint Vector Shape: {obs['joint_action']['vector'].shape}")

#     # 2. 检查 TASK_ENV 的属性 (寻找 robot 对象)
#     print(f"\n2. TASK_ENV Attributes (Partial):")
#     env_attrs = dir(env)
#     # 过滤掉一些通用属性，只看重点
#     interesting_attrs = [a for a in env_attrs if not a.startswith('__')]
#     print(f"   Available: {interesting_attrs[:20]} ... (Total {len(interesting_attrs)})")
    
#     # 重点检查是否存在 left_robot / right_robot
#     has_left = hasattr(env, 'left_robot')
#     has_right = hasattr(env, 'right_robot')
#     print(f"   👉 Has 'left_robot'? {has_left}")
#     print(f"   👉 Has 'right_robot'? {has_right}")

#     # 3. 如果有 robot 对象，检查它有哪些方法 (寻找 get_end_effector_pose)
#     if has_left:
#         print(f"\n3. Investigating 'left_robot' methods:")
#         robot_attrs = dir(env.left_robot)
#         # 寻找包含 'pose', 'end', 'ee' 的方法
#         pose_methods = [m for m in robot_attrs if any(k in m.lower() for k in ['pose', 'end', 'ee', 'tip'])]
#         print(f"   Found potential pose methods: {pose_methods}")
        
#         # 尝试调用一下最像的
#         if 'get_end_effector_pose' in robot_attrs:
#             try:
#                 pose = env.left_robot.get_end_effector_pose()
#                 print(f"   ✅ 调用 get_end_effector_pose() 成功: {pose}")
#                 print(f"   ✅ 数据类型: {type(pose)}")
#             except Exception as e:
#                 print(f"   ❌ 调用 get_end_effector_pose() 失败: {e}")

#     # 4. 检查 take_action 的参数签名
#     print(f"\n4. Checking take_action signature:")
#     try:
#         sig = inspect.signature(env.take_action)
#         print(f"   Signature: {sig}")
#     except Exception as e:
#         print(f"   Cannot get signature: {e}")

#     print("\n" + "="*50)
#     print("🛑 DEBUG 完成，强制退出程序")
#     print("="*50)
#     sys.exit(0) # 查完直接退出，不继续跑

# # Encode observation for the model
# def encode_obs(observation):
#     input_rgb_arr = [
#         observation["observation"]["head_camera"]["rgb"],
#         observation["observation"]["right_camera"]["rgb"],
#         observation["observation"]["left_camera"]["rgb"],
#     ]
#     input_state = observation["joint_action"]["vector"]

#     return input_rgb_arr, input_state


# def get_model(usr_args):
#     train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
#                                                               usr_args["checkpoint_id"], usr_args["pi0_step"])
#     return PI0(train_config_name, model_name, checkpoint_id, pi0_step)


# def eval(TASK_ENV, model, observation):
    
#     # inspect_env_data(TASK_ENV, observation)
#     if model.observation_window is None:
#         instruction = TASK_ENV.get_instruction()
#         model.set_language(instruction)

#     input_rgb_arr, input_state = encode_obs(observation)
#     model.update_observation_window(input_rgb_arr, input_state)

#     # ======== Get Action ========

#     actions = model.get_action()[:model.pi0_step]

#     for action in actions:
#         TASK_ENV.take_action(action)
#         observation = TASK_ENV.get_obs()
#         input_rgb_arr, input_state = encode_obs(observation)
#         model.update_observation_window(input_rgb_arr, input_state)

#     # ============================


# def reset_model(model):
#     model.reset_obsrvationwindows()


import numpy as np
import torch
import dill
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *

# ==========================================
# [安全方案] 使用全局变量存储控制模式
# 避免直接修改 model 对象属性导致潜在报错
# ==========================================
CURRENT_CONTROL_MODE = 'joint'  # 默认为 joint

# ==========================================
# 辅助函数：EE 状态提取
# ==========================================
def extract_ee_state(observation):
    """从 observation['endpose'] 提取 16维 EE 状态"""
    try:
        endpose_dict = observation['endpose']
        l_pose = np.array(endpose_dict['left_endpose']).flatten()
        l_grip = np.array([endpose_dict['left_gripper']]).flatten()
        r_pose = np.array(endpose_dict['right_endpose']).flatten()
        r_grip = np.array([endpose_dict['right_gripper']]).flatten()
        return np.concatenate([l_pose, l_grip, r_pose, r_grip]).astype(np.float32)
    except KeyError:
        print("\033[91m[Warn] EE Mode requires 'endpose' in observation.\033[0m")
        return np.zeros(16, dtype=np.float32)

# ==========================================
# 核心逻辑 1: 通用编码函数
# ==========================================
def encode_obs(observation, override_state=None):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    
    if override_state is not None:
        input_state = override_state
    else:
        # Joint 模式默认路径
        input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_chunk_progress(task_env):
    current_step = getattr(task_env, "take_action_cnt", None)
    if current_step is None:
        current_step = getattr(task_env, "step_count", 0)
    step_limit = getattr(task_env, "step_lim", None)
    if step_limit is None or step_limit <= 1:
        return 0.0
    return float(np.clip(float(current_step) / max(float(step_limit) - 1.0, 1.0), 0.0, 1.0))

# ==========================================
# 核心逻辑 2: 模型加载与模式识别
# ==========================================
def get_model(usr_args):
    # 声明使用全局变量
    global CURRENT_CONTROL_MODE
    
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    
    # 实例化模型
    model = PI0(train_config_name, model_name, checkpoint_id, pi0_step, usr_args=usr_args)
    
    # --- 自动判断控制模式并更新全局变量 ---
    arg_mode = usr_args.get("control_mode", "").lower()
    arm_dim = usr_args.get("left_arm_dim", 6)
    
    if "ee" in arg_mode or arm_dim == 7:
        CURRENT_CONTROL_MODE = 'ee'
        print(f"\033[92m[Deploy] 检测到 EE 模式 (Dim={arm_dim}, Mode={arg_mode})，启用末端控制逻辑。\033[0m")
    else:
        CURRENT_CONTROL_MODE = 'joint'
        print(f"\033[93m[Deploy] 检测到 Joint 模式 (Dim={arm_dim})，启用关节控制逻辑。\033[0m")
        
    return model

# ==========================================
# 核心逻辑 3: 自适应评估循环
# ==========================================
def eval(TASK_ENV, model, observation):
    # 声明读取全局变量 (虽然读取不需要global关键字，但为了清晰加上也无妨)
    global CURRENT_CONTROL_MODE

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    # --- 步骤 A: 根据全局模式变量准备 State ---
    current_state = None
    if CURRENT_CONTROL_MODE == 'ee':
        current_state = extract_ee_state(observation)
    
    # 传入 encode_obs
    input_rgb_arr, input_state = encode_obs(observation, override_state=current_state)
    model.update_observation_window(
        input_rgb_arr,
        input_state,
        chunk_progress=get_chunk_progress(TASK_ENV),
    )

    # --- 步骤 B: 获取动作 ---
    actions = model.get_action()[:model.pi0_step]

    for action in actions:
        # --- 步骤 C: 根据全局模式变量执行动作 ---
        if CURRENT_CONTROL_MODE == 'ee':
            # EE 模式：显式调用 action_type='ee'
            try:
                TASK_ENV.take_action(action, action_type='ee')
            except TypeError:
                TASK_ENV.take_action(action) # Fallback
        else:
            # Joint 模式：默认调用
            TASK_ENV.take_action(action)
        
        # 更新观测
        observation = TASK_ENV.get_obs()
        
        # --- 步骤 D: 更新模型窗口 ---
        next_state = None
        if CURRENT_CONTROL_MODE == 'ee':
            next_state = extract_ee_state(observation)
            
        input_rgb_arr, input_state = encode_obs(observation, override_state=next_state)
        model.update_observation_window(
            input_rgb_arr,
            input_state,
            chunk_progress=get_chunk_progress(TASK_ENV),
        )

def reset_model(model):
    model.reset_obsrvationwindows()
