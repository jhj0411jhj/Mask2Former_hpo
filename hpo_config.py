"""
如果要添加超参数，请修改 get_configspace 和 HP_NAME_TO_YAML_KEY
"""
from openbox import space as sp

CONFIG_ROOT = "/root/jhj/cv/Mask2Former/configs/youtubevis_2019/swin/"
MODEL_ROOT = "/root/jhj/cv/models/"
SOLVER_CHECKPOINT_PERIOD = 2000     # default: 5000
SOLVER_IMS_PER_BATCH = 16           # default: 16
SOLVER_MAX_ITER = 6000              # default: 6000 (default STEPS: (4000, ))

LR_SCHEDULER_STEPS_MAPPING = {
    '2/3': [int(SOLVER_MAX_ITER * 2 / 3)],
    '1/2': [int(SOLVER_MAX_ITER / 2)],
    '1/2+3/4': sorted({int(SOLVER_MAX_ITER / 2), int(SOLVER_MAX_ITER * 3 / 4)}),
}

CONFIG_MAPPING = {
    'tiny': 'video_maskformer2_swin_tiny_bs16_8ep.yaml',
    'small': 'video_maskformer2_swin_small_bs16_8ep.yaml',
    'base': 'video_maskformer2_swin_base_IN21k_384_bs16_8ep.yaml',
    'large': 'video_maskformer2_swin_large_IN21k_384_bs16_8ep.yaml',
}
MODEL_MAPPING = {
    'tiny': 'model_final_86143f.pkl',
    'small': 'model_final_1e7f22.pkl',
    'base': 'model_final_83d103.pkl',
    'large': 'model_final_e5f453.pkl',
}
assert set(CONFIG_MAPPING.keys()) == set(MODEL_MAPPING.keys())

MAX_CKPT = 5
MASTER_CMD_PREFIX = ""
WORKER_CMD_PREFIX = "source /root/anaconda3/bin/activate jhj_mask && cd /root/jhj/cv/Mask2Former && "  # todo


def get_configspace(space_type='full'):
    base_lr = sp.Real("base_lr", 1e-5, 1e-3, default_value=0.0001, log=True)
    backbone_multiplier = sp.Real("backbone_multiplier", 0.01, 1.0, default_value=0.1, log=True)
    weight_decay = sp.Real("weight_decay", 1e-5, 0.5, default_value=0.05, log=True)
    optimizer = sp.Categorical("optimizer", choices=["SGD", "ADAMW"], default_value="ADAMW")
    sgd_momentum = sp.Real("sgd_momentum", 0.8, 0.99, default_value=0.9)
    clip_gradients_clip_value = sp.Ordinal(
        "clip_gradients_clip_value", sequence=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 0.0], default_value=0.01)
    # See detectron2.solver.build.build_lr_scheduler "WarmupMultiStepLR"
    lr_scheduler_steps = sp.Categorical("lr_scheduler_steps", choices=['2/3', '1/2', '1/2+3/4'], default_value='2/3')
    lr_scheduler_gamma = sp.Real("lr_scheduler_gamma", 0.01, 1.0, default_value=0.1, log=True)

    # model_num_object_queries = sp.Int("model_num_object_queries", 100, 900, default_value=200, q=100)
    model_num_object_queries = sp.Ordinal(
        "model_num_object_queries", sequence=[100, 200, 300, 600, 900], default_value=200)

    cond_optimizer = sp.EqualsCondition(sgd_momentum, optimizer, "SGD")

    space = sp.Space()
    if space_type == 'lr':
        space.add_variables([base_lr])
    elif space_type == 'small':
        space.add_variables([base_lr, weight_decay])
    elif space_type == 'mid':
        space.add_variables([base_lr, weight_decay, optimizer, sgd_momentum])
        space.add_conditions([cond_optimizer])
    elif space_type == 'full':
        space.add_variables([
            base_lr, backbone_multiplier, weight_decay, optimizer, sgd_momentum, clip_gradients_clip_value,
            lr_scheduler_steps, lr_scheduler_gamma,
        ])
        space.add_conditions([cond_optimizer])

    elif space_type == 'mid_v2':
        space.add_variables([base_lr, weight_decay, optimizer, sgd_momentum, model_num_object_queries])
        space.add_conditions([cond_optimizer])

    else:
        raise NotImplementedError(f'Unknown space_type: {space_type}')

    # check valid
    assert set(lr_scheduler_steps.choices) == set(LR_SCHEDULER_STEPS_MAPPING.keys())
    for hp in space.get_hyperparameter_names():
        assert hp in HP_NAME_TO_YAML_KEY, f'Please add {hp} to HP_NAME_TO_YAML_KEY!'
    return space


HP_NAME_TO_YAML_KEY = {
    'base_lr': 'SOLVER.BASE_LR',
    'backbone_multiplier': 'SOLVER.BACKBONE_MULTIPLIER',
    'weight_decay': 'SOLVER.WEIGHT_DECAY',
    'optimizer': 'SOLVER.OPTIMIZER',
    'sgd_momentum': 'SOLVER.MOMENTUM',
    'clip_gradients_clip_value': 'SOLVER.CLIP_GRADIENTS.CLIP_VALUE',
    'lr_scheduler_steps': 'SOLVER.STEPS',
    'lr_scheduler_gamma': 'SOLVER.GAMMA',
    'model_num_object_queries': 'MODEL.MASK_FORMER.NUM_OBJECT_QUERIES',
}
