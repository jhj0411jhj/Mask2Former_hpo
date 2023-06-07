"""
如果num_machines>1，请确保已配置ssh免密登录
    https://www.thegeekstuff.com/2008/11/3-steps-to-perform-ssh-login-without-password-using-ssh-keygen-ssh-copy-id/
测试指令: ssh user@host -p port "pwd && python --version"
"""

import os
import sys
import json
import socket
import argparse
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from openbox import space as sp, Advisor, History, Observation, logger
from openbox.utils.constants import FAILED
from ConfigSpace.util import deactivate_inactive_hyperparameters
from detectron2.config import get_cfg
from utils import execute_command
from hpo_config import *

last_port = None


def get_port():
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def get_init_port():
        # PyTorch still may leave orphan processes in multi-gpu training.
        # Therefore we use a deterministic way to obtain port,
        # so that users are aware of orphan processes by seeing the port occupied.
        port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
        port = port - 2000
        return port

    global last_port
    if last_port is not None:
        port = last_port + 10
        if port > 65535:
            port = get_init_port()
    else:
        port = get_init_port()
    try_times = 0
    while is_port_in_use(port):
        port += 10
        if port > 65535:
            port = get_init_port()
        try_times += 1
        if try_times > 100:
            raise RuntimeError('Cannot find available port.')
    logger.info(f'last_port: {last_port}. Get port: {port}')
    last_port = port
    return port


def append_hp_to_cmd(cmd: str, config: sp.Configuration):
    config = config.get_dictionary().copy()
    k_pop = []
    for hp, value in config.items():
        if hp not in HP_NAME_TO_YAML_KEY:
            continue
        if hp in ['lr_scheduler_steps', ]:
            continue
        cmd += f' {HP_NAME_TO_YAML_KEY[hp]} {value}'
        k_pop.append(hp)
    for hp in k_pop:
        config.pop(hp)

    lr_scheduler_steps = config.pop('lr_scheduler_steps', '2/3')
    steps = LR_SCHEDULER_STEPS_MAPPING[lr_scheduler_steps]
    steps = ','.join(map(str, sorted(steps))) + ','
    cmd += f' {HP_NAME_TO_YAML_KEY["lr_scheduler_steps"]} {steps}'

    if len(config) > 0:
        raise ValueError(f'Unknown hp in config: {config}')
    return cmd


def retrieve_config(config_file: str, config_space: sp.ConfigurationSpace) -> sp.Configuration:
    assert os.path.exists(config_file), f'config_file not exists: {config_file}'
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    cfg.freeze()

    config_dict = {}
    for hp in config_space.get_hyperparameter_names():
        assert hp in HP_NAME_TO_YAML_KEY
        config_dict[hp] = eval('cfg.' + HP_NAME_TO_YAML_KEY[hp])

    if 'lr_scheduler_steps' in config_dict:
        lr_scheduler_steps = config_dict['lr_scheduler_steps']
        for k, v in LR_SCHEDULER_STEPS_MAPPING.items():
            if set(v) == set(lr_scheduler_steps):
                config_dict['lr_scheduler_steps'] = k
                break
        else:
            raise ValueError(f'Unknown lr_scheduler_steps: {lr_scheduler_steps}. '
                             f'LR_SCHEDULER_STEPS_MAPPING: {LR_SCHEDULER_STEPS_MAPPING}')
    if config_dict.get('sgd_momentum') == -1.0:
        config_dict.pop('sgd_momentum')  # backward compatibility

    config = deactivate_inactive_hyperparameters(configuration=config_dict, configuration_space=config_space)
    return config


def get_metrics(output_dir):
    # log_file = os.path.join(output_dir, 'log.txt')
    # with open(log_file, 'r') as f:
    #     lines = f.readlines()
    # final_line = lines[-1]
    # metrics = final_line.split('copypaste:')[1].strip(' ')
    # AP = float(metrics.split(',')[0])
    # loss = -AP
    try:
        file = os.path.join(output_dir, 'test_result.json')
        with open(file, 'r') as f:
            result = json.load(f)
        # negative (mean) Average Precision
        # to minimize
        if 'ytvis_2019_test' in result:
            val_neg_AP = -float(result['ytvis_2019_val']['segm']['AP'])
            test_neg_AP = -float(result['ytvis_2019_test']['segm']['AP'])
        else:
            val_neg_AP = -float(result['segm']['AP'])
            test_neg_AP = None
        metrics = {
            'val_neg_AP': val_neg_AP,
            'test_neg_AP': test_neg_AP,
        }
    except Exception:
        logger.exception('Exception in get_metric! output_dir: %s' % output_dir)
        raise
    return metrics


def objective_function(config, output_dir, args, cmd_prefix='', return_obs=True):
    start_time = time.time()
    extra_info = {'output_dir': output_dir, 'metrics': {}}

    num_machines = args.num_machines
    port = get_port()
    dist_url = f'tcp://{args.master}:{port}'  # todo: check url for master
    cmd = ' ' + cmd_prefix + (
        f" python train_net_video_new.py"
        f" --config-file {CONFIG_FILE}"
        f" --num-gpus {args.num_gpus}"
        f" --num-machines {num_machines}"
        f" --machine-rank %d"
        f" --dist-url {dist_url}"
        f" --resume"
        f" --max_ckpt {MAX_CKPT}"
        f" SEED 1"
        f" MODEL.WEIGHTS {MODEL_WEIGHTS}"
        f" SOLVER.CHECKPOINT_PERIOD {SOLVER_CHECKPOINT_PERIOD}"
        f" SOLVER.IMS_PER_BATCH {SOLVER_IMS_PER_BATCH}"
        f" SOLVER.MAX_ITER {SOLVER_MAX_ITER}"
        f" OUTPUT_DIR {output_dir}"
        # fr" DATASETS.TEST $'(\'ytvis_2019_val\',\'ytvis_2019_test\')'"
    )
    cmd = append_hp_to_cmd(cmd, config)

    if num_machines > 1:
        worker_list = args.workers.split(',')
        assert len(worker_list) == num_machines - 1, f'num_machines: {num_machines}, workers: {args.workers}'
        for i in range(1, num_machines):
            worker_host = worker_list[i - 1]
            host, port = worker_host.split(':') if ':' in worker_host else (worker_host, '22')

            worker_cmd = WORKER_CMD_PREFIX + cmd % i
            assert '"' not in worker_cmd, rf'worker_cmd: {worker_cmd}. try to use \' instead of "'
            # worker_cmd = f'pssh -H {worker_host}' + worker_cmd + ' &'
            worker_cmd = f'ssh {host} -p {port} "{worker_cmd}" >/dev/null 2>&1 &'
            p = execute_command(worker_cmd, f'train_worker{i}', capture_output=True, timeout=None, encoding='utf8',
                                max_output_line=100)

    master_cmd = MASTER_CMD_PREFIX + cmd % 0
    try:
        p = execute_command(master_cmd, 'train_master', capture_output=True, timeout=None, encoding='utf8',
                            max_output_line=100)  # may raise
    except Exception:
        elapsed_time = time.time() - start_time
        if return_obs and elapsed_time > 600:  # 10min. consider as failed because of config
            obs = Observation(config=config, objectives=[np.inf], elapsed_time=elapsed_time, extra_info=extra_info,
                              trial_state=FAILED)
            return obs
        else:  # something wrong happened. should check
            raise
    if not return_obs:
        return None
    metrics = get_metrics(output_dir)  # may raise
    objectives = [metrics['val_neg_AP']]
    extra_info['metrics'] = metrics
    elapsed_time = time.time() - start_time
    obs = Observation(config=config, objectives=objectives, elapsed_time=elapsed_time, extra_info=extra_info)
    return obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=2, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")

    parser.add_argument('--master', type=str, default='127.0.0.1', help='master ip')
    parser.add_argument('--workers', type=str, default='', help='user@host:port,user@host:port,...')

    parser.add_argument('--scale', type=str, choices=list(CONFIG_MAPPING.keys()), default='tiny')
    parser.add_argument('--space', type=str, default='full')
    parser.add_argument('--resume_task', type=str, default=None, help='resume an HPO task')
    parser.add_argument('--surrogate_type', type=str, default='gp')
    parser.add_argument('--acq_optimizer_type', type=str, default='local_random')
    parser.add_argument('--max_runs', type=int, default=20)
    parser.add_argument('--retrain_infer', action='store_true',
                        help='retrain the model on the whole training set and infer on the original test set')
    args = parser.parse_args()

    CONFIG_FILE = os.path.join(CONFIG_ROOT, CONFIG_MAPPING[args.scale])
    MODEL_WEIGHTS = os.path.join(MODEL_ROOT, MODEL_MAPPING[args.scale])

    assert os.path.exists(CONFIG_FILE), f'config file {CONFIG_FILE} not found'
    assert os.path.exists(MODEL_WEIGHTS), f'model_weights {MODEL_WEIGHTS} not found'

    # get task_id
    if args.resume_task is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        task_id = f'm2fswin_{args.scale}_{timestamp}'
    else:
        task_id = args.resume_task
    output_dir = f'output_hpo/{task_id}'

    logger.init(name='hpo', level='INFO', logdir=output_dir, force_init=True)
    logger.info('args: %s' % args)
    logger.info('task_id: %s' % task_id)
    logger.info('output_dir: %s' % os.path.abspath(output_dir))
    logger.info('CONFIG_FILE: %s' % CONFIG_FILE)
    logger.info('MODEL_WEIGHTS: %s' % MODEL_WEIGHTS)

    cs = get_configspace(args.space)
    logger.info('config_space: %s' % cs)
    # load hpo history
    history_path = os.path.join(output_dir, 'history.json')
    history = None if args.resume_task is None else History.load_json(history_path, cs)

    advisor = Advisor(
        config_space=cs,
        initial_trials=3,
        init_strategy='random_explore_first',
        surrogate_type=args.surrogate_type,
        acq_type='ei',
        acq_optimizer_type=args.acq_optimizer_type,
        rand_prob=0.1,
        task_id=task_id,
        logger_kwargs=dict(force_init=False),
        random_state=47,
    )
    if history is not None:
        advisor.history = history

    history = advisor.get_history()
    start_iter = len(history)
    max_iter = args.max_runs
    if start_iter >= max_iter:
        logger.warning(f'Already finished. start_iter={start_iter}, max_iter={max_iter}')
    try:
        for i in range(start_iter + 1, max_iter + 1):
            logger.info(f'===== start iter {i}/{max_iter} =====')
            iter_dir = os.path.join(output_dir, f'iter_{i:03d}')

            if i != start_iter + 1:
                assert not os.path.exists(iter_dir), f'iter_dir exists: {iter_dir}'
            next_iter_dir = os.path.join(output_dir, f'iter_{i + 1:03d}')
            assert not os.path.exists(next_iter_dir), f'next_iter_dir exists: {next_iter_dir}'
            # If resuming from history, make sure to suggest the same config
            config_file = os.path.join(iter_dir, 'config.yaml')
            if os.path.exists(config_file):
                config = retrieve_config(config_file, cs)
                logger.info(f'retrieve_config from {config_file}: {config}')
            else:
                assert not os.path.exists(os.path.join(iter_dir, 'last_checkpoint')), \
                    'no config.yaml but last_checkpoint exists in %s' % iter_dir
                config = advisor.get_suggestion()
                logger.info('get_suggestion: ' + str(config))

            obs = objective_function(config, output_dir=iter_dir, args=args)
            advisor.update_observation(obs)
            logger.info('update_observation: ' + str(obs))

            history.save_json(history_path)
            history.plot_convergence()
            plt.savefig(os.path.join(output_dir, 'convergence.jpg'))
            plt.close()
            time.sleep(60)

        # inference
        if args.retrain_infer:
            logger.info('===== start retrain inference =====')
            output_infer_dir = f'output_inference/retrain/{task_id}'
            default_iter = 1
            best_config = history.get_incumbent_configs()[0]
            best_iter = history.configurations.index(best_config) + 1
            logger.info(f'default_iter: {default_iter}, best_iter: {best_iter}')
            for i in sorted(list({default_iter, best_iter})):
                logger.info(f'===== start retrain inference iter {i} =====')
                iter_dir = os.path.join(output_infer_dir, f'iter_{i:03d}')
                if os.path.exists(iter_dir):
                    logger.warning(f'Caution! {iter_dir} exists.')
                config = history.configurations[i - 1]
                logger.info(f'config: {config}')
                cmd_prefix = ' RETRAIN_INFER=1 '
                objective_function(config, output_dir=iter_dir, args=args, cmd_prefix=cmd_prefix, return_obs=False)
                time.sleep(60)

    except Exception:
        logger.exception('Exception in HPO: %s' % task_id)
    else:
        logger.info('HPO task completed: %s' % task_id)
