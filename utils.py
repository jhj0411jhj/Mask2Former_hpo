import time
import subprocess
try:
    from openbox import logger
except ModuleNotFoundError:
    print('cannot import logger from openbox, use logging instead.')
    import logging
    logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    pass


def str_proc(process: subprocess.CompletedProcess, max_output_line=None):
    assert max_output_line is None or (isinstance(max_output_line, int) and max_output_line > 0)
    if not isinstance(process, subprocess.CompletedProcess):
        logger.warning('error process type: %s' % type(process))
        return str(process)
    args = ['args={!r}'.format(process.args),
            'returncode={!r}'.format(process.returncode)]
    if process.stdout is not None:
        stdout = process.stdout if max_output_line is None else '\n'.join(process.stdout.split('\n')[-max_output_line:])
        args.append('stdout=\'{!s}\''.format(stdout))
    if process.stderr is not None:
        stderr = process.stderr if max_output_line is None else '\n'.join(process.stderr.split('\n')[-max_output_line:])
        args.append('stderr=\'{!s}\''.format(stderr))
    return "{}({})".format(type(process).__name__, ', '.join(args))


def execute_command(cmd, logtitle: str, *, capture_output=False, timeout=None,
                    encoding='utf8', max_output_line=None) -> subprocess.CompletedProcess:
    if isinstance(cmd, list):
        cmd = ' '.join(cmd)
    assert isinstance(cmd, str), 'cmd should be str: %s' % str(cmd)
    assert max_output_line is None or (isinstance(max_output_line, int) and max_output_line > 0)
    logger.info('[%s] Execute command: %s' % (logtitle, cmd))
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=capture_output, timeout=timeout, encoding=encoding,
                           shell=True, executable='/bin/bash')  # type: subprocess.CompletedProcess
    except subprocess.TimeoutExpired as exc:  # todo: timeout may not work
        stdout = exc.stdout if exc.stdout else ''
        stderr = exc.stderr if exc.stderr else ''
        if isinstance(stdout, bytes):
            stdout = stdout.decode()
        if isinstance(stderr, bytes):
            stderr = stderr.decode()
        logger.exception('[%s] Command execution timeout! cost: %.2fs. Process stdout:\'%s\', stderr:\'%s\'.'
                         % (logtitle, time.time() - t, stdout, stderr))
        raise

    logger.info('[%s] Command execution finished. cost: %.2fs. returncode: %d'
                % (logtitle, time.time() - t, p.returncode))
    if p.returncode != 0:
        logger.error('[%s] Error! Process status: %s' % (logtitle, str_proc(p, max_output_line)))
        raise ExecutionError('[%s] Error! returncode: %d' % (logtitle, p.returncode))
    else:
        logger.debug('[%s] Process status: %s' % (logtitle, str_proc(p, max_output_line)))
    return p
