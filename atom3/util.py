import logging
import os
import socket
import subprocess
import sys


def setup_logger(log_file):
    root = logging.getLogger()
    map(root.removeHandler, root.handlers[:])
    if len(log_file) == 0:
        print("WFKWJFW")
        logging.basicConfig(stream=sys.stdout,
                            format='%(asctime)s %(levelname)s %(process)d: ' +
                            '%(message)s',
                            level=logging.INFO)
    else:
        print("SJFLJSF")
        log_dir = os.path.dirname(log_file)
        if len(log_dir) != 0 and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(filename=log_file,
                            format='%(asctime)s %(levelname)s %(process)d: ' +
                            '%(message)s',
                            level=logging.INFO)
    logging.info('=================== CALL ===================')
    logging.info('Host is {:}'.format(socket.gethostname()))
    logging.info('Git hash is {:}'.format(
        subprocess.check_output(['git', 'rev-parse', 'HEAD'])).rstrip())
    logging.info('{}'.format(' '.join(sys.argv)))


def end_logger():
    logging.info('================= END CALL =================')
