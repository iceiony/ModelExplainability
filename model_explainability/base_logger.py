import os

LOG_PATH = './logs/file.log'


if not os.path.exists(os.path.dirname(LOG_PATH)):
    os.makedirs(os.path.dirname(LOG_PATH))

if os.path.exists(LOG_PATH):
    try:
        folder_path = os.path.dirname(LOG_PATH)
        file_name   = os.path.basename(LOG_PATH)

        suffix = len([f for f in os.listdir(folder_path) if file_name in f])
        os.rename(LOG_PATH, os.path.join(folder_path, f'{file_name}.{suffix}'))
    except:
        pass

import sys
import logging
from logging import FileHandler

log = logging

handlers = [FileHandler(LOG_PATH)]

if 'pytest' not in sys.modules.keys():
    handlers.append(logging.StreamHandler())

logging.basicConfig(
    handlers = handlers,
    format   = '%(asctime)s.%(msecs)03d %(levelname)s[%(name)s] %(message)s',
    datefmt  = '%Y-%m-%d %H:%M:%S',
    level    = logging.INFO)
