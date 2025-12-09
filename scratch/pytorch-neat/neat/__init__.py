import logging
import sys


LOG_FORMAT = "%(asctime)s [%(threadName)-12.12s] [%(module)-10.10s] [%(levelname)-5.5s]  %(message)s"
logging.basicConfig(
    filename='neat/experiments/gameof2048/2048.log',
    format=LOG_FORMAT, 
    level=logging.INFO
)
# logging.basicConfig(
#     filename='gameof2048.log',
#     format=LOG_FORMAT, 
#     level=logging.DEBUG
# )