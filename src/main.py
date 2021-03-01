import logging
import sys

from score_eval import pred_eval

LOG_FILENAME = "logfile.log"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
logging.info("Running the Main Function.....")
pred_eval()
