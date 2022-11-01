import logging
import traceback
import easytorch
from easytorch import launch_training


def test(model, dataset, exception_log_file):
    CFG = __import__("examples.{0}.{0}_{1}".format(model, dataset),
                     fromlist=["examples.{0}.{0}_{1}".format(model, dataset)]).CFG
    # CFG.TRAIN.NUM_EPOCHS = 1
    # CFG.ENV.SEED = seed
    print(("*" * 60 + "{0:>10}" + "@{1:<10}" + "*" * 60).format(model, dataset))                
    try:
        launch_training(CFG, "0")
    except Exception as e:
        exception_log_file.write("\n" + "*" * 60 + "{0:>10}@{1:<22}".format(
            model, dataset + "test failed.") + "*" * 60 + "\n")
        traceback.print_exc(limit=1, file=exception_log_file)
    # safely delete all the handlers of 'easytorch-training' logger and re-add them, so as to get the correct log file path.
    logger = logging.getLogger("easytorch-training")
    for h in logger.handlers:
        h.close()
    logger.handlers = []
    if "easytorch-training" in easytorch.utils.logging.logger_initialized:
        easytorch.utils.logging.logger_initialized.remove("easytorch-training")

    print("*" * 141)
