from typing import Callable, Optional, Tuple
from tianshou.utils.logger.base import LOG_DATA_TYPE, BaseLogger
from utils.logging.logger import Logger

class TianshouLoggerAdapter(BaseLogger):
    def __init__(self, logger: Logger, train_interval: int = 1000, test_interval: int = 1, update_interval: int = 1000) -> None:
        super().__init__(train_interval, test_interval, update_interval)
        self.logger = logger
    
    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        """Specify how the writer is used to log data.

        :param str step_type: namespace which the data dict belongs to.
        :param int step: stands for the ordinate of the data dict.
        :param dict data: the data to write with format ``{key: value}``.
        """
        data[step_type] = step
        self.logger.log_metrics(data, step=step)

    
    def save_data(self, epoch: int, env_step: int, gradient_step: int, save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None) -> None:
        return 
    
    def restore_data(self) -> Tuple[int, int, int]:
        return