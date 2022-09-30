import logging
import os
from datetime import date
import glob


class Log:
    def __init__(self, directory: str, level=logging.INFO, format: str = '%(levelname)s - %(asctime)s - %(message)s', datefmt: str = '%m-%d-%Y %H:%M:%S') -> None:
        self.__file_path = ''
        self.__level = level
        self.__directory = directory

        self.__format = format
        self.__datefmt = datefmt
        # Create dir if it dosent exist
        self.createDir(self.__directory)
        # create log file of the day if it dosent exist
        self.createLogFile()

    def config(self) -> None:
        logging.basicConfig(filename=self.__file_path, format=self.__format,
                            datefmt=self.__datefmt, level=self.__level)

    def removeHandler(self) -> None:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    @staticmethod
    def getDate() -> str:
        today = date.today()
        return today.strftime("%Y-%m-%d")

    def createDir(self, dir_path: str) -> bool:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        return True

    def createLogFile(self) -> None:
        today = Log.getDate()
        file_name = '.'.join([today, 'log'])
        file_path = os.path.join(self.__directory, file_name)
        if not os.path.isfile(file_path):
            file = open(file_path, 'x')
            file.close()
        self.__file_path = file_path

    def keep_last_k_days(self, num_days: int) -> None:
        path = os.path.join(self.__directory, '*.log')
        files = glob.glob(path)
        files.sort()
        # finish up
        num_files = len(files)
        if num_files > num_days:
            for file in files[:num_files - num_days]:
                os.remove(file)
                msg = ' '.join(['Purging file:', file])
                logging.info(msg)

    def message(self, message_type: str, message: str) -> None:
        output_message = list()
        if isinstance(message, list):
            for string in message:
                if not isinstance(string, str):
                    output_message.append(str(string))
                else:
                    output_message.append(string)
            message = ' '.join(output_message)

        elif not isinstance(message, str):
            message = str(message)
        message_type = message_type.upper().strip()
        if message_type not in ('INFO', 'WARNING', 'ERROR', 'DEBUG'):
            message_type = 'INFO'
        if message_type == 'INFO':
            logging.info(message)
        elif message_type == 'WARNING':
            logging.warning(message)
        elif message_type == 'ERROR':
            logging.error(message)
        elif message_type == 'DEBUG':
            logging.debug(message)
