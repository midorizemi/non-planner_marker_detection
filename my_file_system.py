import os
from typing import Tuple


class FileSystemExpt:
    """
    実験名のディレクトリ下に入力画像名のテストケースディレクトリを作る
    テストケース下には，実験結果，ログなどを出力する
    """

    def get_test_case(self):
        # 入力画像
        pass

    def get_output(self):
        pass

    def get_expt_name(self):
        pass

    def get_log(self):
        pass

def getd_outpts(dir_name: Tuple[str, str, str]) -> str:
    """(expt_name, testcase_name) or list type
    :param dir_name: tuple[str]
    :return: str
    """
    outputs = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'outputs'))
    if dir_name is not None:
        return os.path.abspath(os.path.join(outputs, *dir_name))
    return outputs

def getd_inputs(dir_name: str) -> str:
    inputs = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data", "inputs"))
    if dir_name is not None:
        return os.path.abspath(os.path.join(inputs, dir_name))
    return inputs

def getd_templates(dir_name: Tuple[str]) -> str:
    templates = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'templates'))
    if dir_name is not None:
        return os.path.abspath(os.path.join(templates, *dir_name))
    return templates

def getf_log(expt_testcase: Tuple[str, str, str], logfn: str) -> str:
    """(expt_name, testcase_name) or list type
    :param expt_testcase: tuple[str]
    :param logfn: str
    :return: str
    """
    testcase_root = getd_outpts(expt_testcase)
    return os.path.abspath(os.path.join(testcase_root, "log", logfn))

def getf_template(template: Tuple[str]) -> str:
    return getd_templates(template)

def getf_input(testcase: str, test_sample: str) -> str:
    testcase_dir = getd_inputs(testcase)
    return os.path.abspath(os.path.join(testcase_dir, test_sample))

def getf_output(expt_testcase: Tuple[str, str, str], test_sample: str) -> str:
    """(expt_name, testcase_name) or list type
    :param expt_testcase: tuple[str]
    :param test_sample: str
    :return: str
    """
    testcase_dir = getd_outpts(expt_testcase)
    return os.path.abspath(os.path.join(testcase_dir, test_sample))

if __name__ == '__main__':
    print(getd_outpts(("aaa", "bbb", "aaa")))
    expt_names = ("expt_test", "testcase", "test1")
    # os.makedirs(os.path.join(getd_outpts(expt_names), 'log'))

