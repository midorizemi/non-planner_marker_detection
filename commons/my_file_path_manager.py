import os
import enum
from typing import Tuple
import inspect
import logging
logger = logging.getLogger(__name__)


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

class DirNames(enum.Enum):
    TEMPLATES = 'templates'
    INPUTS = 'inputs'
    OUTPUTS = 'outputs'

def get_dir_full_path_(dirname='templates'):
    return os.path.abspath(os.path.join(os.path.dirname(__file__),  os.pardir, os.pardir, 'data', dirname))

def get_template_file_full_path_(fn):
    return os.path.join(get_dir_full_path_(DirNames.TEMPLATES.value), fn)

def getd_outpts(dir_name: Tuple[str, str, str]) -> str:
    """(expt_name, testcase_name) or list type
    :param dir_name: tuple[str]
    :return: str
    """
    outputs = os.path.abspath(os.path.join(os.path.dirname(__file__),  os.pardir, os.pardir, 'data', 'outputs'))
    if dir_name is not None:
        return os.path.abspath(os.path.join(outputs, *dir_name))
    return outputs

def get_inputs_dir_full_path(*dir_name) -> str:
    return os.path.join(get_dir_full_path_(DirNames.INPUTS.value), *dir_name)

def getd_inputs(dir_name: str) -> str:
    return os.path.join(get_dir_full_path_(DirNames.INPUTS.value), dir_name)

def getd_templates(dir_name: Tuple[str]) -> str:
    templates = os.path.abspath(os.path.join(os.path.dirname(__file__),  os.pardir, os.pardir, 'data', 'templates'))
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

def get_dump_dir(template_fn):
    return os.path.join(get_dir_full_path_(DirNames.TEMPLATES.value), 'dump_features', template_fn)
def get_pikle_path(template_fn):
    return os.path.join(get_dump_dir(template_fn), template_fn + '.pikle')

def getf_output(expt_testcase, test_sample: str) -> str:
    """(expt_name, testcase_name) or list type
    :param expt_testcase: tuple[str]
    :param test_sample: str
    :return: str
    """
    testcase_dir = getd_outpts(expt_testcase)
    return os.path.abspath(os.path.join(testcase_dir, test_sample))

def setup_expt_directory(base_name):
    logger.info('Now in {}'.format(inspect.currentframe().f_code.co_name))
    outputs_dir = get_dir_full_path_(DirNames.OUTPUTS.value)
    expt_name, ext = os.path.splitext(base_name)
    expt_path = os.path.join(outputs_dir, expt_name)
    if os.path.exists(expt_path):
        return expt_path
    os.mkdir(expt_path)
    return expt_path

def setup_output_directory(base_name, *dirs):
    logger.info('Now in {}'.format(inspect.currentframe().f_code.co_name))
    outputs_dir = get_dir_full_path_(DirNames.OUTPUTS.value)
    expt_name, ext = os.path.splitext(base_name)
    path = os.path.join(outputs_dir, expt_name, *dirs)
    if os.path.exists(path):
        return path
    os.makedirs(path, exist_ok=True)
    logger.info('make dir in ' + path)
    return path

def get_dir_full_path_testset(*option_dirs, prefix_shape, template_fn):
    name, ext = os.path.splitext(template_fn)
    testset_name = prefix_shape + name
    return get_inputs_dir_full_path(*option_dirs, testset_name)

def make_list_testcase_file_name(*option_dirs, prefix_shape, template_fn):
    testset_full_path = get_dir_full_path_testset(*option_dirs, prefix_shape=prefix_shape, template_fn=template_fn)
    b = os.listdir(testset_full_path).sort()
    return b

def make_list_template_filename():
    a = os.listdir(get_dir_full_path_(DirNames.TEMPLATES.value))
    newlist = [template_fn for template_fn in a if not template_fn.startswith('mesh')]
    return newlist

if __name__ == '__main__':
    print(getd_outpts(("aaa", "bbb", "aaa")))
    expt_names = ("expt_test", "testcase", "test1")
    # os.makedirs(os.path.join(getd_outpts(expt_names), 'log'))

