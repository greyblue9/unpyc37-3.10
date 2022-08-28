import compileall
import dis
import io
import marshal
import os
import fnmatch
import unpyc3
import importlib.util
import sys


def get_compiled_binary(fp):
    if sys.version_info >= (3, 4):
        return importlib.util.cache_from_source(fp)
    else:
        import imp
        return imp.cache_from_source(fp)


def roundtrip_test(fp: str) -> (str, str):
    compileall.compile_file(fp)
    compiled = get_compiled_binary(fp)
    with open(fp) as f:
        src = f.read() \
            .lstrip(' ') \
            .rstrip(' ').replace('\r', '').rstrip(os.linesep)
        print(src)
        result = unpyc3.decompile(compiled)
        rsrc = os.linesep.join((str(r) for r in result)).replace('\r', '').rstrip(os.linesep)
        return src, rsrc


def empty_trace(*args):
    pass


def make_roundtrip_test(fp,use_trace=True):
    def test_dec(self):
        compiled = get_compiled_binary(fp)
        try:
            src, result_src = roundtrip_test(fp)
            self.assertEqual(src, result_src)
        except Exception as ae:
            if use_trace and not unpyc3.get_trace():
                try:
                    unpyc3.set_trace(print)
                    unpyc3.decompile(compiled)
                except Exception:
                    pass
                finally:
                    unpyc3.set_trace(None)
            raise ae

    return test_dec


def make_tests_from_folder(test_dir, test_class):
    tps = []
    for root, dirs, files in os.walk(test_dir):
        for filename in fnmatch.filter(files, '*.py'):
            full_path = os.path.join(root, filename)
            if full_path == __file__:
                continue
            test = make_roundtrip_test(full_path)
            test_name = filename.replace('.py', '')
            r = list(filter(None, root.replace(test_dir, '').split(os.sep)))
            prefix = 'test_'
            if any(r):
                test_name = str.join('_', r) + '_' + test_name
            if not test_name[:len(prefix)] == prefix:
                test_name = prefix + test_name
            setattr(test_class,test_name,test)


def make_decompile_test(full_path,trace_failure=True):
    def test(self):
        root = os.path.dirname(full_path)
        filename = os.path.basename(full_path)
        try:
            result = unpyc3.decompile(full_path)
            result_src = os.linesep.join((str(r) for r in result)).replace('\r', '')
            with io.open(full_path.replace('.pyc', '.py'), 'w') as f:
                f.write(result_src)
            co = compile(result_src,'<string>','exec')
        except Exception as ae:

            with io.open(full_path, 'rb') as x:
                x.read(16)
                dis.dis(marshal.load(x))

            raise ae

    return test


def add_decompile_test_to_fixture(full_path, root, fixture,trace_failure=True):
    relative_path = full_path.replace(root, '').lstrip(os.path.sep)
    test = make_decompile_test(full_path,trace_failure)
    test_name = relative_path.replace(os.path.sep, '_').replace('.cpython-37', '').replace('.pyc', '')
    prefix = 'test_'
    if not test_name[:len(prefix)] == prefix:
        test_name = prefix + test_name
    setattr(fixture, test_name, test)
