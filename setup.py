from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.spawn import find_executable
from distutils import sysconfig, log
import setuptools
import setuptools.command.build_ext

from contextlib import contextmanager
import os
import shlex
import subprocess
import sys
import struct
import multiprocessing

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
CMAKE_BUILD_DIR = os.path.join(TOP_DIR, 'build')

WINDOWS = (os.name == 'nt')

CMAKE = find_executable('cmake3') or find_executable('cmake')
MAKE = find_executable('make')

install_requires = []
setup_requires = []
tests_require = []
extras_require = {}

################################################################################
# Global variables for controlling the build variant
################################################################################

ONNX_NAMESPACE = os.getenv('ONNX_NAMESPACE', 'onnx_xla')

DEBUG = bool(os.getenv('DEBUG'))
COVERAGE = bool(os.getenv('COVERAGE'))

################################################################################
# Pre Check
################################################################################

assert CMAKE, 'Could not find "cmake" executable!'

################################################################################
# Format Script
################################################################################
subprocess.call(os.path.join(TOP_DIR, 'scripts/format.sh'))


################################################################################
# Utilities
################################################################################


@contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError('Can only cd to absolute path, got: {}'.format(path))
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)

################################################################################
# Commands
################################################################################

class cmake_build(setuptools.Command):
    """
    Compiles everything when `python setupmnm.py build` is run using cmake.
    Custom args can be passed to cmake by specifying the `CMAKE_ARGS`
    environment variable.
    The number of CPUs used by `make` can be specified by passing `-j<ncpus>`
    to `setup.py build`.  By default all CPUs are used.
    """
    user_options = [
        (str('jobs='), str('j'), str('Specifies the number of jobs to use with make'))
    ]

    built = False

    def initialize_options(self):
        self.jobs = multiprocessing.cpu_count()

    def finalize_options(self):
        self.jobs = int(self.jobs)

    def run(self):
        if cmake_build.built:
            return
        cmake_build.built = True
        if not os.path.exists(CMAKE_BUILD_DIR):
            os.makedirs(CMAKE_BUILD_DIR)

        with cd(CMAKE_BUILD_DIR):
            # configure
            cmake_args = [
                CMAKE,
                '-DPYTHON_INCLUDE_DIR={}'.format(sysconfig.get_python_inc()),
                '-DPYTHON_EXECUTABLE={}'.format(sys.executable),
                '-DONNX_NAMESPACE={}'.format(ONNX_NAMESPACE),
                '-DPY_EXT_SUFFIX={}'.format(sysconfig.get_config_var('EXT_SUFFIX') or ''),
            ]
            if COVERAGE:
                cmake_args.append('-DONNX_COVERAGE=ON')
            if COVERAGE or DEBUG:
                # in order to get accurate coverage information, the
                # build needs to turn off optimizations
                cmake_args.append('-DCMAKE_BUILD_TYPE=Debug')
            if WINDOWS:
                cmake_args.extend([
                    # we need to link with libpython on windows, so
                    # passing python version to window in order to
                    # find python in cmake
                    '-DPY_VERSION={}'.format('{0}.{1}'.format(*sys.version_info[:2])),
                    '-DONNX_XLA_USE_MSVC_STATIC_RUNTIME=ON',
                ])
                if 8 * struct.calcsize("P") == 64:
                    # Temp fix for CI
                    # TODO: need a better way to determine generator
                    cmake_args.append('-DCMAKE_GENERATOR_PLATFORM=x64')
            if 'CMAKE_ARGS' in os.environ:
                extra_cmake_args = shlex.split(os.environ['CMAKE_ARGS'])
                # prevent crossfire with downstream scripts
                del os.environ['CMAKE_ARGS']
                log.info('Extra cmake args: {}'.format(extra_cmake_args))
                cmake_args.extend(extra_cmake_args)
            cmake_args.append(TOP_DIR)
            subprocess.check_call(cmake_args)

            build_args = [CMAKE, '--build', os.curdir]
            if WINDOWS:
                build_args.extend(['--', '/maxcpucount:{}'.format(self.jobs)])
            else:
                build_args.extend(['--', '-j', str(self.jobs)])
            subprocess.check_call(build_args)

class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        self.run_command('cmake_build')
        setuptools.command.build_ext.build_ext.run(self)

    def build_extensions(self):
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = os.path.basename(self.get_ext_filename(fullname))

            lib_path = CMAKE_BUILD_DIR
            src = os.path.join(lib_path, filename)
            dst = os.path.join(os.path.realpath(self.build_lib), filename)
            self.copy_file(src, dst)


cmdclass = {
    'cmake_build': cmake_build,
    'build_ext': build_ext,
}

################################################################################
# Packages and Modules
################################################################################

package_dir = {str('') : str('python_onnxifi')}
py_modules = [str('onnxifi_backend')]

install_requires.extend([
    'protobuf',
    'numpy',
    'six',
    'typing>=3.6.4',
    'typing-extensions>=3.6.2.1',
    'onnx',
])

################################################################################
# Extensions
################################################################################

ext_modules = [
    setuptools.Extension(
        name=str('python_onnxifi'),
        sources=[]),
]

################################################################################
# Test
################################################################################

################################################################################
# Final
################################################################################

setuptools.setup(
    name="onnx-xla",
    version='1.0',
    description="ONNX and XLA Integration through ONNXIFI",
    ext_modules=ext_modules,
    py_modules=py_modules,
    package_dir=package_dir,
    cmdclass=cmdclass,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    author='varunjain99',
    author_email='varunjain99@gmail.com',
    url='https://github.com/varunjain99/onnx-xla')

