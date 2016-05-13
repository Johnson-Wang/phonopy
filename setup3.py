from distutils.core import setup, Extension
import numpy
include_dirs_numpy = [numpy.get_include()]

extension = Extension(
    'anharmonic._phono3py',
    include_dirs=(['c/harmonic_h',
                   'c/anharmonic_h',
                   'c/spglib_h'] +
                  include_dirs_numpy),
    extra_compile_args=['-fopenmp', '-Wno-unknown-pragmas'],
    extra_link_args=['-lgomp',
                     '-lopenblas'],
    sources=['c/_phono3py.c',
             'c/harmonic/dynmat.c',
             'c/harmonic/lapack_wrapper.c',
             'c/anharmonic/phonoc_array.c',
             'c/anharmonic/phonoc_math.c',
             'c/anharmonic/phonoc_utils.c',
             'c/anharmonic/phonon3/fc3.c',
             'c/anharmonic/phonon3/interaction.c',
             'c/anharmonic/phonon3/real_to_reciprocal.c',
             'c/anharmonic/phonon3/reciprocal_to_normal.c',
             'c/anharmonic/phonon3/imag_self_energy.c',
             'c/anharmonic/phonon3/kappa.c',
             'c/anharmonic/other/isotope.c',
             'c/anharmonic/phonon4/real_to_reciprocal.c',
             'c/anharmonic/phonon4/frequency_shift.c',
             'c/spglib/mathfunc.c',
             'c/spglib/tetrahedron_method.c'])

extension_phono4py = Extension(
    'anharmonic._phono4py',
    include_dirs=(['c/harmonic_h',
                   'c/anharmonic_h'] +
                  include_dirs_numpy),
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-lgomp',
                     '-lopenblas'],
    sources=['c/_phono4py.c',
             'c/harmonic/dynmat.c',
             'c/harmonic/lapack_wrapper.c',
             'c/anharmonic/phonoc_array.c',
             'c/anharmonic/phonoc_math.c',
             'c/anharmonic/phonoc_utils.c',
             'c/anharmonic/phonon3/fc3.c',
             'c/anharmonic/phonon4/fc4.c',
             'c/anharmonic/phonon4/real_to_reciprocal.c',
             'c/anharmonic/phonon4/frequency_shift.c'])

extension_forcefit = Extension(
    'anharmonic._forcefit',
    include_dirs=(['c/harmonic_h',
                   'c/anharmonic_h'] +
                  include_dirs_numpy),
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-lgomp',
                     '-lopenblas'],
    sources=['c/_forcefit.c',
             'c/harmonic/lapack_wrapper.c'])

setup(name='phono3py',
      version='1.1.7',
      description='This is the phono3py module.',
      author='Atsushi Togo, Xinjiang Wang',
      author_email='atz.togo@gmail.com, swanxinjiang@gmail.com',
      url='http://phonopy.sourceforge.net/',
      packages=['anharmonic',
                'anharmonic.force_fit',
                'anharmonic.phonon3',
                'anharmonic.other',
                'anharmonic.phonon4'],
      scripts=['scripts/force-fit',
               'scripts/phono3py',
               'scripts/phono4py'],
      ext_modules=[extension,
                   extension_phono4py,
                   extension_forcefit])
