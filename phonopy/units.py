# Copyright (C) 2011 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import time
from math import pi, sqrt
from functools import wraps

kb_J = 1.3806504e-23 # [J/K]
PlanckConstant = 4.13566733e-15 # [eV s]
Hbar = PlanckConstant/(2*pi) # [eV s]
Avogadro = 6.02214179e23
SpeedOfLight = 299792458 # [m/s]
AMU = 1.6605402e-27 # [kg]
Newton = 1.0        # [kg m / s^2]
Joule = 1.0         # [kg m^2 / s^2]
EV = 1.60217733e-19 # [J]
Angstrom = 1.0e-10  # [m]
THz = 1.0e12        # [/s]
Mu0 = 4.0e-7 * pi
Epsilon0 = 1.0 / Mu0 / SpeedOfLight**2
Me = 9.10938215e-31

Bohr = 4e10 * pi * Epsilon0 * Hbar**2 / Me  # Bohr radius [A] 0.5291772
Hartree = Me * EV / 16 / pi**2 / Epsilon0**2 / Hbar**2 # Hartree [eV] 27.211398
Rydberg = Hartree / 2 # Rydberg [eV] 13.6056991

THzToEv = PlanckConstant * 1e12 # [eV]
Kb = kb_J / EV  # [eV/K] 8.6173383e-05
THzToCm = 1.0e12 / (SpeedOfLight * 100) # [cm^-1] 33.356410
CmToEv = THzToEv / THzToCm # [eV] 1.2398419e-4
VaspToEv = sqrt(EV/AMU)/Angstrom/(2*pi)*PlanckConstant # [eV] 6.46541380e-2
VaspToTHz = sqrt(EV/AMU)/Angstrom/(2*pi)/1e12 # [THz] 15.633302
VaspToCm =  VaspToTHz * THzToCm # [cm^-1] 521.47083
EvTokJmol = EV / 1000 * Avogadro # [kJ/mol] 96.4853910
Wien2kToTHz = sqrt(Rydberg/1000*EV/AMU)/(Bohr*1e-10)/(2*pi)/1e12 # [THz] 3.44595837
EVAngstromToGPa = EV * 1e21


class Timeit():
    def __init__(self):
        self.time_dict = {}
        self._is_print_each = False
        self._main = None

    def timeit(self, method):
        name = '.'.join([method.__module__, method.__name__])
        if name not in self.time_dict.keys():
            self.time_dict[name] = 0.
        @wraps(method)
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            if self._is_print_each:
                print '%r %2.2f sec' % \
                      (name, te-ts)
            self.time_dict[name] += te - ts
            return result
        return timed

    def set_main(self, method):
        name = '.'.join([method.__module__, method.__name__])
        def wrapper(*args, **kw):
            self._main = name
            return method(*args, **kw)
        return wrapper

    def output(self, is_ignore_zero=True):
        all_time = sum(self.time_dict.values())
        print "#################Time consumption of major functions#################"
        if self._main is not None and self._main in self.time_dict.keys():
                print 'Main function %r %2.2f sec' % \
                      (self._main, self.time_dict[self._main])
        keys = sorted(self.time_dict.keys(), key=lambda k: self.time_dict[k], reverse=True)
        for key in keys:
            if is_ignore_zero and all_time != 0:
                if not self.time_dict[key] / all_time > 0.01:
                  continue
            if key != self._main:
                print '%r %5.2f sec' % \
                      (key, self.time_dict[key])
        print "#################Time consumption of major functions#################"
        print

    def reset(self):
        self._main = None
        for key in self.time_dict.keys():
            self.time_dict[key] = 0.


total_time = Timeit()