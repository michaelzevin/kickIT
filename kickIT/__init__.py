# -*- coding: utf-8 -*-
# Copyright (C) Michael Zevin (2018)
#
# This file is part of the progenitor python package.
#
# progenitor is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# progenitor is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with progenitor.  If not, see <http://www.gnu.org/licenses/>.

"""The progenitor algorithm
"""

from ._version import get_versions

__version__ = get_versions()['version']
__author__ = ['Michael Zevin <michael.zevin@ligo.org>', 'Chase Kimball <charles.kimball@ligo.org']
__credits__ = ['Scott Coughlin <scott.coughlin@ligo.org>', 'Duncan Macleod <duncan.macleod@ligo.org>']

del get_versions
