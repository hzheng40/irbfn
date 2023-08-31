# MIT License

# Copyright (c) 2023 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author: Hongrui Zheng
# Last Modified: 04/10/2023

import pyclothoids
import numpy as np
import argparse
import joblib
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--minx', type=float, default=5.0)
parser.add_argument('--maxx', type=float, default=30.0)
parser.add_argument('--dx', type=float, default=0.1)
parser.add_argument('--miny', type=float, default=-8.0)
parser.add_argument('--maxy', type=float, default=8.0)
parser.add_argument('--dy', type=float, default=0.1)
parser.add_argument('--mint', type=float, default=-1.57)
parser.add_argument('--maxt', type=float, default=1.57)
parser.add_argument('--dt', type=float, default=0.02)
args = parser.parse_args()

xlut = np.arange(args.minx, args.maxx + args.dx, args.dx)
ylut = np.arange(args.miny, args.maxy + args.dy, args.dy)
tlut = np.arange(args.mint, args.maxt + args.dt, args.dt)

xlutm, ylutm, tlutm = np.meshgrid(xlut, ylut, tlut, indexing="ij")
idxlut = np.stack((xlutm, ylutm, tlutm), axis=-1)

# look up columns are: x, y, theta
flattened_idxlut = idxlut.reshape((-1, 3))
# stored data colums are: k0, kd, sf
# lut = np.empty((len(xlut), len(ylut), len(tlut), 3))

# generate with parfor
def gentraj(goal):
    clothoid = pyclothoids.Clothoid.G1Hermite(0, 0, 0, goal[0], goal[1], goal[2])
    k0 = clothoid.Parameters[3]
    dk = clothoid.Parameters[4]
    s = clothoid.Parameters[5]
    k1 = k0 + (1/3)*s*dk
    k2 = k0 + (2/3)*s*dk
    k3 = k0 + (3/3)*s*dk
    return [k0, k1, k2, k3, s]

lut = joblib.Parallel(n_jobs=100)(joblib.delayed(gentraj)(goal_i) for goal_i in tqdm(flattened_idxlut))

lut = np.array(lut).reshape((len(xlut), len(ylut), len(tlut), 5))

np.savez('lut_allkappa.npz', lut=lut, xlut=xlut, ylut=ylut, tlut=tlut)