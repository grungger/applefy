#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 19:45:28 2023

@author: Gian
"""
import sys
sys.path.append("/Users/Gian/Documents/GitHub/applefy/applefy/wrappers")
from JWSTpynpoint_wrap import JWSTSimpleSubtractionPynPoint

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

dirpath = "/Users/Gian/Documents/GitHub/Pynpoint_testcode/MRS_PSF_subtraction_FS23/Data_cube_input"
star  = "/Users/Gian/Documents/JWST_Central-Database/Reduced_cubes/cubes_obs3_red_red"
refstar = "/Users/Gian/Documents/JWST_Central-Database/Reduced_cubes/cubes_obs9_red_red"
wrapper = JWSTSimpleSubtractionPynPoint(dirpath)

A = np.zeros((50,40))
A[20:29,16:24] = 5
A[23:26,19:21] = 10

B = np.zeros((49,41))
B[21:25,17:25] = 5
B[23:24,19:21] = 10

residuals = wrapper(star,refstar,"wrap","exp_id")