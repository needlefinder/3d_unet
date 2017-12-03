#!/bin/bash
thr=6

parallel --bar -j $thr --header : python split.py -k {i} ::: i `seq 0 1 100`