#!/usr/bin/env python

import smurff

smurff.read_ini("macau.ini")


def read_ini(fname):
    from configparser import ConfigParser
    cfg = configparser.ConfigParser()
    cfg.read(fname)

