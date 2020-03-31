#!/usr/bin/env python

import matrix_io as mio
import smurff

def read_list(cfg, prefix):
    return [ cfg[d] for d in cfg.keys() if d.startswith(prefix) ]

def read_data(cfg, section):
    pos = cfg.get(section, "pos", fallback = None)
    if pos is not None:
        pos = map(int, pos.split(","))

    data = mio.read_matrix(cfg.get(section, "file"))
    matrix_type = cfg.get(section, "type", fallback = None)

    noise_model = cfg.get(section, "noise_model", fallback=None)
    if noise_model is not None:
        precision = cfg.getfloat(section, "precision")
        sn_init   = cfg.getfloat(section, "sn_init")
        sn_max    = cfg.getfloat(section, "sn_max")
        threshold = cfg.getfloat(section, "noise_threshold")
        noise = smurff.wrapper.NoiseConfig(noise_model, precision, sn_init, sn_max, threshold)
    else:
        noise = None

    direct = cfg.getboolean(section, "direct", fallback=None)
    tol = cfg.getfloat(section, "tol", fallback=None)

    return data, matrix_type, noise, pos, direct, tol

def read_ini(fname):
    from configparser import ConfigParser
    cfg = ConfigParser()
    cfg.read(fname)

    priors = read_list(cfg["global"], "prior_")
    seed = cfg.getint("global", "random_seed") if cfg.getboolean("global", "random_seed_set") else None
    threshold = cfg.getfloat("global", "threshold") if cfg.getboolean("global", "classify") else None 

    session = smurff.TrainSession(
        priors,
        cfg.getint("global", "num_latent"),
        cfg.getint("global", "num_threads", fallback=None),
        cfg.getint("global", "burnin"),
        cfg.getint("global", "nsamples"),
        seed, 
        threshold,
        cfg.getint("global", "verbose"),
        cfg.get   ("global", "save_name", fallback=smurff.temp_savename()),
        cfg.getint("global", "save_freq", fallback=None),
        cfg.getint("global", "checkpoint_freq", fallback=None),
    )

    data, matrix_type, noise, *_  = read_data(cfg, "train")
    session.setTrain(data, noise, matrix_type == "scarce")

    data, *_ = read_data(cfg, "test")
    session.setTest(data)

    for mode in range(len(priors)):
        section = "side_info_%d" % mode
        if section in cfg.keys():
            data, matrix_type, noise, pos, direct, tol = read_data(cfg, section) 
            session.addSideInfo(mode, data, noise, direct)

    session.init()

session = read_ini("macau.ini")