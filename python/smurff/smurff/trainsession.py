from .helper import PyNoiseConfig, PyStatusItem, SparseTensor
from .wrapper import Config, PythonSession

class TrainSession:
    """Class for doing a training run in smurff

    A simple use case could be:

    >>> trainSession = smurff.TrainSession(burnin = 5, nsamples = 5)
    >>> trainSession.addTrainAndTest(Ydense)
    >>> trainSession.run()

        
    Attributes
    ----------

    priors: list, where element is one of { "normal", "normalone", "macau", "macauone", "spikeandslab" }
        The type of prior to use for each dimension

    num_latent: int
        Number of latent dimensions in the model

    burnin: int
        Number of burnin samples to discard
    
    nsamples: int
        Number of samples to keep

    num_threads: int
        Number of OpenMP threads to use for model building

    verbose: {0, 1, 2}
        Verbosity level for C++ library

    seed: float
        Random seed to use for sampling

    save_prefix: path
        Path where to store the samples. The path includes the directory name, as well
        as the initial part of the file names.

    save_freq: int
        - N>0: save every Nth sample
        - N==0: never save a sample
        - N==-1: save only the last sample

    save_extension: { ".csv", ".ddm" }
        - .csv: save in textual csv file format
        - .ddm: save in binary file format

    checkpoint_freq: int
        Save the state of the trainSession every N seconds.

    csv_status: filepath
        Stores limited set of parameters, indicative for training progress in this file. See :class:`StatusItem`

    """
    #
    # construction functions
    #
    def __init__(self,
        priors           = None,
        num_latent       = None,
        num_threads      = None,
        burnin           = None,
        nsamples         = None,
        seed             = None,
        threshold        = None,
        verbose          = None,
        save_prefix      = None,
        save_extension   = None,
        save_freq        = None,
        checkpoint_freq  = None,
        ):

        self.config = Config()

        if priors is not None:         self.config.setPriorTypes(priors)
        if num_latent is not None:     self.config.setNumLatent(num_latent)
        if num_threads is not None:    self.config.setNumThreads(num_threads)
        if burnin is not None:         self.config.setBurnin(burnin)
        if nsamples is not None:       self.config.setNSamples(nsamples)
        if seed is not None:           self.config.setRandomSeed(seed)
        if threshold is not None:      self.config.setThreshold(threshold)
        if verbose is not None:        self.config.setVerbose(verbose)
        if save_prefix is not None:    self.config.setSavePrefix(save_prefix.encode('UTF-8'))
        if save_extension is not None: self.config.setSaveExtension(save_extension.encode('UTF-8'))
        if save_freq is not None:      self.config.setSaveFreq(save_freq)
        if checkpoint_freq is not None:self.config.setCheckpointFreq(checkpoint_freq)

    def addTrainAndTest(self, Y, Ytest = None, noise = PyNoiseConfig(), is_scarce = True):
        """Adds a train and optionally a test matrix as input data to this TrainSession

        Parameters
        ----------

        Y : :class: `numpy.ndarray`, :mod:`scipy.sparse` matrix or :class: `SparseTensor`
            Train matrix/tensor 
       
        Ytest : :mod:`scipy.sparse` matrix or :class: `SparseTensor`
            Test matrix/tensor. Mainly used for calculating RMSE.

        noise : :class: `PyNoiseConfig`
            Noise model to use for `Y`

        is_scarce : bool
            When `Y` is sparse, and `is_scarce` is *True* the missing values are considered as *unknown*.
            When `Y` is sparse, and `is_scarce` is *False* the missing values are considered as *zero*.
            When `Y` is dense, this parameter is ignored.

        """
        self.noise_config = noise.toNoiseConfig()
        train, test = prepare_train_and_test(Y, Ytest, self.noise_config, is_scarce)
        self.config.setTrain(train)

        if Ytest is not None:
            self.config.setTest(test)

    def addSideInfo(self, mode, Y, noise = PyNoiseConfig(), tol = 1e-6, direct = False):
        """Adds fully known side info, for use in with the macau or macauone prior

        mode : int
            dimension to add side info (rows = 0, cols = 1)

        Y : :class: `numpy.ndarray`, :mod:`scipy.sparse` matrix
            Side info matrix/tensor 
            Y should have as many rows in Y as you have elemnts in the dimension selected using `mode`.
            Columns in Y are features for each element.

        noise : :class: `PyNoiseConfig`
            Noise model to use for `Y`
        
        direct : boolean
            - When True, uses a direct inversion method. 
            - When False, uses a CG solver 

            The direct method is only feasible for a small (< 100K) number of features.

        tol : float
            Tolerance for the CG solver.

        """
        self.noise_config = prepare_noise_config(noise)
        self.config.addSideInfoConfig(mode, prepare_sideinfo(Y, self.noise_config, tol, direct))

    def addPropagatedPosterior(self, mode, mu, Lambda):
        """Adds mu and Lambda from propagated posterior

        mode : int
            dimension to add side info (rows = 0, cols = 1)

        mu : :class: `numpy.ndarray` matrix
            mean matrix  
            mu should have as many rows as `num_latent`
            mu should have as many columns as size of dimension `mode` in `train`

        Lambda : :class: `numpy.ndarray` matrix
            co-variance matrix  
            Lambda should be shaped like K x K x N 
            Where K == `num_latent` and N == dimension `mode` in `train`
        """
        self.noise_config = prepare_noise_config(PyNoiseConfig())
        if len(Lambda.shape) == 3:
            assert Lambda.shape[0] == self.num_latent
            assert Lambda.shape[1] == self.num_latent
            Lambda = Lambda.reshape(self.num_latent * self.num_latent, Lambda.shape[2], order='F')

        self.config.addPropagatedPosterior(
            mode,
            shared_ptr[MatrixConfig](prepare_dense_matrix(mu, self.noise_config)),
            shared_ptr[MatrixConfig](prepare_dense_matrix(Lambda, self.noise_config))
        )


    def addData(self, pos, Y, is_scarce = False, noise = PyNoiseConfig()):
        """Stacks more matrices/tensors next to the main train matrix.

        pos : shape
            Block position of the data with respect to train. The train matrix/tensor
            has implicit block position (0, 0). 

        Y : :class: `numpy.ndarray`, :mod:`scipy.sparse` matrix or :class: `SparseTensor`
            Data matrix/tensor to add

        is_scarce : bool
            When `Y` is sparse, and `is_scarce` is *True* the missing values are considered as *unknown*.
            When `Y` is sparse, and `is_scarce` is *False* the missing values are considered as *zero*.
            When `Y` is dense, this parameter is ignored.

        noise : :class: `PyNoiseConfig`
            Noise model to use for `Y`
        
        """
        self.noise_config = prepare_noise_config(noise)
        self.config.addAuxData(prepare_auxdata(Y, pos, is_scarce, self.noise_config))

    # 
    # running functions
    #

    def init(self):
        """Initializes the `TrainSession` after all data has been added.

        You need to call this method befor calling :meth:`step`, unless you call :meth:`run`

        Returns
        -------
        :class:`StatusItem` of the trainSession.

        """

        self.pySession = PythonSession()
        self.pySession.fromConfig(self.config)
        self.pySession.init()
        logging.info(self)
        return self.getStatus()

    def __dealloc__(self):
        if (self.ptr.get()):
            self.ptr.reset()

    def step(self):
        """Does on sampling or burnin iteration.

        Returns
        -------
        - When a step was executed: :class:`StatusItem` of the trainSession.
        - After the last iteration, when no step was executed: `None`.

        """
        not_done = self.ptr_get().step()
        
        if self.ptr_get().interrupted():
            raise KeyboardInterrupt

        if not_done:
            return self.getStatus()
        else:
            return None

    def run(self):
        """Equivalent to:

        .. code-block:: python
        
            self.init()
            while self.step():
                pass
        """
        self.init()
        while self.step():
            pass

        return self.getTestPredictions()

    #
    # get state
    #

    def __str__(self):
        try:
            return self.ptr_get().infoAsString().decode('UTF-8')
        except ValueError:
            return "Uninitialized SMURFF Train TrainSession (call .init())"


    def getStatus(self):
        """ Returns :class:`StatusItem` with current state of the trainSession

        """
        if self.ptr_get().getStatus():
            self.status_item = self.ptr_get().getStatus()
            status =  PyStatusItem(
                self.status_item.get().phase,
                self.status_item.get().iter,
                self.status_item.get().phase_iter,
                self.status_item.get().model_norms,
                self.status_item.get().rmse_avg,
                self.status_item.get().rmse_1sample,
                self.status_item.get().train_rmse,
                self.status_item.get().auc_1sample,
                self.status_item.get().auc_avg,
                self.status_item.get().elapsed_iter,
                self.status_item.get().nnz_per_sec,
                self.status_item.get().samples_per_sec)

            logging.info(status)
            
            return status
        else:
            return None

    def getConfig(self):
        """Get this `TrainSession`'s configuration in ini-file format

        """
        config_filename = tempfile.mkstemp()[1]
        self.config.save(config_filename.encode('UTF-8'))
        
        with open(config_filename, 'r') as f:
            ini_string = "".join(f.readlines())

        os.unlink(config_filename)

        return ini_string

    def makePredictSession(self):
        """Makes a :class:`PredictSession` based on the model
           that as built in this `TrainSession`.

        """
        rf = self.ptr_get().getOutputFile().get().getFullPath().decode('UTF-8')
        return PredictSession(rf)

    def getTestPredictions(self):
        """Get predictions for test matrix.

        Returns
        -------
        list 
            list of :class:`Prediction`

        """
        py_items = []

        if self.ptr_get().getResultItems().size():
            cpp_items = self.ptr_get().getResultItems()
            it = cpp_items.begin()
            while it != cpp_items.end():
                py_items.append(prepare_result_item(deref(it)))
                inc(it)

        return py_items
    
    def getRmseAvg(self): 
        """Average RMSE across all samples for the test matrix

        """
        return self.ptr_get().getRmseAvg()
