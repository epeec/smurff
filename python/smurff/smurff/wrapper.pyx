##
#
# Cython file that bridges between the smurff python code and the C++ code
# 
# This file will be compiled via CMake
#


from libc.stdint cimport *
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector
from libcpp.string cimport string

from Config cimport *
from ISession cimport ISession
from NoiseConfig cimport *
from MatrixConfig cimport MatrixConfig
from TensorConfig cimport TensorConfig
from SessionFactory cimport SessionFactory
from PVec cimport PVec

cimport numpy as np
import  numpy as np
import  scipy as sp
import pandas as pd
import scipy.sparse
import numbers

from .prepare import make_train_test, make_train_test_df
from .result import Result, ResultItem

DENSE_MATRIX_TYPES  = [np.ndarray, np.matrix]
SPARSE_MATRIX_TYPES = [sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]

DENSE_TENSOR_TYPES  = [np.ndarray]
SPARSE_TENSOR_TYPES = [pd.DataFrame]

def remove_nan(Y):
    if not np.any(np.isnan(Y.data)):
        return Y
    idx = np.where(np.isnan(Y.data) == False)[0]
    return sp.sparse.coo_matrix( (Y.data[idx], (Y.row[idx], Y.col[idx])), shape = Y.shape )

cdef MatrixConfig* prepare_sparse_matrix(X, NoiseConfig noise_config, is_scarse):
    if type(X) not in [sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        raise ValueError("Matrix must be either coo, csr or csc (from scipy.sparse)")

    X = X.tocoo(copy = False)
    X = remove_nan(X)

    cdef np.ndarray[uint32_t] irows = X.row.astype(np.uint32, copy=False)
    cdef np.ndarray[uint32_t] icols = X.col.astype(np.uint32, copy=False)
    cdef np.ndarray[double] vals = X.data.astype(np.double, copy=False)

    # Get begin and end pointers from numpy arrays
    cdef uint32_t* irows_begin = &irows[0]
    cdef uint32_t* irows_end = irows_begin + irows.shape[0]
    cdef uint32_t* icols_begin = &icols[0]
    cdef uint32_t* icols_end = icols_begin + icols.shape[0]
    cdef double* vals_begin = &vals[0]
    cdef double* vals_end = vals_begin + vals.shape[0]

    # Create vectors from pointers
    cdef vector[uint32_t]* rows_vector_ptr = new vector[uint32_t]()
    rows_vector_ptr.assign(irows_begin, irows_end)
    cdef vector[uint32_t]* cols_vector_ptr = new vector[uint32_t]()
    cols_vector_ptr.assign(icols_begin, icols_end)
    cdef vector[double]* vals_vector_ptr = new vector[double]()
    vals_vector_ptr.assign(vals_begin, vals_end)

    cdef shared_ptr[vector[uint32_t]] rows_vector_shared_ptr = shared_ptr[vector[uint32_t]](rows_vector_ptr)
    cdef shared_ptr[vector[uint32_t]] cols_vector_shared_ptr = shared_ptr[vector[uint32_t]](cols_vector_ptr)
    cdef shared_ptr[vector[double]] vals_vector_shared_ptr = shared_ptr[vector[double]](vals_vector_ptr)

    cdef MatrixConfig* matrix_config_ptr = new MatrixConfig(<uint64_t>(X.shape[0]), <uint64_t>(X.shape[1]), rows_vector_shared_ptr, cols_vector_shared_ptr, vals_vector_shared_ptr, noise_config, is_scarse)
    return matrix_config_ptr

cdef MatrixConfig* prepare_dense_matrix(X, NoiseConfig noise_config):
    cdef np.ndarray[np.double_t] vals = np.squeeze(np.asarray(X.flatten(order='F')))
    cdef double* vals_begin = &vals[0]
    cdef double* vals_end = vals_begin + vals.shape[0]
    cdef vector[double]* vals_vector_ptr = new vector[double]()
    vals_vector_ptr.assign(vals_begin, vals_end)
    cdef shared_ptr[vector[double]] vals_vector_shared_ptr = shared_ptr[vector[double]](vals_vector_ptr)
    cdef MatrixConfig* matrix_config_ptr = new MatrixConfig(<uint64_t>(X.shape[0]), <uint64_t>(X.shape[1]), vals_vector_shared_ptr, noise_config)
    return matrix_config_ptr

cdef MatrixConfig* prepare_sideinfo(in_matrix, NoiseConfig noise_config):
    if type(in_matrix) in [sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        return prepare_sparse_matrix(in_matrix, noise_config, False)
    else:
        return prepare_dense_matrix(in_matrix, noise_config)

cdef TensorConfig* prepare_dense_tensor(tensor, NoiseConfig noise_config):
    if type(tensor) not in DENSE_TENSOR_TYPES:
        error_msg = "Unsupported dense tensor data type: {}".format(tensor)
        raise ValueError(error_msg)
    raise NotImplementedError()

cdef TensorConfig* prepare_sparse_tensor(tensor, shape, NoiseConfig noise_config, is_scarse):
    if type(tensor) not in SPARSE_TENSOR_TYPES:
        error_msg = "Unsupported sparse tensor data type: {}".format(tensor)
        raise ValueError(error_msg)

    cdef vector[uint64_t] cpp_dims_vector
    cdef vector[uint32_t] cpp_columns_vector
    cdef vector[double] cpp_values_vector

    if type(tensor) == pd.DataFrame:
        idx_column_names = list(filter(lambda c: tensor[c].dtype==np.int64 or tensor[c].dtype==np.int32, tensor.columns))
        val_column_names = list(filter(lambda c: tensor[c].dtype==np.float32 or tensor[c].dtype==np.float64, tensor.columns))

        if len(val_column_names) != 1:
            error_msg = "tensor has {} float columns but must have exactly 1 value column.".format(len(val_column_names))
            raise ValueError(error_msg)

        idx = [i for c in idx_column_names for i in np.array(tensor[c], dtype=np.int32)]
        val = np.array(tensor[val_column_names[0]],dtype=np.float64)

        if shape is not None:
            cpp_dims_vector = shape
        else:
            cpp_dims_vector = [tensor[c].max() + 1 for c in idx_column_names]

        cpp_columns_vector = idx
        cpp_values_vector = val

        return new TensorConfig(make_shared[vector[uint64_t]](cpp_dims_vector), make_shared[vector[uint32_t]](cpp_columns_vector), make_shared[vector[double]](cpp_values_vector), NoiseConfig(), is_scarse)
    else:
        error_msg = "Unsupported sparse tensor data type: {}".format(tensor)
        raise ValueError(error_msg)

cdef TensorConfig* prepare_auxdata(in_tensor, shape, NoiseConfig noise_config):
    if type(in_tensor) in DENSE_TENSOR_TYPES:
        return prepare_dense_tensor(in_tensor, noise_config)
    elif type(in_tensor) in SPARSE_TENSOR_TYPES:
        return prepare_sparse_tensor(in_tensor, shape, noise_config, True)
    else:
        error_msg = "Unsupported tensor type: {}".format(type(in_tensor))
        raise ValueError(error_msg)

cdef (shared_ptr[TensorConfig], shared_ptr[TensorConfig]) prepare_data(train, test, shape, NoiseConfig noise_config):
    # Check train data type
    if (type(train) not in DENSE_MATRIX_TYPES and
        type(train) not in SPARSE_MATRIX_TYPES and
        type(train) not in DENSE_TENSOR_TYPES and
        type(train) not in SPARSE_TENSOR_TYPES):
        error_msg = "Unsupported train data type: {}".format(type(train))
        raise ValueError(error_msg)

    # Check test data type
    if test is not None:
        if (type(test) not in DENSE_MATRIX_TYPES and
            type(test) not in SPARSE_MATRIX_TYPES and
            type(test) not in DENSE_TENSOR_TYPES and
            type(test) not in SPARSE_TENSOR_TYPES):
            error_msg = "Unsupported test data type: {}".format(type(test))
            raise ValueError(error_msg)

    # Check train and test data for mismatch
    if test is not None:
        if ((type(train) in DENSE_MATRIX_TYPES or type(train) in SPARSE_MATRIX_TYPES) and
            (type(test) not in DENSE_MATRIX_TYPES and type(test) not in SPARSE_MATRIX_TYPES)):
            error_msg = "Train and test data must be the same type: {} != {}".format(type(train), type(test))
            raise ValueError(error_msg)
        if (type(train) in DENSE_MATRIX_TYPES or type(train) in SPARSE_MATRIX_TYPES) and train.shape != test.shape:
            raise ValueError("Train and test data must be the same shape: {} != {}".format(train.shape, test.shape))
        if type(train) in SPARSE_TENSOR_TYPES and train.ndim != test.ndim:
            raise ValueError("Train and test data must have the same number of dimensions: {} != {}".format(train.ndim, test.ndim))

    cdef TensorConfig* train_config
    cdef TensorConfig* test_config

    # Prepare train data
    if type(train) in DENSE_MATRIX_TYPES and len(train.shape) == 2:
        train_config = prepare_dense_matrix(train, noise_config)
    elif type(train) in SPARSE_MATRIX_TYPES:
        train_config = prepare_sparse_matrix(train, noise_config, True)
    elif type(train) in DENSE_TENSOR_TYPES and len(train.shape) > 2:
        train_config = prepare_dense_tensor(train, noise_config)
    elif type(train) in SPARSE_TENSOR_TYPES:
        train_config = prepare_sparse_tensor(train, shape, noise_config, True)
    else:
        error_msg = "Unsupported train data type: {}".format(type(train))
        raise ValueError(error_msg)
    train_config.setNoiseConfig(noise_config)

    # Prepare test data
    if test is not None:
        if type(test) in DENSE_MATRIX_TYPES and len(test.shape) == 2:
            test_config = prepare_dense_matrix(test, noise_config)
        elif type(test) in SPARSE_MATRIX_TYPES:
            test_config = prepare_sparse_matrix(test, noise_config, True)
        elif type(test) in DENSE_TENSOR_TYPES and len(test.shape) > 2:
            test_config = prepare_dense_tensor(test, noise_config)
            test_config = prepare_sparse_tensor(test, shape, noise_config, True)
        else:
            error_msg = "Unsupported test data type: {}".format(type(test))
            raise ValueError(error_msg)

    if test is not None:
        # Adjust train and test dims. Makes sense only for sparse tensors
        # It does not have any effect in other case
        if shape is None:
            for i in range(train_config.getDimsPtr().get().size()):
                max_dim = max(train_config.getDimsPtr().get().at(i), test_config.getDimsPtr().get().at(i))
                train_config.getDimsPtr().get()[0][i] = max_dim
                test_config.getDimsPtr().get()[0][i] = max_dim
        return shared_ptr[TensorConfig](train_config), shared_ptr[TensorConfig](test_config)
    else:
        return shared_ptr[TensorConfig](train_config), shared_ptr[TensorConfig]()

cdef class PySession:
    cdef shared_ptr[ISession] ptr;

    def init(self):
        return self.ptr.get().init()

    def step(self):
        return self.ptr.get().step()

    def getResult(self):
        """ Create Python list of ResultItem from C++ vector of ResultItem """
        cpp_result_items_ptr = self.ptr.get().getResult()
        py_result_items = []

        if cpp_result_items_ptr:
            for i in range(cpp_result_items_ptr.get().size()):
                cpp_result_item_ptr = &(cpp_result_items_ptr.get().at(i))
                py_result_item_coords = []
                for coord_index in range(cpp_result_item_ptr.coords.size()):
                    coord = cpp_result_item_ptr.coords[coord_index]
                    py_result_item_coords.append(coord)
                py_result_item = ResultItem( tuple(py_result_item_coords)
                                           , cpp_result_item_ptr.val
                                           , cpp_result_item_ptr.pred_1sample
                                           , cpp_result_item_ptr.pred_avg
                                           , cpp_result_item_ptr.var
                                           , cpp_result_item_ptr.stds
                                           )
                py_result_items.append(py_result_item)

        return Result(py_result_items, self.ptr.get().getRmseAvg())

    # Create and initialize smurff-cpp Config instance
    @classmethod
    def fromConfig(cls,
               Y              = None,
               Ytest          = None,
               data_shape     = None,
               priors         = [],
               side_info      = [],
               aux_data       = [],
               beta_precision = 10.0,
               num_latent     = 10,
               precision      = None,
               sn_init        = None,
               sn_max         = None,
               burnin         = 50,
               nsamples       = 400,
               tol            = 1e-6,
               direct         = True,
               seed           = None,
               threshold      = None,
               verbose        = True,
               init_model     = None,
               save_prefix    = None,
               save_extension = None,
               save_freq      = None,
               csv_status     = None):

        cdef Config config
        cdef NoiseConfig nc
        cdef shared_ptr[PVec] pos
        cdef string prior_type_str

def smurff(Y                = None,
           Ynoise           = None,
           Ytest            = None,
           data_shape       = None,
           priors           = [],
           side_info        = [],
           side_info_noises = [],
           aux_data         = [],
           aux_data_noises  = [],
           num_latent       = NUM_LATENT_DEFAULT_VALUE,
           burnin           = BURNIN_DEFAULT_VALUE,
           nsamples         = NSAMPLES_DEFAULT_VALUE,
           tol              = TOL_DEFAULT_VALUE,
           direct           = True,
           seed             = RANDOM_SEED_DEFAULT_VALUE,
           verbose          = VERBOSE_DEFAULT_VALUE,
           quite            = False,
           init_model       = None,
           save_prefix      = SAVE_PREFIX_DEFAULT_VALUE,
           save_extension   = SAVE_EXTENSION_DEFAULT_VALUE,
           save_freq        = SAVE_FREQ_DEFAULT_VALUE,
           csv_status       = STATUS_DEFAULT_VALUE,
           root_path        = None,
           ini_path         = None):

    cdef Config config
    cdef NoiseConfig noise_config
    cdef shared_ptr[ISession] session
    cdef shared_ptr[PVec] pos

    if root_path is not None:
        session = SessionFactory.create_py_session_from_root_path(root_path)
    elif ini_path is not None:
        success = config.restore(ini_path)
        if not success:
            raise f"Could not load init file '{ini_path}'"
        session = SessionFactory.create_py_session_from_config(config)
    else:
        # Create and initialize smurff-cpp Config instance

        if Ynoise is not None:
            noise_config.setNoiseType(stringToNoiseType(Ynoise[0]))

            if Ynoise[1] is not None:
                noise_config.setPrecision(Ynoise[1])
            if Ynoise[2] is not None:
                noise_config.setSnInit(Ynoise[2])
            if Ynoise[3] is not None:
                noise_config.setSnMax(Ynoise[3])
            if Ynoise[4] is not None:
                noise_config.setThreshold(Ynoise[4])
        else:
            noise_config = NoiseConfig(NOISE_TYPE_DEFAULT_VALUE)

        train, test = prepare_data(Y, Ytest, data_shape, noise_config)
        train.get().setNoiseConfig(noise_config)
        config.setTrain(train)

        if Ytest is not None:
            config.setTest(test)

        if len(side_info) == 0:
            side_info = [None] * len(priors)

        if len(aux_data) == 0:
            aux_data = [None] * len(priors)

        for prior_type_str in priors:
            config.getPriorTypes().push_back(stringToPriorType(prior_type_str.encode('UTF-8')))

        for i in range(len(side_info)):
            if len(side_info_noises) > 0 and side_info_noises[i] is not None:
                noise_config.setNoiseType(stringToNoiseType(side_info_noises[i][0]))
                if side_info_noises[i][1] is not None:
                    noise_config.setPrecision(side_info_noises[i][1])
                if side_info_noises[i][2] is not None:
                    noise_config.setSnInit(side_info_noises[i][2])
                if side_info_noises[i][3] is not None:
                    noise_config.setSnMax(side_info_noises[i][3])
                if side_info_noises[i][4] is not None:
                    noise_config.setThreshold(side_info_noises[i][4])
            else:
                noise_config = NoiseConfig(NOISE_TYPE_DEFAULT_VALUE)

            if side_info[i] is None:
                config.getSideInfo().push_back(shared_ptr[MatrixConfig]())
            else:
                config.getSideInfo().push_back(shared_ptr[MatrixConfig](prepare_sideinfo(side_info[i], noise_config)))

        for i in range(len(aux_data)):
            aux_data_list = aux_data[i]
            if aux_data_list is not None:
                pos = make_shared[PVec](len(priors))
                for j in range(len(aux_data_list)):
                    ad = aux_data_list[j]
                    if ad is not None:
                        if i == 0:
                            (&pos.get()[0].at(1))[0] += 1
                        elif i == 1:
                            (&pos.get()[0].at(0))[0] += 1
                        else:
                            (&pos.get()[0].at(i))[0] += 1
                    if len(aux_data_noises) > 0 and aux_data_noises[i] is not None and len(aux_data_noises[i]) > 0 and aux_data_noises[i][j] is not None:
                        noise_config.setNoiseType(stringToNoiseType(aux_data_noises[i][j][0]))
                        if aux_data_noises[i][j][1] is not None:
                            noise_config.setPrecision(aux_data_noises[i][j][1])
                        if aux_data_noises[i][j][2] is not None:
                            noise_config.setSnInit(aux_data_noises[i][j][2])
                        if aux_data_noises[i][j][3] is not None:
                            noise_config.setSnMax(aux_data_noises[i][j][3])
                        if aux_data_noises[i][j][4] is not None:
                            noise_config.setThreshold(aux_data_noises[i][j][4])
                    else:
                        noise_config = NoiseConfig(NOISE_TYPE_DEFAULT_VALUE)
                    config.getAuxData().push_back(shared_ptr[TensorConfig](prepare_auxdata(ad, data_shape, noise_config)))
                    config.getAuxData().back().get().setPos(pos.get()[0])

        config.setNumLatent(num_latent)
        config.setBurnin(burnin)
        config.setNSamples(nsamples)
        config.setTol(tol)
        config.setDirect(direct)

        if seed:
            config.setRandomSeedSet(True)
            config.setRandomSeed(seed)

        config.setVerbose(verbose)

        if quite:
            config.setVerbose(False)

        if init_model:
            config.setModelInitType(stringToModelInitType(init_model))
        else:
            config.setModelInitType(INIT_MODEL_DEFAULT_VALUE)

        if save_prefix:
            config.setSavePrefix(save_prefix)

        if save_extension:
            config.setSaveExtension(save_extension)

        if save_freq:
            config.setSaveFreq(save_freq)

        if csv_status:
            config.setCsvStatus(csv_status)

        session = SessionFactory.create_py_session_from_config(config)

    # Create and run session
    session.get().init()
    while session.get().step():
        pass

    # Create Python list of ResultItem from C++ vector of ResultItem
    cpp_result_items_ptr = session.get().getResult()
    py_result_items = []

    if cpp_result_items_ptr:
        for i in range(cpp_result_items_ptr.get().size()):
            cpp_result_item_ptr = &(cpp_result_items_ptr.get().at(i))
            py_result_item_coords = []
            for coord_index in range(cpp_result_item_ptr.coords.size()):
                coord = cpp_result_item_ptr.coords[coord_index]
                py_result_item_coords.append(coord)
            py_result_item = ResultItem( tuple(py_result_item_coords)
                                       , cpp_result_item_ptr.val
                                       , cpp_result_item_ptr.pred_1sample
                                       , cpp_result_item_ptr.pred_avg
                                       , cpp_result_item_ptr.var
                                       , cpp_result_item_ptr.stds
                                       )
            py_result_items.append(py_result_item)

    return Result(py_result_items, session.get().getRmseAvg())
