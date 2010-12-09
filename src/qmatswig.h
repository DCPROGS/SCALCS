/*
   qmatswig.h
   Implementation of Python wrapper functions
   2008-09-19, C. Schmidt-Hieber (christsc_at_gmx.de)
*/

#ifndef _QMAT_SWIG_H_
#define _QMAT_SWIG_H_

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

    void init_matrix( double* inout_arr, int dim1, int dim2 );
    PyObject* p_inf( double* in_arr, int dim1, int dim2 );
    PyObject* mat_solve( double* in_arr, int dim1, int dim2 );
    PyObject* p( PyObject* t, PyObject* p_0, PyObject* p_inf, PyObject* lambda_r, PyObject* A );

#ifdef __cplusplus
}
#endif

#endif
