%define DOCSTRING
"qmat: Solving kinetic schemes using the Q matrix approach by
Colquhoun & Hawkes
Python wrapper written by C. Schmidt-Hieber."
%enddef

%module(docstring=DOCSTRING) qmat

%{
#define SWIG_FILE_WITH_INIT
#include "qmatswig.h"
%}
%include "numpy.i"
%init %{
import_array();
%}

%define %apply_numpy_typemaps(TYPE)

%apply (TYPE* INPLACE_ARRAY2, int DIM1, int DIM2) {(TYPE* inout_arr, int dim1, int dim2)};
%apply (TYPE* IN_ARRAY2, int DIM1, int DIM2) {(TYPE* in_arr, int dim1, int dim2)};

%enddef    /* %apply_numpy_typemaps() macro */

%apply_numpy_typemaps(double)

//--------------------------------------------------------------------
%feature("autodoc", 0) init_matrix;
%feature("docstring", "Initializes the diagonal elements of Q in-place,
i.e. Q will be changed by this function.
Ch. 18, Eq. 126, 127
      
Arguments:
inout_arr -- The Q matrix to be initialized. This needs to be a
             square-shaped 2D numpy array.

Returns:
Q will be updated in-place.") init_matrix;
void init_matrix(double* inout_arr, int dim1, int dim2);
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) p_inf;
%feature("docstring", "Calculates p_inf for a Q matrix.
Ch. 20, Eq. 16, 17
      
Arguments:
in_arr -- The initialized Q matrix.

Returns:
p_inf as a numpy array.") p_inf;
PyObject* p_inf(double* in_arr, int dim1, int dim2);
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) mat_solve;
%feature("docstring", "Calculates the spectral matrices according to
Ch. 20, Eq. 88, 89
      
Arguments:
in_arr -- The initialized Q matrix.

Returns:
(lambda, A) as a tuple.") mat_solve;
PyObject* mat_solve(double* in_arr, int dim1, int dim2);
//--------------------------------------------------------------------


//--------------------------------------------------------------------
%feature("autodoc", 0) p;
%feature("docstring", "Returns the probality of a channel being in a certain 
state during the time period t. The state is characterised by p_inf (the
probablity at equilibrium) and p_0 (the initial probability).
The rates lambda_r and the amplitude terms A are the eigen-
values and the eigenvectors of -Q, respectively, as returned
by a call to mat_solve.
Ch. 20, Eq. 24, 26, 27.

Arguments:
t        -- time range (e.g. np.arange(0,100,0.01))
p_0      -- The initial probability (as returned by a call to p_inf or
            to this function)
p_inf    -- The probability at equilibrium (as returned by a call to
            p_inf)
lambda_r -- Eigenvalues and...
A        -- ... eigenvectors of -Q, as returned by mat_solve

Returns:
Probability matrix p. First dimension denotes the states, second
dimension denotes time.") p;
PyObject* p(PyObject* t, PyObject* p_0, PyObject* p_inf, PyObject*
            lambda_r, PyObject* A);
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%pythoncode {

}
//--------------------------------------------------------------------
