/*
   qmatswig.cxx
   Python wrapper functions
   2008-09-16, C. Schmidt-Hieber (christsc_at_gmx.de)
*/

#include "qmatswig.h"

#include <math.h>
#include <Python.h>
#include <iostream>
#include <blitz/array.h>
#include <numpy/arrayobject.h>
#include <vector>

#define array_data(a)          (((PyArrayObject *)a)->data)

#ifdef __cplusplus
extern "C" {
#endif
    /* LAPACK routines */
    extern int dgetri_(int *n, double *a, int *lda, int *ipiv, double *work,
                       int *lwork, int *info);
    extern int dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv,
                       int *info);
    extern int dgeev_(char *jobvl, char* jobvr, int* n, double* a, int* lda,
                      double* wr, double* wi, double* vl, int* ldvl, double* vr,
                      int* ldvr, double* work, int* lwork, int* info);
    
    /* BLAS routines */
    extern int dgemm_(char* transa, char* transb, int *m, int* n, int *k,
                      double *alpha, double *a, int *lda, double *b, int *ldb,
                      double *beta, double *c, int* ldc);
    extern int dgemv_(char* trans, int *m, int* n, double *alpha, double *a,
                      int *lda, double *x, int *incx, double *beta, double *y,
                      int* incy);
    
    void init_matrix(double* inout_arr, int dim1, int dim2) {

        /* Updates the diagonal elements of Q in-place,
           i.e. Q will be changed by this function. */
        // Make sure that Q has square shape:
        if (dim1 != dim2) {
            std::cout << "Q doesn't have square shape in init_matrix();"
                      << "aborting now.\n";
            exit(1);
        }
        for (int d=0; d < dim1; ++d) {
            inout_arr[d+d*dim1]=0;
            // sum up all elements in a row:
            double sum = 0;
            for (int col = 0; col < dim2; ++col) {
                sum += inout_arr[d*dim2 + col];
            }
            inout_arr[d+d*dim1] = -sum;
        }
        return;
    }
    
    blitz::Array<double, 2> product_blas(const blitz::Array<double, 2>& A,
                                         const blitz::Array<double, 2>& B) {
        char transa = 'N';
        char transb = 'N';
        
        int m = A.shape()[0];
        int n = B.shape()[1];
        int k = A.shape()[1];
        double alpha = 1.0;
        int lda = m;
        int ldb = B.shape()[0];
        double beta = 0;
        int ldc = m;
        
        blitz::Array<double, 2> prod(m, n);
        prod = 0;

        // deep copies (probably unnecessary, would be nice to get rid of)
        blitz::Array<double, 2> At(k, m);
        blitz::Array<double, 2> Bt(n, ldb);
        At = A.copy().transpose(1,0);
        Bt = B.copy().transpose(1,0);
        
        int res = dgemm_(&transa, &transb, &m, &n, &k, &alpha, At.data(), &lda,
                          Bt.data(), &ldb, &beta, prod.data(), &ldc);
        
        return prod.transpose(1,0);
    }

    /* Hand-coded version; slightly slower than BLAS */
    blitz::Array<double, 2> product(const blitz::Array<double, 2>& A,
                                    const blitz::Array<double, 2>& B) {
        int n = A.shape()[0];
        int m = A.shape()[1];
        int k = B.shape()[1];
        blitz::Array<double, 2> prod(n, k);
        prod = 0;
        for (int i=0; i<n; ++i) {
            for (int j=0; j<k; ++j) {
                for (int r=0; r < m; ++r) {
                    prod(i,j) += A(i,r)*B(r,j);
                }
            }
        }
        return prod;
    }

    blitz::Array<double, 2> product_v_blas(const blitz::Array<double, 1>& x,
                                           const blitz::Array<double, 2>& A) {
        char trans = 'N';
        int m = A.shape()[0];
        int n = A.shape()[1];
        double alpha = 1.0;
        int lda = m;
        int incx = 1;
        double beta = 0;
        int incy = 1;
        
        blitz::Array<double, 2> prod(1,m);
        prod = 0;

        // deep copies (probably unnecessary, would be nice to get rid of)
        blitz::Array<double, 2> At(n, m);
        blitz::Array<double, 1> xt(m);
        At = A.copy().transpose(1,0);
        xt = x.copy(); // no need to transpose, has only one dimension
        
        int res = dgemv_(&trans, &m, &n, &alpha, At.data(), &lda,
                          xt.data(), &incx, &beta, prod.data(), &incy);
        
        return prod; // must not be transposed; has only one dimension
    }
    
    /* Hand-coded version; slightly slower than BLAS */
    blitz::Array<double, 2> product_v(const blitz::Array<double, 1>& A,
                                      const blitz::Array<double, 2>& B) {
        int n = 1;
        int k = B.shape()[1];
        blitz::Array<double, 2> prod(1,k);
        prod = 0;
        int i=0;
        for (int j=0; j<k; ++j) {
            for (int r=0; r < B.shape()[0]; ++r) {
                prod(i,j) += A(r)*B(r,j);
            }
        }
        return prod;
    }

    void inverse(blitz::Array<double, 2>& A) {
        
        /* inverses A in place */

        int m = A.shape()[0];
        int n = A.shape()[1];
        int lda_f = m;
        int ipiv_size = (m < n) ? m : n;
        blitz::Array<int, 1> ipiv(ipiv_size); ipiv = 0;
        int info=0;

        dgetrf_(&m, &n, A.transpose(1,0).data(), &lda_f, ipiv.data(), &info);
        if (info != 0)
            std::cout << "Error in LAPACK dgetrf\n";
        int lwork = -1;
        blitz::Array<double, 1> work(1); work = 0;
        dgetri_(&m, A.transpose(1,0).data(), &lda_f, ipiv.data(), work.data(),
                &lwork, &info);
        lwork = (int)work(0);
        work.resize(lwork);
        dgetri_(&m, A.transpose(1,0).data(), &lda_f, ipiv.data(), work.data(),
                &lwork, &info);
        if (info != 0)
            std::cout << "Error in LAPACK dgetri\n";
    }

    void wrap_array() {
        import_array();
    }
    
    PyObject* p_inf(double* in_arr, int dim1, int dim2) {
        /* Calculates p_inf for a Q matrix. Eq. 16, 17*/

        wrap_array();
#ifdef _CDEBUG
        std::cout << "Entered p_inf\n";
#endif
        bool debug = true;
        // Make sure that Q has square shape:
        if (dim1 != dim2) {
            std::cout << "Q doesn't have square shape in p_inf(); aborting now.\n";
            exit(1);
        }

        blitz::Array<double, 2> S(in_arr, blitz::shape(dim1, dim2),
                                  blitz::duplicateData);

        // add a unit vector as a column:
        S.resizeAndPreserve(dim1, dim2+1);
        S(blitz::Range(0,dim1-1), dim2) = 1.0;

        blitz::Array<double, 2> SST (product_blas(S, S.transpose(1,0)));

        inverse(SST);

        blitz::Array<double, 1> u(dim2); u=1.0;
        blitz::Array<double, 2> p_mat(product_v_blas(u, SST));
        
        npy_intp dims[1] = {dim1};
        PyObject* np_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        double* gDataP = (double*)array_data(np_array);
        blitz::TinyVector<int,1> dimsb(dims[0]);
        blitz::Array<double, 1> gData(gDataP, dimsb, blitz::neverDeleteData);
        gData = p_mat(0, blitz::Range(0,dim1));
#ifdef _CDEBUG
        if (np_array == NULL) {
            std::cout << "NULL pointer in p_inf; exiting now\n";
            exit(1);
        }
        std::cout << "Exiting p_inf\n";
#endif
        return np_array;
    }
    
    PyObject* mat_solve(double* in_arr, int dim1, int dim2) {
        /* Returns the solutions to the equations given by
           Colquhoun & Hawkes, chapter 20, as a tuple:
           lambda, A. Eq. 88, 89. */

        wrap_array();
#ifdef _CDEBUG
        std::cout << "Entered mat_solve\n";
#endif
        char jobvl = 'N';
        char jobvr = 'V';
        int n = dim1;
        blitz::Array<double, 2> Q(in_arr, blitz::shape(dim1, dim2),
                                  blitz::duplicateData);
        blitz::Array<double, 2> Qt(dim2, dim1); // need a deep copy
        Qt = -Q.transpose(1,0);

        int lda = dim1;

        npy_intp dims_L[1] = {dim1};
        PyObject* lambda_np = PyArray_SimpleNew(1, dims_L, NPY_DOUBLE);
        double* gLambdaP = (double*)array_data(lambda_np);
        
        blitz::TinyVector<int,1> dimsb_L(dims_L[0]);
        blitz::Array<double, 1> gLambda(gLambdaP, dimsb_L,
                                        blitz::neverDeleteData);
        blitz::Array<double, 1> wi(dim1); wi=0;
        int ldvl = 1;
        blitz::Array<double, 1> vl(1); vl=0;
        int ldvr = dim1;
        blitz::Array<double, 2> Y(ldvr, n); Y=0;
        int lwork = -1;
        blitz::Array<double, 1> work(1); work=0;
        int info = 0;

        // Estimate required working memory:
        dgeev_(&jobvl, &jobvr, &n, Qt.data(), &lda, gLambda.data(), wi.data(),
                vl.data(), &ldvl, Y.data(), &ldvr, work.data(), &lwork, &info);
        lwork = (int)work(0);
        work.resize(lwork);

        // get eigenvalues- and vectors
        dgeev_(&jobvl, &jobvr, &n, Qt.data(), &lda, gLambda.data(), wi.data(),
                vl.data(), &ldvl, Y.data(), &ldvr, work.data(), &lwork, &info);
        if (info != 0)
            std::cout << "Error in LAPACK dgeev\n";
        Y.transposeSelf(1,0);

        blitz::Array<double, 2> X(ldvr, n);
        X = Y;
        inverse(Y);

        npy_intp dims_A[3] = {Y.shape()[0], X.shape()[0], Y.shape()[1]};
        PyObject* A_np = PyArray_SimpleNew(3, dims_A, NPY_DOUBLE);
        double* gAP = (double*)array_data(A_np);

        blitz::TinyVector<int,3> dimsb_A(dims_A[0],dims_A[1],dims_A[2]);
        blitz::Array<double, 3> gA(gAP, dimsb_A, blitz::neverDeleteData);
        for (int i=0; i<Y.shape()[0]; ++i) {
            gA(i,blitz::Range::all(), blitz::Range::all()) =
                product_blas(X(blitz::Range::all(), blitz::Range(i,i)),
                              Y(blitz::Range(i,i), blitz::Range::all()));
        }

        // sort lambda and A. std::sort won't work here because
        // we need to synchronize lambda and A, so we use a
        // simple bubble sort.
        int n_b = gLambda.shape()[0];
        bool exchanged = true;
        while (exchanged && n_b >= 0) {
            exchanged = false;
            blitz::Array<double, 2> temp_a(gA(0,blitz::Range::all(),
                                              blitz::Range::all()).shape());
            temp_a = 0;
            for (int i=0; i<n_b-1; ++i) {
                if (gLambda(i) > gLambda(i+1)) {
                    temp_a = gA(i,blitz::Range::all(),blitz::Range::all());
                    gA(i,blitz::Range::all(),blitz::Range::all()) =
                        gA(i+1,blitz::Range::all(),blitz::Range::all());
                    gA(i+1,blitz::Range::all(),blitz::Range::all()) = temp_a;
                    std::swap(gLambda(i), gLambda(i+1));
                    exchanged = true;
                }
            }
            n_b -= 1;
        }
        
        PyObject* A_list = PyList_New(2);
        PyList_SetItem(A_list, 0, lambda_np);
        PyList_SetItem(A_list, 1, A_np);

        return A_list;
    }

    PyObject* p(PyObject* t, PyObject* p_0, PyObject* p_inf,
                PyObject* lambda_r, PyObject* A) {
        
        /* Returns the probality of a channel being in a certain 
           state at time t. The state is characterised by p_inf (the
           probablity at equilibrium) and p_0 (the initial probability).
           The rates lambda_r and the amplitude terms A are the eigen-
           values and the eigenvectors of -Q, respectively. Eq. 24,26,27. */

        wrap_array();
        
        npy_intp t_len = PyArray_DIM(t, 0);
        npy_intp p_0_len = PyArray_DIM(p_0, 0);
        npy_intp p_inf_len = PyArray_DIM(p_inf, 0);
        npy_intp lambda_r_len = PyArray_DIM(lambda_r, 0);
        npy_intp* A_len = PyArray_DIMS(A);

        blitz::TinyVector<int,3> Ab_len(A_len[0],A_len[1],A_len[2]);
        blitz::Array<double, 3>
            A_np((double*)PyArray_DATA(A), Ab_len, blitz::neverDeleteData);
        blitz::Array<double, 1>
            t_np((double*)PyArray_DATA(t), blitz::shape(t_len),
                 blitz::neverDeleteData);
        blitz::Array<double, 1>
            p_0_np((double*)PyArray_DATA(p_0), blitz::shape(p_0_len),
                   blitz::neverDeleteData);
        blitz::Array<double, 1>
            p_inf_np((double*)PyArray_DATA(p_inf), blitz::shape(p_inf_len),
                     blitz::neverDeleteData);
        blitz::Array<double, 1>
            lambda_r_np((double*)PyArray_DATA(lambda_r),
                        blitz::shape(lambda_r_len), blitz::neverDeleteData);

        npy_intp dims_p[2] = {p_0_len, t_len};
        PyObject* p_ret = PyArray_SimpleNew(2, dims_p, NPY_DOUBLE);
        double* gP_retP = (double*)array_data(p_ret);
        
        blitz::TinyVector<int,2> dimsb_p(dims_p[0],dims_p[1]);
        blitz::Array<double, 2>
            gP_ret(gP_retP, dimsb_p, blitz::neverDeleteData);
        blitz::Array<double, 1> sum_p(t_len); sum_p = 0;
        for (int j=0; j<p_0_len; ++j) {
            sum_p = 0;
            for (int i=1; i<A_len[0]; ++i) {
                double w_ij = 0;
                for (int r=0; r<A_len[2]; ++r) {
                    w_ij += p_0_np(r) * A_np(i,r,j);
                }
                sum_p += w_ij * exp(-t_np * lambda_r_np(i));
            }
            gP_ret(j,blitz::Range::all()) = p_inf_np(j) + sum_p;
        }
        return p_ret;
    }

#ifdef __cplusplus
}
#endif
