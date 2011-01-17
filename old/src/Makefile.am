lib_LTLIBRARIES = libqmat.la

libqmat_la_SOURCES = $(srcdir)/qmat_wrap.cxx $(srcdir)/qmatswig.cxx

noinst_HEADERS = $(srcdir)/qmatswig.h

$(srcdir)/qmat_wrap.cxx : $(srcdir)/qmat.i
	$(SWIG) $(SWIG_PYTHON_OPT) -o $@ $<

INCLUDES = $(LIBNUMPY_INCLUDES)

libqmat_la_CPPFLAGS = $(SWIG_PYTHON_CPPFLAGS) $(LIBBLITZ_CFLAGS) -I$(top_srcdir)/src
libqmat_la_LDFLAGS = $(LIBLAPACK_LDFLAGS) $(LIBBLAS_LDFLAGS) $(LIBBLITZ_LIBS) -lm

EXTRA_DIST = qmat.i qmat.py qmatpy.py

install-exec-hook:
	$(MKDIR_P) ${PYTHON_SITE_PKG}/qmat
	$(LN_S) -f ${prefix}/lib/${QMAT_PYTHON_LIBNAME} ${PYTHON_SITE_PKG}/qmat/_qmat.so
	cp $(srcdir)/qmat.py ${PYTHON_SITE_PKG}/qmat
	cp $(srcdir)/qmatpy.py ${PYTHON_SITE_PKG}/qmat
	echo qmat > ${PYTHON_SITE_PKG}/qmat.pth
	echo qmat > ${PYTHON_SITE_PKG}/qmatpy.pth
