#ifndef __LIBMYPY_H__
#define __LIBMYPY_H__

#include <Python.h>

PyObject * fetch_parameters(PyObject *, unsigned int**, double**, int*);
PyObject * probability(PyObject *, PyObject *);
PyObject * expected_length(PyObject *, PyObject *);

#endif
