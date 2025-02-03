#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include "libmypy.h"
#include "algo.h"

PyObject * fetch_parameters(
    PyObject* args,
    unsigned int** graph,
    double** ps,
    int* index
    ){
    Py_ssize_t byte_len = 0;      // Length of the bytes object
    PyObject* py_list = NULL;

    // Parse a single bytes object from `args`.
    // "y#" = parse read-only bytes + length (Python 3).
    if (!PyArg_ParseTuple(args, "y#Oi", graph, &byte_len, &py_list, index)) {
        return NULL; // indicates error
    }


    // Check if the second argument is a list or iterable
    if (!PyList_Check(py_list) && !PySequence_Check(py_list)) {
        PyErr_SetString(PyExc_TypeError, "Second argument must be a list or iterable of floats.");
        return NULL;
    }

    // Convert the Python list/iterable to a C array of doubles
    Py_ssize_t list_size = PySequence_Size(py_list);
    if (list_size < 0) {
        return NULL;  // PySequence_Size sets an exception if it fails
    }

    * ps = (double*) malloc(list_size * sizeof(double));
    if (!*ps) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for the list.");
        return NULL;
    }

    for (Py_ssize_t i = 0; i < list_size; i++) {
        PyObject* item = PySequence_GetItem(py_list, i);  // Borrowed reference
        if (!PyFloat_Check(item)) {
            free(*ps);
            PyErr_SetString(PyExc_TypeError, "All elements in the list must be floats.");
            return NULL;
        }
        (* ps)[i] = PyFloat_AsDouble(item);
        Py_DECREF(item);  // Clean up borrowed reference
    }
}


PyObject * probability(PyObject* self, PyObject* args) {    

    unsigned int* graph = NULL;
    double* ps = NULL;
    int index = 0;
    fetch_parameters(args, &graph, &ps, &index);
    double result = 0.0;
    Py_BEGIN_ALLOW_THREADS
	result = prob(graph, ps, index);
    Py_END_ALLOW_THREADS

    free(ps);
    // Return it as a Python float
    return PyFloat_FromDouble(result);
}


PyObject * expected_length(PyObject* self, PyObject* args) {    

    unsigned int* graph = NULL;
    double* ps = NULL;
    int index = 0;
    fetch_parameters(args, &graph, &ps, &index);

    double result = 0.0;
    Py_BEGIN_ALLOW_THREADS
	result = explen(graph, ps, index);
    Py_END_ALLOW_THREADS

    free(ps);
    // Return it as a Python float
    return PyFloat_FromDouble(result);
}
