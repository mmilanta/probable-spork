#include <Python.h>
#include "libmypy.h"

char probabilityfunc_docs[] = "Compute Probability of winning the match.";
char expected_lengthfunc_docs[] = "Compute Expected leng of the match.";

PyMethodDef proby_funcs[] = {
	{	"probability",
		(PyCFunction)probability,
		METH_VARARGS,
		probabilityfunc_docs},
	{	"expected_length",
		(PyCFunction)expected_length,
		METH_VARARGS,
		expected_lengthfunc_docs},

	{	NULL}
};

char probymod_docs[] = "Module to compute fast graph algorithms.";

PyModuleDef probycapi_mod = {
	PyModuleDef_HEAD_INIT,
	"probycapi",
	probymod_docs,
	-1,
	proby_funcs,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC PyInit_probycapi(void) {
	return PyModule_Create(&probycapi_mod);
}
