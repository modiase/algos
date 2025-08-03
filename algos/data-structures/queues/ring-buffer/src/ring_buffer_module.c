#include <Python.h>
#include "buffer.h"

typedef struct {
	PyObject_HEAD struct ring_buffer *buffer;
} PyRingBuffer;

static void PyRingBuffer_dealloc(PyRingBuffer *self)
{
	if (self->buffer) {
		PyObject *obj;
		while (ring_buffer_dequeue(self->buffer, &obj) ==
		       RING_BUFFER_SUCCESS) {
			Py_DECREF(obj);
		}
		ring_buffer_destroy(self->buffer);
	}
	Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *PyRingBuffer_new(PyTypeObject *type, PyObject *args,
				  PyObject *kwds)
{
	PyRingBuffer *self = (PyRingBuffer *) type->tp_alloc(type, 0);
	if (self != NULL) {
		self->buffer = NULL;
	}
	return (PyObject *) self;
}

static int PyRingBuffer_init(PyRingBuffer *self, PyObject *args, PyObject *kwds)
{
	static char *kwlist[] = { NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist)) {
		return -1;
	}

	self->buffer = ring_buffer_create(sizeof(PyObject *));
	if (!self->buffer) {
		PyErr_SetString(PyExc_MemoryError,
				"Failed to create ring buffer");
		return -1;
	}
	return 0;
}

static PyObject *PyRingBuffer_enqueue(PyRingBuffer *self, PyObject *args)
{
	PyObject *obj;
	if (!PyArg_ParseTuple(args, "O", &obj)) {
		return NULL;
	}

	if (!self->buffer) {
		PyErr_SetString(PyExc_RuntimeError,
				"Ring buffer not initialized");
		return NULL;
	}

	Py_INCREF(obj);
	int result = ring_buffer_enqueue(self->buffer, &obj);

	if (result != RING_BUFFER_SUCCESS) {
		Py_DECREF(obj);
		PyErr_SetString(PyExc_OverflowError, "Ring buffer is full");
		return NULL;
	}

	Py_RETURN_NONE;
}

static PyObject *PyRingBuffer_dequeue(PyRingBuffer *self,
				      PyObject *Py_UNUSED(ignored))
{
	if (!self->buffer) {
		PyErr_SetString(PyExc_RuntimeError,
				"Ring buffer not initialized");
		return NULL;
	}

	PyObject *obj;
	int result = ring_buffer_dequeue(self->buffer, &obj);

	if (result != RING_BUFFER_SUCCESS) {
		PyErr_SetString(PyExc_IndexError, "Ring buffer is empty");
		return NULL;
	}

	return obj;
}

static PyObject *PyRingBuffer_size(PyRingBuffer *self,
				   PyObject *Py_UNUSED(ignored))
{
	if (!self->buffer) {
		PyErr_SetString(PyExc_RuntimeError,
				"Ring buffer not initialized");
		return NULL;
	}
	return PyLong_FromSize_t(ring_buffer_size(self->buffer));
}

static PyObject *PyRingBuffer_is_empty(PyRingBuffer *self,
				       PyObject *Py_UNUSED(ignored))
{
	if (!self->buffer) {
		PyErr_SetString(PyExc_RuntimeError,
				"Ring buffer not initialized");
		return NULL;
	}
	return PyBool_FromLong(ring_buffer_is_empty(self->buffer));
}

static PyObject *PyRingBuffer_is_full(PyRingBuffer *self,
				      PyObject *Py_UNUSED(ignored))
{
	if (!self->buffer) {
		PyErr_SetString(PyExc_RuntimeError,
				"Ring buffer not initialized");
		return NULL;
	}
	return PyBool_FromLong(ring_buffer_is_full(self->buffer));
}

static PyObject *PyRingBuffer_clear(PyRingBuffer *self,
				    PyObject *Py_UNUSED(ignored))
{
	if (!self->buffer) {
		PyErr_SetString(PyExc_RuntimeError,
				"Ring buffer not initialized");
		return NULL;
	}
	// Dequeue all objects to properly decref them
	PyObject *obj;
	while (ring_buffer_dequeue(self->buffer, &obj) == RING_BUFFER_SUCCESS) {
		Py_DECREF(obj);
	}

	Py_RETURN_NONE;
}

// Support len() builtin
static Py_ssize_t PyRingBuffer_length(PyRingBuffer *self)
{
	if (!self->buffer) {
		return 0;
	}
	return (Py_ssize_t) ring_buffer_size(self->buffer);
}

// Support bool() builtin
static int PyRingBuffer_bool(PyRingBuffer *self)
{
	if (!self->buffer) {
		return 0;
	}
	return !ring_buffer_is_empty(self->buffer);
}

static PyMethodDef PyRingBuffer_methods[] = {
	{"enqueue", (PyCFunction) PyRingBuffer_enqueue, METH_VARARGS,
	 "Add object to the end of the ring buffer"},
	{"dequeue", (PyCFunction) PyRingBuffer_dequeue, METH_NOARGS,
	 "Remove and return object from the front of the ring buffer"},
	{"size", (PyCFunction) PyRingBuffer_size, METH_NOARGS,
	 "Return the current number of elements"},
	{"is_empty", (PyCFunction) PyRingBuffer_is_empty, METH_NOARGS,
	 "Return True if buffer is empty"},
	{"is_full", (PyCFunction) PyRingBuffer_is_full, METH_NOARGS,
	 "Return True if buffer is full"},
	{"clear", (PyCFunction) PyRingBuffer_clear, METH_NOARGS,
	 "Remove all elements from the buffer"},
	{NULL}
};

static PySequenceMethods PyRingBuffer_as_sequence = {
	.sq_length = (lenfunc) PyRingBuffer_length,
};

static PyNumberMethods PyRingBuffer_as_number = {
	.nb_bool = (inquiry) PyRingBuffer_bool,
};

static PyTypeObject PyRingBufferType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	    .tp_name = "ring_buffer.RingBuffer",
	.tp_doc =
	    "Dynamic ring buffer for Python objects with automatic resizing",
	.tp_basicsize = sizeof(PyRingBuffer),
	.tp_itemsize = 0,
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
	.tp_new = PyRingBuffer_new,
	.tp_init = (initproc) PyRingBuffer_init,
	.tp_dealloc = (destructor) PyRingBuffer_dealloc,
	.tp_methods = PyRingBuffer_methods,
	.tp_as_sequence = &PyRingBuffer_as_sequence,
	.tp_as_number = &PyRingBuffer_as_number,
};

static PyModuleDef ring_buffer_module = {
	PyModuleDef_HEAD_INIT,
	.m_name = "ring_buffer",
	.m_doc = "High-performance ring buffer implementation for Python",
	.m_size = -1,
};

PyMODINIT_FUNC PyInit_ring_buffer(void)
{
	PyObject *m;

	if (PyType_Ready(&PyRingBufferType) < 0)
		return NULL;

	m = PyModule_Create(&ring_buffer_module);
	if (m == NULL)
		return NULL;

	Py_INCREF(&PyRingBufferType);
	if (PyModule_AddObject(m, "RingBuffer", (PyObject *) & PyRingBufferType)
	    < 0) {
		Py_DECREF(&PyRingBufferType);
		Py_DECREF(m);
		return NULL;
	}

	return m;
}
