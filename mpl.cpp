// Copyright 2017 Carolina Feher da Silva <carolfsu@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <vector>
#include <utility>
#include <Python.h>
#include <cmath>
#include <cerrno>
#include <iostream>
#include <limits>
#include <cassert>

void mpl_rs(PyObject* xs, std::vector<double>& rs, int k, double A, double rho, double theta) {
    if (k == 0) {
        double mem = 0;
        for (unsigned int t = 0; t < rs.size(); t++) {
            int x = PyLong_AsLong(PyList_GET_ITEM(xs, t));
            rs[t] = theta*mem;
            mem = rho*A*mem + 2*x - 1;
        }
    }
    else {
        const int num_mem = (1 << k);
        int eta = 0;
        std::vector<double> mem(num_mem, 0);
        for (int t = 0; t < k; t++) {
            int x = PyLong_AsLong(PyList_GET_ITEM(xs, t));
            eta = (eta << 1) + x;
            rs[t] = 0;
        }
        for (unsigned int t = k; t < rs.size(); t++) {
            int x = PyLong_AsLong(PyList_GET_ITEM(xs, t));
            rs[t] = theta*mem[eta];
            for (int i = 0; i < num_mem; i++) {
                mem[i] = A*mem[i];
            }
            mem[eta] = rho*mem[eta] + 2*x - 1;
            eta = ((eta << 1) + x) % num_mem;
        }
    }
}

PyObject* logl(PyObject* xs, PyObject* ys, int ini, int end, int k, double A, double rho, double theta) {
    double ll = 0;
    std::vector<double> rs(end);
    mpl_rs(xs, rs, k, A, rho, theta);
    for(int t = ini; t < end; t++) {
        int y = PyLong_AsLong(PyList_GET_ITEM(ys, t));
        double f = rs[t];
        if (f > 0) {
            double e = exp(-f);
            if (y == 0) {
                ll -= f + log(e + 1);
            }
            else {
                ll -= log(e + 1);
            }
        }
        else {
            double e = exp(f);
            if (y == 0) {
                ll -= log(e + 1);
            }
            else {
                ll += f - log(e + 1);
            }
        }
    }
    return PyFloat_FromDouble(ll);
}

static PyObject *
mpl_logl(PyObject* self, PyObject* args)
{
    PyObject* xs = 0;
    PyObject* ys = 0;
    int ini;
    int end;
    int k;
    double A;
    double rho;
    double theta;
    if (!PyArg_ParseTuple(args, "OOiiiddd", &xs, &ys, &ini, &end, &k, &A, &rho, &theta)) {
        return 0;
    }
    return logl(xs, ys, ini, end, k, A, rho, theta);
}

PyObject* model(PyObject* xs, int k, double A, double rho, double theta) {
    unsigned int M = PyList_GET_SIZE(xs);
    std::vector<double> rs(M);
    mpl_rs(xs, rs, k, A, rho, theta);
    PyObject* p1s = PyTuple_New(PyList_GET_SIZE(xs));
    for(unsigned int t = 0; t < M; t++) {
        double f = rs[t];
        double p1;
        if (f > 40) {
            p1 = 1;
        }
        else {
            double m = exp(f);
            p1 = m / (1 + m);
        }
        PyTuple_SET_ITEM(p1s, t, PyFloat_FromDouble(p1));
    }
    return p1s;
}

static PyObject *
mpl_model(PyObject* self, PyObject* args)
{
    PyObject* xs = 0;
    int k;
    double A;
    double rho;
    double theta;
    if (!PyArg_ParseTuple(args, "Oiddd", &xs, &k, &A, &rho, &theta)) {
        return 0;
    }
    return model(xs, k, A, rho, theta);
}

static PyMethodDef MarkovModelMethods[] = {
    {"mpl_logl",  mpl_logl, METH_VARARGS,
    "Returns the log-likehood of the sequence ys."},
    {"mpl",  mpl_model, METH_VARARGS,
    "Simulates the MPL model, returns list of p1s."},
    {0, 0, 0, 0}        /* Sentinel */
};

static struct PyModuleDef mpl_module = {
   PyModuleDef_HEAD_INIT,
   "mpl",
   0,
   -1,
   MarkovModelMethods
};

PyMODINIT_FUNC
PyInit_mpl(void)
{
    return PyModule_Create(&mpl_module);
}
