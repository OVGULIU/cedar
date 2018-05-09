/* Generated by Cython 0.25.2 */

#ifndef __PYX_HAVE__pyplanes
#define __PYX_HAVE__pyplanes


#ifndef __PYX_HAVE_API__pyplanes

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#ifndef DL_IMPORT
  #define DL_IMPORT(_T) _T
#endif

__PYX_EXTERN_C DL_IMPORT(void) pyplane(int, int, int, double *, double *, int);
__PYX_EXTERN_C DL_IMPORT(void) pyplane27(int, int, int, double *, double *, int);
__PYX_EXTERN_C DL_IMPORT(void) pyplane_xz(int, int, int, double *, double *, int);
__PYX_EXTERN_C DL_IMPORT(void) pyplane_xz27(int, int, int, double *, double *, int);
__PYX_EXTERN_C DL_IMPORT(void) pyplane_yz(int, int, int, double *, double *, int);
__PYX_EXTERN_C DL_IMPORT(void) pyplane_yz27(int, int, int, double *, double *, int);

#endif /* !__PYX_HAVE_API__pyplanes */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initpyplanes(void);
#else
PyMODINIT_FUNC PyInit_pyplanes(void);
#endif

#endif /* !__PYX_HAVE__pyplanes */