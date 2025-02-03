from setuptools import setup, Extension

setup_args = dict(
    ext_modules=[
        Extension(
            name="probycapi",
            sources=[
                "proby/probycapi/bind.c",
                "proby/probycapi/libmypy.c",
                "proby/probycapi/algo.c",
            ],
            include_dirs=["proby/probycapi"],
            py_limited_api=True,
        )
    ],
)
setup(**setup_args)
