# Introduction
__clang-format__ is a great tool but yet some features are still missing for many years.
This project helps you to format `#pragma omp`, `#pragma unroll`, etc. with the right indent.

The name of this project comes from patched __clang-format__.
There are many usages brought by this script binary to help patch the formating rules of __clang-format__.

# Usage
* Use it the way you use `clang-format`
* You can also run with env var to replace any pragma `REPLACE_ANY_PRAGMA=1 ./p-clang-format ...`

# Vim Autoformat support
If you are interested in how this works or how to add this capability to your vim script,
please refer to this [blog](https://medicineyeh.wordpress.com/2017/07/13/clang-format-with-pragma/)

# Install
Simply copy this binary to any folder listed in `$PATH` or `/usr/bin/`.

# License
MIT License.

