# redeem_properties Documentation

This directory contains the source files and configuration for building the Sphinx documentation for the `redeem_properties` Python package.

## Prerequisites

To build the documentation locally, you need to have the Rust toolchain installed, as well as the Python dependencies for Sphinx.

1. **Install the Python package from source:**
   Because Sphinx uses `autodoc` to extract docstrings directly from the compiled PyO3 extension, you must build the extension first.
   ```bash
   cd ../
   pip install maturin
   maturin develop --release --features pretrained
   ```

2. **Install the documentation dependencies:**
   ```bash
   cd docs
   pip install -r requirements.txt
   ```

## Building the Documentation

To build the HTML documentation, run the following command from this `docs` directory:

```bash
make html
```

The generated HTML files will be placed in `build/html/`. You can open `build/html/index.html` in your web browser to view the documentation.

## Updating the Documentation

- **API Reference:** The API documentation is automatically generated from the Rust docstrings in `src/lib.rs` and the Python docstrings in `python/redeem_properties/__init__.py`. If you update a docstring, you must re-run `maturin develop` before running `make html`.
- **Guides:** The guides and tutorials are written in Markdown using MyST parser. You can edit `source/getting_started.md` or add new Markdown files to the `source/` directory. Remember to add any new files to the `toctree` in `source/index.md`.

## Read the Docs Integration

This documentation is configured to be automatically built and hosted on [Read the Docs](https://readthedocs.org/). 

The configuration for Read the Docs is located in the root of the repository at `.readthedocs.yaml`. It instructs Read the Docs to install `maturin`, compile the Rust extension, and then build the Sphinx documentation.
