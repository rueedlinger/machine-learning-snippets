# Build

## Run the Tests (nbval)

To test the Jupyter notebooks this project uses [nbval](https://github.com/computationalmodelling/nbval), which is a `py.test`
plugin for validating Jupyter notebooks.

This will check all Jupyter notebooks for errors.

```bash
pipenv run py.test --nbval-lax
```

Or you can run the tests in parallel with the _pytest-xdist_ plugin.

```bash
pipenv run py.test --nbval-lax -n auto --dist loadscope
```

## Upgrade Python Packages

Check which packages have changed.

```
pipenv update --outdated
```

This will upgrade everything.

```bash
pipenv update
```

## Git LFS

Some of the files (\*.png) are stored in Git LFS. When you want to work with them locally you need to install git-lfs.

Checkout out the data from Git LFS.

```bash
git lfs checkout
```

## Export the Jupyter Notebooks to Markdown

To export the Jupyter notebooks to Markdown just run the [export-notebooks.sh](export-notebooks.sh) script.
This scrip uses `nbconvert` to convert the Jupyter notebooks.

```bash
chmod 755 export-notebooks.sh
./export-notebooks.sh
```
