# Build

## Run the Tests (nbval)

To test the Jupyter notebooks this project uses [nbval](https://github.com/computationalmodelling/nbval), which is a `py.test`
plugin for validating Jupyter notebooks.

This will check all Jupyter notebooks for errors.

```bash
pipenv run py.test --nbval-lax
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

Some of the files (\*.png) are stored in Git LFS. When you want to work with them locally you need to install git-lfs and check them out.

```bash
git lfs checkout
```

## CI Build (GitHub Actions)

See the GitHub Actions [build.yml](.github/workflows/build.yml) file for more details.
![CI Build](https://github.com/rueedlinger/machine-learning-snippets/workflows/CI%20Build/badge.svg)

## Export the Jupyter Notebooks to Markdown

To export the Jupyter notebooks to Markdown just run the [export-notebooks.sh](export-notebooks.sh) script.
This scrip uses `nbconvert` to convert the Jupyter notebooks.

```bash
chmod 755 export-notebooks.sh
./export-notebooks.sh
```
