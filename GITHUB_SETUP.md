# GitHub setup

## 1. Create the repository

On GitHub, create a new repository named:

`floristic-diversity-yamoussoukro`

Do not add another README, licence, or .gitignore during repository creation because these files are already included here.

## 2. Upload the repository

From this folder, run:

```bash
git init
git branch -M main
git add .
git commit -m "Initial release: floristic inventory data and analysis workflow"
git remote add origin https://github.com/YOUR-USERNAME/floristic-diversity-yamoussoukro.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

## 3. Before making the repository public

Check:

- whether the Excel dataset can legally be redistributed;
- whether exact site/plot coordinates or other sensitive biodiversity information should be removed;
- whether the manuscript has already been published and whether the journal requires a particular data repository;
- whether the final DOI should be added to `CITATION.cff`.

## 4. Recommended repository release

For the first public release, use a version tag such as:

`v1.0.0`

and describe it as the dataset and analysis workflow accompanying the manuscript.
