# PSALM site

Astro static site deployed to GitHub Pages at
`https://SharathSPhD.github.io/PSALM/`. It presents the depth of the paper but
reframes it for technical and non-technical audiences, with the actual results
and figures published as each phase closes.

## Develop / build

```bash
cd site
npm install
npm run dev      # local preview
npm run build    # outputs to site/dist/
```

## Deploy

GitHub Pages is built and deployed by the `pages` workflow on pushes to `main`
(configure Pages source = GitHub Actions in the repo settings). The `base` path
in `astro.config.mjs` is `/PSALM` to match the project-pages URL.

## Content rule

Numbers and figures on the site come from verified, signed-off findings (the same
source as the paper), never from projections.
