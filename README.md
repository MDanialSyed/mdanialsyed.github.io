# danialsyed.com

Source for **[www.danialsyed.com](https://www.danialsyed.com)**, the personal site and blog of M. Danial Syed on data science and AI for economics.

It's a [Jekyll](https://jekyllrb.com) site built on the Clean Blog theme by Start Bootstrap. GitHub Pages builds and publishes it automatically from the `master` branch.

## Writing

- **Posts:** add a Markdown file to `_posts/` named `YYYY-MM-DD-short-title.md`, with front matter like this:

  ```yaml
  ---
  layout: post
  title: "Post title"
  subtitle: "One-line summary, also used when the post is shared"
  background: '/img/posts/short-title/cover.jpg'
  math: true   # only if the post has equations
  ---
  ```

  Put the post's images in `img/posts/short-title/` and give each one a description: `![What the image shows](/img/posts/short-title/chart.png)`.
- **Pages:** `about.md`, `contact.html` and `posts/index.html`.
- **Settings** (name, email, social links, contact form, analytics) are in `_config.yml`.

## Previewing locally

```sh
bundle install
bundle exec jekyll serve
```

Then open http://localhost:4000. The `Gemfile` uses the `github-pages` gem, so the preview uses the same versions as the live site.

## Security notes

- All scripts, styles and fonts are served from this site, pinned to exact versions in `assets/vendor/`. No third-party scripts load unless Google Analytics is switched on in `_config.yml`.
- A Content Security Policy in `_includes/head.html` blocks anything else. If you embed something new (a video, a chart from another site), add its domain there, or it won't load.
- Never commit passwords, API keys or tokens. Keep them in GitHub's encrypted secrets.

## Licence

Posts, images and other content © M. Danial Syed, all rights reserved. See [LICENSE](LICENSE).

The theme code is MIT-licensed by Start Bootstrap (see [THEME-LICENSE](THEME-LICENSE)). The bundled libraries and fonts keep their own licences.
