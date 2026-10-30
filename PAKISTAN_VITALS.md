# Pakistan Vitals: how it works and how to set it up

Pakistan Vitals lives at **www.danialsyed.com/pakistan/**. It has a dashboard
of the economy's main figures, with charts and a source link on every row, and
a short weekly note. Every Friday an automated run updates both and sends you a
pull request. Nothing goes live until you merge it.

## How the weekly update works

1. **Friday, 6pm Pakistan time.** GitHub Actions starts
   `.github/workflows/pakistan-weekly.yml`.
2. **Claude does the research.** It follows the playbook in
   `.claude/skills/pakistan-weekly-update/SKILL.md`:
   - finds the week's releases (SBP reserves, SPI, auctions, market closes,
     and CPI, trade and the rest when they're due);
   - updates the data files in `_data/pakistan/`, citing each source;
   - writes the week's note in `_pakistan_notes/`.

   It can search and read the web and edit those files. It can't commit,
   push or run anything other than the two scripts below.
3. **The checks run.** `scripts/pakistan/derive.py` recomputes the derived
   figures (year-to-date changes, real interest rates, gaps to targets).
   `scripts/pakistan/check_data.py` then checks that:
   - every row has a source link;
   - the note and the headline only quote numbers that are in the data;
   - any big jump since last week is flagged.
4. **You get a pull request.** Its description lists what changed and where
   each figure came from, what Claude couldn't find, anything it was unsure
   of, and the check results. If something failed, the PR is a draft marked
   **[Needs fixes]** and the run shows as failed, so GitHub emails you.
5. **You review and merge.** GitHub Pages publishes the site a minute or two
   after you merge.

## One-time setup

**1. Merge the pull request that adds Pakistan Vitals.**

**2. Give the workflow access to Claude.** Pick one of these:

- *Anthropic API key (billed per use).*
  1. Create a key at console.anthropic.com.
  2. In the repo, go to **Settings → Secrets and variables → Actions → New
     repository secret**.
  3. Name it `ANTHROPIC_API_KEY` and paste in the key.
- *Your Claude subscription.*
  1. Run `claude setup-token` in Claude Code on your computer.
  2. Save the token as a secret named `CLAUDE_CODE_OAUTH_TOKEN`.
  3. In the workflow file, comment out the `anthropic_api_key` line and
     uncomment the `claude_code_oauth_token` line.

Never paste a key into a file in the repo. It belongs only in the secret.

**3. Let Actions open pull requests.** In **Settings → Actions → General →
Workflow permissions**, choose **Read and write permissions** and tick
**Allow GitHub Actions to create and approve pull requests**. Save.

**4. Test it.** Go to **Actions → Pakistan Vitals weekly update → Run
workflow**. A run takes about 10–30 minutes. Then open the pull request it
creates.

## Reviewing a weekly pull request (about 10 minutes)

- Read the summary at the top. It's the week in two or three sentences.
- Open two or three source links in the **Updated** table and check the
  figure against the page, especially anything listed under **Check before
  merging**.
- Read the note under **Files changed**. Edit wording right there on GitHub
  if you want to. If you change a number, change it in the data file too, or
  the check will fail on the next run.
- Merge. To skip a week, close the PR without merging.

## Sending the note out

The notes have their own feed at
**https://www.danialsyed.com/pakistan/feed.xml**. Point any RSS-to-email
service at it (Buttondown, Mailchimp and Substack all have one) and each note
you merge is emailed to subscribers. You can do the same for a WhatsApp
channel or LinkedIn post with an automation tool such as Zapier.

The figures are also published as one file at
**https://www.danialsyed.com/pakistan/data.json**, for anyone who wants to
reuse them.

## Cost and limits

- Each run reads a few dozen web pages. Watch the usage of the first few runs
  in the Anthropic Console (or your plan's usage page) and set a spend limit
  there.
- To make runs cheaper, add `--model claude-sonnet-5-5` to `claude_args` in
  the workflow. To cap how long a run can go on, lower `--max-turns`.
- GitHub turns off scheduled workflows in public repos after 60 days with no
  activity. Merging the weekly PRs counts as activity. If you stop for a
  while, re-enable the workflow in the Actions tab.

## Changing things by hand

- **A figure:** edit its row in `_data/pakistan/indicators.json` (value,
  display, period, source, url), then run:

  ```
  python3 scripts/pakistan/derive.py
  python3 scripts/pakistan/check_data.py
  ```
- **What the weekly run looks for, or how it writes:** edit
  `.claude/skills/pakistan-weekly-update/SKILL.md`. It's plain English.
- **The page itself:** `pakistan/index.html`, `assets/pakistan/pakistan.css`
  and `assets/pakistan/pakistan.js`.
- **The day or time:** the `cron` line in the workflow (times are UTC).

To preview the site on your computer, run `bundle exec jekyll serve` and open
http://localhost:4000/pakistan/.
