---
title: 删除已经提交的Ignore file
date: 2017-03-31 18:55:30
categories: git
tags:
- git
- ignore
toc: false
---

<!-- more -->

To untrack a single file that has already been added/initialized to your repository, i.e., stop tracking the file but not delete it from your system use:

```bash
git rm --cached filename
```

To untrack every file that is now in your `.gitignore`:

First commit any outstanding code changes, and then, run this command:

```bash
git rm -r --cached .
```

This removes any changed files from the index(staging area), then just run:

```bash
git add .
```

Commit it:

```bash
git commit -m ".gitignore is now working"
```
