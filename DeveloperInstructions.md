# How to commit changes to the hg repository #

  1. Create a .hgrc file in your home directory so that your user name is added to any changes you apply to the repository. Mine simply looks like that:
```
    [ui]
    username = Christoph Schmidt-Hieber <christsc@gmx.de>
```
  1. Before you start contributing, check out a fresh working copy (you only need to do this once):
```
    $ hg clone http://dc-pyps.googlecode.com/hg/ dc-pyps
```
  1. Take a look at the .hgignore file to see which files are _excluded_ from the repository. Note that any files that you add that are _not_ in .hgignore will be _included_ in the repository.
  1. **Keep the repository as small as possible.** It should only consist of text files, i.e. there shouldn't be any binaries in the directory. In particular, all building should be performed in an out-of-tree directory so that the binary files (`*`.o, `*`.so etc.) are not added to the repository. This includes building the documentation. If you absolutely need some large files in-tree, put them into a directory that is blacklisted in the .hgignore file.
  1. It's very important to **update your local working repository from the Google server right before committing any changes**. Otherwise, we have to spend a lot of time merging revisions.
  1. When you commit changes, you will be asked for a password. Strangely, this isn't your Google account password by default, but rather some separate password that you can find here:
> > http://code.google.com/hosting/settings
> > once you are logged in.
> > Alternatively, you can use your default Google password by adjusting the Security settings in your profile.
  1. When committing, a text editor will be started to add a brief comment on your changes. Keep this as short as possible (e.g. "Updated documentation").
  1. For some reason that I forgot, you need a google email account to commit changes with hg (I think it had something to do with the `@` sign that you can't use in a hg login name).
  1. Here's the shell script that I use for committing:
```
    #! /bin/bash
    
    REMOTE=https://christoph.schmidthieber@dc-pyps.googlecode.com/hg
    
    # make sure we have the latest revision before committing:
    echo hg -v pull -u $REMOTE
    hg -v pull -u $REMOTE
    
    echo hg -v addremove
    hg -v addremove
    
    echo hg -v ci
    hg -v ci
    
    echo hg -v push $REMOTE
    hg -v push $REMOTE
```