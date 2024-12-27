# Developers Guide

## Setup

> [!NOTE]
> If you are developing on Stanford's sapling cluster, instead see the instructions at [TODO].
> If you don't know what this means, don't worry about it: just continue reading.

1. flexflow-train uses [nix](https://nix.dev/manual/nix/2.24/) to manage dependencies and the development environment. 
   There exist a number of ways to install nix, but we recommend one of the following:

   1. If you have root permissions: [DeterminateSystems/nix-installer](https://github.com/DeterminateSystems/nix-installer)

   2. If you don't have root permissions: [DavHau/nix-portable](https://github.com/DavHau/nix-portable). 
      
      > [!NOTE]
      > nix-portable does not work particularly well if the nix store is in NFS, so if you are running on an 
      > HPC cluster where the home directory is mounted via NFS we recommend setting the `NP_LOCATION` environment to `/tmp` or 
      > some other non-NFS location. 

      While you should at least skim nix-portable's setup instructions, you'll probably end up doing something like this:
      ```console
      $ USERBIN="${XDG_BIN_HOME:-$HOME/.local/bin}"
      $ wget 'https://github.com/DavHau/nix-portable/releases/download/v010/nix-portable' -O "$USERBIN/nix-portable"
      ...
      $ chmod u+x "$USERBIN/nix-portable"
      ...
      $ ln -sf "$USERBIN/nix-portable" "$USERBIN/nix"
      ...
      ```
      Now if everything is setup properly, you should be able to see something like the following (don't worry if the version number is slightly different) if you run `nix --version`:
      ```
      $ nix --version
      nix (Nix) 2.20.6
      ```

2. Clone the flexflow-train repository (or, if you'd prefer, follow the alternative setup instructions in the [ff-dev](#ff-dev) section)

```console
$ FF_DIR="$HOME/flexflow-train" # or wherever else you want to put the repository
$ git clone --recursive git@github.com:flexflow/flexflow-train.git "$FF_DIR"
...
```

3. Enter the nix-provided development environment

```console
$ cd "$FF_DIR"
$ nix develop . --accept-flake-config
```

4. Build and run the tests

```console
(ff) $ proj cmake
...
(ff) $ proj test
...
```
If everything is correctly configured, you should see a bunch of build messages followed by something like
```console
(ff) $ proj test
TODO
```

If you don't, or you see any tests failing, please double check that you have followed
the instructions above. If you have and are still encountering an issue, please [contact us] with a detailed description of your platform and the 
commands you have run.

### ff-dev

Many of the flexflow-train developers use an additional set of scripts called [ff-dev](https://github.com/lockshaw/ff-dev) 
to automate many common git operations associated with flexflow-train development. 

> [!NOTE]
> ff-dev is totally optional: if you feel comfortable working with git's CLI you are more than welcome to skip this part.

To use ff-dev, instead of cloning the flexflow-train repo directly, you'll instead clone ff-dev to `~/ff`:

```console
$ git clone --recursive git@github.com:lockshaw/ff-dev.git "$HOME/ff"
```

and then run the `ff-dev-init` command from within the nix environment provided by `ff-dev`:

```
$ cd ~/ff
$ nix develop . --accept-flake-config
...
$ ff-dev-init
...
```

> [!NOTE]
> The development environment provided by ff-dev is different than the environment provided 
> by flexflow-train. Whenever you are running any scripts from ff-dev, make sure that your 
> shell prompt begins with `(ff-dev)`. Whenever you are actually doing flexflow-train development,
> make sure that your shell prompt begins with `(ff)`.

As part of `ff-dev-init`, you'll likely need to add a github authentication token to allow `ff-dev` to
create and modify your fork of the flexflow-train repository. 
If this is necessary, you'll see a prompt saying something like 

```console
? What account do you want to log into?  [Use arrow keys to move, type to filter]
...
```
At this point, perform the following steps:

1. Select "GitHub.com"
2. Select "SSH"
3. Select "Yes"
4. Select "Paste an authentication token"
5. Now go to <https://github.com/settings/tokens> and click "Generate new token" in the top right-hand corner, in the dropdown that appears select "Generate new token (classic)"
6. You should see a text field called "Note". Enter a brief name to remind yourself what this key is for.
7. Under "Expiration" select "90 days"
8. Under "Select scopes" check the following check boxes: `repo`, `read:org`, and `admin:public_key`
9. Click "Generate token"
10. You should now see a key beginning with `ghp_`. Copy this, save it somewhere to your computer safe (if you lose it, github won't show it to you again)
11. Copy the key beginning with `ghp_` into the prompt "Paste your authentication token:" and hit enter.
12. You should now see a message that says "Logged in as \<your github username\>", followed by a bunch of output from git as it clones the FlexFlow repository.

Once these steps are completed, you should be able to `cd ~/ff/master` and resume the standard setup instructions from step 3 (i.e., entering the nix-provided development environment).
You can find more instructions for how to use ff-dev [here]().

## Code Organization

The bulk of the FlexFlow source code is stored in the following folders:

1. `lib`: 
2. `bin`:
3. `bindings`:
4. `docs`:
5. `cmake`:
6. `deps`:
7. `config`:

## Continuous Integration

We currently implement CI testing using Github Workflows. Each workflow is defined by its corresponding YAML file in the [.github/workflows](.github/workflows) folder of the repo. We currently have the following workflows:

1. `build.yml`: checks that the build & installation of FlexFlow succeed, using both the CMake and Makefile systems
2. `clang-format-check.yml`: ensures that the source code is properly formatted.
4. `gpu-ci.yml`: runs all the tests that require a GPU to run.
8. `shell-check.yml`: runs shellcheck on all bash scripts in the repo

## Contributing to FlexFlow

We want to make contributing to this project as easy and transparent as possible.

### Formatting
We use `clang-format` to format our C++ code. If you make changes to the code and the Clang format CI test is failing, you can lint your code by running: `./scripts/format.sh` from the main folder of this repo.

### Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.

### Contact Us

[Zulip](https://flexflow.zulipchat.com/join/mtiwtwttgggnivrkb6vlakbr/) <!-- guest link -->

### Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

### License

By contributing to FlexFlow, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
