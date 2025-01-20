# Developers Guide

## Setup

> [!NOTE]
> If you are developing on Stanford's sapling cluster, instead see the instructions [here](./docs/SAPLING.md).
> If you don't know what this means, you're not using sapling so you should just continue reading.

1. FlexFlow Train uses [nix](https://nix.dev/manual/nix/2.24/) to manage dependencies and the development environment. 
   There exist a number of ways to install nix, but we recommend one of the following:

   1. If you have root permissions: [DeterminateSystems/nix-installer](https://github.com/DeterminateSystems/nix-installer)

   2. If you don't have root permissions: [DavHau/nix-portable](https://github.com/DavHau/nix-portable). 
      Note that nix-portable does not work particularly well if the nix store is in NFS[^1] or other distributed file systems, 
      so if you are running on an HPC cluster where the home directory is mounted via a distributed file system we recommend setting the 
      `NP_LOCATION` environment to `/tmp` or some other non-NFS location. 

      While you should at least skim nix-portable's setup instructions, you'll probably end up doing something like this:
      ```
      $ USERBIN="${XDG_BIN_HOME:-$HOME/.local/bin}"
      $ wget 'https://github.com/DavHau/nix-portable/releases/download/v010/nix-portable' -O "$USERBIN/nix-portable"
      ...
      $ chmod u+x "$USERBIN/nix-portable"
      ...
      $ ln -sf "$USERBIN/nix-portable" "$USERBIN/nix"
      ...
      $ echo 'export PATH=$USERBIN:$PATH' >> ~/.bashrc
      ...
      ```
      Now if everything is setup properly, you should be able to see something like the following (don't worry if the version number is slightly different) if you run `nix --version`:
      ```
      $ nix --version
      nix (Nix) 2.20.6
      ```

[^1]: [Network File System](https://en.wikipedia.org/wiki/Network_File_System)

2. Clone the FlexFlow Train repository (or, if you'd prefer, follow the alternative setup instructions in the [ff-dev](#ff-dev-optional) section)

```
$ FF_DIR="$HOME/flexflow-train" # or wherever else you want to put the repository
$ git clone --recursive git@github.com:flexflow/flexflow-train.git "$FF_DIR"
...
```

3. Enter the nix-provided `default` development environment[^2]

[^2]: aka "dev shell"

```
$ cd "$FF_DIR"
$ nix develop --accept-flake-config
```

4. Build and run the non-GPU-required tests (systems that have access to CUDA GPUs can also run the GPU-mandatory tests by following the instructions [here](#gpu-setup))

```
(ff) $ proj cmake
...
(ff) $ proj test --skip-gpu-tests
...
```
If everything is correctly configured, you should see a bunch of build messages followed by something like
```
(ff) $ proj test --skip-gpu-tests
421/421 Test #441: get_transformer_computation_graph
100% tests passed, 0 tests failed out of 421

Label Time Summary:
compiler-tests                  =   6.13 sec*proc (19 tests)
local-execution-tests           =   0.13 sec*proc (3 tests)
models-tests                    =   0.05 sec*proc (4 tests)
op-attrs-tests                  =   0.48 sec*proc (59 tests)
pcg-tests                       =   0.33 sec*proc (33 tests)
substitution-generator-tests    =   0.06 sec*proc (2 tests)
substitutions-tests             =   0.10 sec*proc (9 tests)
utils-tests                     =   1.20 sec*proc (293 tests)

Total Test time (real) =   8.64 sec
```

If you don't, or if you see any tests failing, please double check that you have followed the instructions above. 
If you have and are still encountering an issue, please [contact us](#contact-us) with a detailed description of your platform and the commands you have run.

### GPU setup

If you are developing on a machine with one or more CUDA GPUs, you can also run the tests that require a GPU by entering the `gpu` devshell instead of the `default` devshell:
```
$ NIXPKGS_ALLOW_UNFREE=1 nix develop .#gpu --accept-flake-config --impure
```
and then running
```
(ff) $ proj test
...
```
You should see the additional GPU tests run. If you instead see a message like 

> `Error: ... Pass --skip-gpu-tests to skip running tests that require a GPU`

Double check that you are correctly in the `gpu` devshell, not the `default` devshell. 
If you've confirmed that you are in the correct devshell and are still encountering issues, [contact us](#contact-us) 
with a detailed description of your platform and the commands you have run.

### ff-dev (optional)

Many of the FlexFlow Train developers use an additional set of scripts called [ff-dev](https://github.com/lockshaw/ff-dev) 
to automate many common git operations associated with FlexFlow Train development. 

To setup ff-dev, run TODO (tracked in [#1573](https://github.com/flexflow/flexflow-train/issues/1573)).

<!--
To use ff-dev, instead of cloning the FlexFlow Train repo directly, you'll instead clone ff-dev to `~/ff`:

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
> by FlexFlow Train. Whenever you are running any scripts from ff-dev, make sure that your 
> shell prompt begins with `(ff-dev)`. Whenever you are actually doing FlexFlow Train development,
> make sure that your shell prompt begins with `(ff)`.

As part of `ff-dev-init`, you'll likely need to add a github authentication token to allow `ff-dev` to
create and modify your fork of the FlexFlow Train repository. 
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
-->

### nix-direnv (optional)

If you installed nix system-wide (e.g., using [DeterminateSystems/nix-installer](https://github.com/DeterminateSystems/nix-installer)), 
you can use [direnv](https://direnv.net/) to automatically enter the FlexFlow Train development environment when you `cd` into the repository, rather
than having to manually run `nix develop`.
[direnv](https://direnv.net) will also automatically exit the environment when you `cd` out of the repository, and (if configured using [nix-direnv](https://github.com/nix-community/nix-direnv)) will even automatically reload the environment if the `flake.nix` file changes.
You can find the installation instructions for direnv [here](https://direnv.net/docs/installation.html), and if you would like automatic environment reloading you can also install nix-direnv using the instructions [here](https://github.com/nix-community/nix-direnv?tab=readme-ov-file#installation).

Once you have direnv (and optionally nix-direnv) installed, cd into the root of your cloned FlexFlow Train repository and run
```
$ echo 'use flake . --accept-flake-config' > .envrc
```
You should see a message that the `.envrc` file you just created is blocked. 
Run the command shown in the error message (i.e., `direnv allow`), and direnv should automatically place you in the environment.
For more information on using direnv with nix, see [here](https://github.com/direnv/direnv/wiki/Nix).

## Building, Testing, etc.

Most operations you'll want to perform while developing FlexFlow Train are provided through a small python utility called [proj](https://github.com/lockshaw/proj). 
`proj` is automatically pulled in by nix when you enter the dev shell, so you should be able to run 
```
(ff) $ proj -h
```
and see the full list of operations that `proj` supports.
`proj` commands can be run from anywhere in the repository (i.e., they do not have to be run from the root).
To help you get started, however, a list of common command invocations is included here:

- To build FlexFlow Train:
  ```
  (ff) $ proj build
  ```
- To build and run FlexFlow Train tests (without a GPU):
  ```
  (ff) $ proj test --skip-gpu-tests
  ```
- To build and run FlexFlow Train tests (with a GPU):
  ```
  (ff) $ proj test
  ```
- To regenerate CMake files (necessary anytime you switch branches or modify the CMake source. If you're ever running into weird build issues, try running this and see if it fixes things):
  ```
  (ff) $ proj cmake
  ```
- To format all of the FlexFlow Train sources files: 
  ```
  (ff) $ proj format
  ```
- To build the FlexFlow Train Doxygen docs:
  ```
  (ff) $ proj doxygen
  ```
  You can also add the `--browser` command to automatically open the built docs in your default browser if you are working on your local machine.

## Code Organization

The bulk of the FlexFlow source code is stored in the following folders:

1. `lib`: The C++ code that makes up FlexFlow's core, split up into a number of libraries. You can find a description of each library [here](./lib/README.md).
2. `bin`: Command-line interfaces for FlexFlow and associated tools (all in C++). Generally, these are just thin wrappers that parse command-line arguments and then call out to functions defined in `lib` for the actual processing/logic. You can find a description of each binary [here](./bin/README.md).
3. `bindings`: Python (or any additional languages added in the future) bindings for FlexFlow Train
4. `docs`: Config files for documentation generators and code for generating diagrams. The actual documentation itself is included in the source directories/files as either `.md` files or inline in the language's documentation syntax (i.e., [Doxygen](https://www.doxygen.nl/manual/index.html) for C++ and [Sphinx](https://www.sphinx-doc.org/en/master/) for Python).
5. `cmake`: CMake configuration for building FlexFlow Train. Note that unless you're modifying the build configuration (i.e., adding a library, additional dependencies, etc.), you generally should use [proj](#building-testing-etc) instead of interacting with CMake directly. 
6. `deps`: Third-party dependencies included as submodules. Note that since FlexFlow Train moved to [nix](https://nix.dev/manual/nix/2.24/) for managing dependencies many (but not all) of these are used in the default configuration.

## Continuous Integration

We currently implement CI testing using Github Workflows. Each workflow is defined by its corresponding YAML file in the [.github/workflows](.github/workflows) folder of the repo. We currently have the following workflows:

1. [`tests`](./.github/workflows/per-lib-check.yml): Builds and runs GPU and non-GPU unit tests for all of the code under `lib` and `bin`. Also uploads coverage numbers to [codecov.io](https://app.codecov.io/gh/flexflow/flexflow-train).
2. [`clang-format-check.yml`](./.github/workflows/clang-format-check.yml): ensures that the source code is properly formatted using `clang-format`. To format your code locally, run `proj format` (see [here](#building-testing-etc) for more information on `proj`).
4. [`shell-check.yml`](./.github/workflows/shell-check.yml): runs shellcheck on all bash scripts in the repo.

GPU machines for CI are managed using [runs-on](https://runs-on.com/).

## Contributing to FlexFlow

We actively welcome your pull requests. Note that we may already be working on the feature/fix you're looking for, so we suggest searching through the [open issues](https://github.com/flexflow/flexflow-train/issues), [open PRs](https://github.com/flexflow/flexflow-train/pulls), and [contacting us](#contact-us) to make sure you're not duplicating existing effort!

The steps for getting changes merged into FlexFlow are relatively standard:

1. [Fork the repo](https://github.com/flexflow/flexflow-train/fork) and either create a new branch based on `master`, or just modify `master` directly.
2. If you've added code that should be tested, add tests. The process for adding tests for code under `lib` is documented [here](./lib/README.md#tests). Adding tests for other parts of the code is currently undocumented, so you will [contact us](#contact-us) for information on how to do it.
3. Ensure the code builds (i.e., run `proj build`).
4. Ensure the test suite passes (i.e., run `proj test`).
5. Format the code (i.e., run `proj format`).
6. Create a new PR from your modified branch to the `master` branch in FlexFlow Train. 
   Provide a brief description of the changes you've made and link any related/closed issues.

Code review is done using [Reviewable](https://reviewable.io/).
If you haven't used Reviewable before, please read through (or at least skim) the ["Reviews" section](https://docs.reviewable.io/reviews.html) of the Reviewable documentation.

## Contact Us

Either [create an issue](https://github.com/flexflow/flexflow-train/issues/new) or join the FlexFlow [Zulip](https://flexflow.zulipchat.com/join/mtiwtwttgggnivrkb6vlakbr/) instance. 
For any reported bugs, please ensure that your description clear and has sufficient information for us to reproduce the issue.

## License

By contributing to FlexFlow Train, you agree that your contributions will be licensed
under the [LICENSE](./LICENSE) file in the root directory of this source tree.
