# export-model-arch

A tool for exporting and visualizing the model computation graphs defined in [models](../lib/models).
To build and run `export-model-arch`, run the following commands from the root of the FlexFlow Train repository:
```console
$ proj cmake # if you haven't already
...
$ proj build
...
$ ./build/normal/bin/export-model-arch/export-model-arch -h
```
The above should print the help message for `export-model-arch`. A few example commands are also listed below:

- Export the `split_test` model in JSON (e.g., for processing outside of FlexFlow Train):
```console
$ ./build/normal/bin/export-model-arch/export-model-arch split_test
```

- Export the `split_test` model in JSON along with the SP decomposition of the model's computation graph:
```console
$ ./build/normal/bin/export-model-arch/export-model-arch --sp-decomposition split_test
```

- Export the `split_test` model as DOT (e.g., for visualization using a [local](https://github.com/jrfonseca/xdot.py) or [web-based](https://dreampuf.github.io/GraphvizOnline/) DOT viewer)
```console
$ ./build/normal/bin/export-model-arch/export-model-arch --dot split_test
```
