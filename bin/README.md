# bin

This directory contains command-line interfaces for FlexFlow Train and associated tools (all in C++). 
A short description of each is included below--more information can be found in the `README.md` files
in each of the corresponding directories (e.g., [here](./export-model-arch/README.md) for `export-model-arch`):

- `export-model-arch`: Exports the model computation graphs defined in the [models](../lib/models/) library as either JSON (for use outside of FlexFlow) or as DOT (for visualization). Can also optionally export the SP decompositions of the computation graphs.
- `substitution-to-dot`: Converts TASO-generated substitutions from the legacy JSON format ([example](../substitutions/graph_subst_3_v2.json)) into DOT for visualization.
- `protobuf-to-json`: Converts TASO-generated substitutions from the legacy protobuf format ([example](../substitutions/graph_subst_3_v2.pb)) to the legacy JSON format ([example](../substitutions/graph_subst_3_v2.json)). Will be removed in the future once the substitution generator is integrated natively into FlexFlow Train (tracked in [#351](https://github.com/flexflow/flexflow-train/issues/351)).
