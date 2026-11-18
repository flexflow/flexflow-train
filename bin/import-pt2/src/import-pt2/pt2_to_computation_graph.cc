#include "import-pt2/pt2_to_computation_graph.h"
#include "op-attrs/tensor_dims.h"
#include "pcg/computation_graph_builder.h"
#include <fmt/format.h>
#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace FlexFlow {

namespace {

using json = nlohmann::json;
using Shape = std::vector<int64_t>;

// torch/_export/serde/schema.py: ScalarType
DataType datatype_from_torch_scalar_type(int scalar_type,
                                         std::string const &tensor_name) {
  switch (scalar_type) {
    case 4:
      return DataType::INT32;
    case 5:
      return DataType::INT64;
    case 6:
      return DataType::HALF;
    case 7:
      return DataType::FLOAT;
    case 8:
      return DataType::DOUBLE;
    case 12:
      return DataType::BOOL;
    default:
      throw Pt2ImportError(fmt::format(
          "tensor {}: unsupported torch dtype (ScalarType {})",
          tensor_name,
          scalar_type));
  }
}

std::string to_str(std::vector<int64_t> const &v) {
  std::string result = "[";
  for (size_t i = 0; i < v.size(); i++) {
    result += (i == 0 ? "" : ", ") + std::to_string(v.at(i));
  }
  return result + "]";
}

int64_t normalize_dim(int64_t dim, size_t rank) {
  return dim < 0 ? dim + static_cast<int64_t>(rank) : dim;
}

/**
 * @brief The arguments of one serialized node, with typed accessors.
 */
struct NodeArgs {
  std::string node_name;
  std::string target;
  std::map<std::string, json> args;

  [[noreturn]] void fail(std::string const &msg) const {
    throw Pt2ImportError(
        fmt::format("node {} ({}): {}", this->node_name, this->target, msg));
  }

  bool has(std::string const &name) const {
    auto it = this->args.find(name);
    return it != this->args.end() && !it->second.contains("as_none");
  }

  json const &raw(std::string const &name) const {
    auto it = this->args.find(name);
    if (it == this->args.end()) {
      this->fail(fmt::format("missing argument '{}'", name));
    }
    return it->second;
  }

  bool is_tensor(std::string const &name) const {
    return this->has(name) && this->raw(name).contains("as_tensor");
  }

  std::string tensor(std::string const &name) const {
    json const &a = this->raw(name);
    if (!a.contains("as_tensor")) {
      this->fail(fmt::format("argument '{}' is not a tensor: {}", name, a.dump()));
    }
    return a.at("as_tensor").at("name").get<std::string>();
  }

  std::optional<std::string> optional_tensor(std::string const &name) const {
    if (!this->has(name)) {
      return std::nullopt;
    }
    return this->tensor(name);
  }

  std::vector<std::string> tensors(std::string const &name) const {
    json const &a = this->raw(name);
    if (!a.contains("as_tensors")) {
      this->fail(fmt::format("argument '{}' is not a tensor list", name));
    }
    std::vector<std::string> result;
    for (json const &t : a.at("as_tensors")) {
      result.push_back(t.at("name").get<std::string>());
    }
    return result;
  }

  int64_t get_int(std::string const &name,
                  std::optional<int64_t> default_value = std::nullopt) const {
    if (!this->has(name)) {
      if (default_value.has_value()) {
        return *default_value;
      }
      this->fail(fmt::format("missing int argument '{}'", name));
    }
    json const &a = this->raw(name);
    if (!a.contains("as_int")) {
      this->fail(fmt::format("argument '{}' is not a constant int: {}",
                             name,
                             a.dump()));
    }
    return a.at("as_int").get<int64_t>();
  }

  std::vector<int64_t>
      get_ints(std::string const &name,
               std::optional<std::vector<int64_t>> default_value =
                   std::nullopt) const {
    if (!this->has(name)) {
      if (default_value.has_value()) {
        return *default_value;
      }
      this->fail(fmt::format("missing int list argument '{}'", name));
    }
    json const &a = this->raw(name);
    if (!a.contains("as_ints")) {
      this->fail(fmt::format("argument '{}' is not a constant int list: {}",
                             name,
                             a.dump()));
    }
    return a.at("as_ints").get<std::vector<int64_t>>();
  }

  double get_float(std::string const &name,
                   std::optional<double> default_value = std::nullopt) const {
    if (!this->has(name)) {
      if (default_value.has_value()) {
        return *default_value;
      }
      this->fail(fmt::format("missing float argument '{}'", name));
    }
    json const &a = this->raw(name);
    if (a.contains("as_float")) {
      return a.at("as_float").get<double>();
    }
    if (a.contains("as_int")) {
      return static_cast<double>(a.at("as_int").get<int64_t>());
    }
    this->fail(
        fmt::format("argument '{}' is not a constant number: {}", name, a.dump()));
  }

  bool get_bool(std::string const &name,
                std::optional<bool> default_value = std::nullopt) const {
    if (!this->has(name)) {
      if (default_value.has_value()) {
        return *default_value;
      }
      this->fail(fmt::format("missing bool argument '{}'", name));
    }
    json const &a = this->raw(name);
    if (!a.contains("as_bool")) {
      this->fail(fmt::format("argument '{}' is not a constant bool", name));
    }
    return a.at("as_bool").get<bool>();
  }

  std::optional<std::string> get_string(std::string const &name) const {
    if (!this->has(name)) {
      return std::nullopt;
    }
    json const &a = this->raw(name);
    if (!a.contains("as_string")) {
      this->fail(fmt::format("argument '{}' is not a constant string", name));
    }
    return a.at("as_string").get<std::string>();
  }

  // Expands 1-element int lists (e.g. stride=[2]) to 2 elements.
  std::pair<int64_t, int64_t>
      get_int_pair(std::string const &name,
                   std::optional<std::vector<int64_t>> default_value =
                       std::nullopt) const {
    std::vector<int64_t> v = this->get_ints(name, default_value);
    if (v.size() == 1) {
      return {v.at(0), v.at(0)};
    }
    if (v.size() != 2) {
      this->fail(fmt::format("argument '{}' should have 1 or 2 elements, got {}",
                             name,
                             to_str(v)));
    }
    return {v.at(0), v.at(1)};
  }
};

class Pt2GraphImporter {
public:
  explicit Pt2GraphImporter(json const &model)
      : graph(model.at("graph_module").at("graph")),
        signature(model.at("graph_module").at("signature")) {}

  ComputationGraph run() {
    this->load_tensor_values();
    this->load_signature_inputs();
    for (json const &node : this->graph.at("nodes")) {
      this->import_node(node);
    }
    for (json const &spec : this->signature.at("output_specs")) {
      if (!spec.contains("user_output")) {
        throw Pt2ImportError(fmt::format(
            "unsupported output spec (only user outputs are supported, e.g. "
            "no buffer mutations): {}",
            spec.dump()));
      }
      json const &arg = spec.at("user_output").at("arg");
      if (!arg.contains("as_tensor")) {
        throw Pt2ImportError(
            fmt::format("unsupported non-tensor graph output: {}", arg.dump()));
      }
      std::string name = arg.at("as_tensor").at("name").get<std::string>();
      if (this->tensors.count(name) == 0) {
        throw Pt2ImportError(
            fmt::format("graph output {} was not produced by the import", name));
      }
    }
    return this->cgb.computation_graph;
  }

private:
  json const &graph;
  json const &signature;
  ComputationGraphBuilder cgb;
  std::map<std::string, Shape> shapes;
  std::map<std::string, int> dtypes;
  std::map<std::string, bool> requires_grad;
  std::set<std::string> constants; // parameters, buffers, lifted constants
  std::map<std::string, tensor_guid_t> tensors;

  void load_tensor_values() {
    for (auto const &[name, meta] : this->graph.at("tensor_values").items()) {
      Shape shape;
      for (json const &s : meta.at("sizes")) {
        if (!s.contains("as_int")) {
          throw Pt2ImportError(fmt::format(
              "tensor {} has a symbolic (dynamic) size; only static shapes "
              "are supported. Export without dynamic_shapes.",
              name));
        }
        shape.push_back(s.at("as_int").get<int64_t>());
      }
      this->shapes.emplace(name, shape);
      this->dtypes.emplace(name, meta.at("dtype").get<int>());
      this->requires_grad.emplace(name, meta.value("requires_grad", false));
    }
  }

  void load_signature_inputs() {
    for (json const &spec : this->signature.at("input_specs")) {
      auto const &[kind, body] = *spec.items().begin();
      if (kind == "user_input") {
        json const &arg = body.at("arg");
        if (!arg.contains("as_tensor")) {
          throw Pt2ImportError(fmt::format(
              "unsupported non-tensor user input: {}", arg.dump()));
        }
        std::string name = arg.at("as_tensor").at("name").get<std::string>();
        this->create_input(name);
      } else if (kind == "parameter" || kind == "buffer" ||
                 kind == "tensor_constant") {
        this->constants.insert(body.at("arg").at("name").get<std::string>());
      } else {
        throw Pt2ImportError(
            fmt::format("unsupported input spec kind '{}'", kind));
      }
    }
  }

  void create_input(std::string const &name) {
    TensorShape shape = TensorShape{
        TensorDims{FFOrdered<positive_int>{this->positive_dims(name)}},
        datatype_from_torch_scalar_type(this->dtypes.at(name), name),
    };
    CreateGrad create_grad =
        this->requires_grad.at(name) ? CreateGrad::YES : CreateGrad::NO;
    this->tensors.emplace(name, this->cgb.create_input(shape, create_grad, name));
  }

  Shape const &shape_of(std::string const &name) const {
    auto it = this->shapes.find(name);
    if (it == this->shapes.end()) {
      throw Pt2ImportError(fmt::format("no tensor_values entry for {}", name));
    }
    return it->second;
  }

  std::vector<positive_int> positive_dims(std::string const &name) const {
    Shape const &shape = this->shape_of(name);
    if (shape.empty()) {
      throw Pt2ImportError(fmt::format(
          "tensor {} is 0-dimensional; FlexFlow tensors need at least 1 dim",
          name));
    }
    std::vector<positive_int> dims;
    for (int64_t d : shape) {
      if (d <= 0) {
        throw Pt2ImportError(
            fmt::format("tensor {} has a non-positive dim: {}", name, to_str(shape)));
      }
      dims.push_back(positive_int{static_cast<int>(d)});
    }
    return dims;
  }

  tensor_guid_t data_tensor(NodeArgs const &n, std::string const &name) const {
    if (this->constants.count(name)) {
      n.fail(fmt::format("parameter/buffer {} is used as a data input, which "
                         "is not supported for this op",
                         name));
    }
    auto it = this->tensors.find(name);
    if (it == this->tensors.end()) {
      n.fail(fmt::format("input tensor {} was not produced by a supported op",
                         name));
    }
    return it->second;
  }

  tensor_guid_t data_arg(NodeArgs const &n, std::string const &arg) const {
    return this->data_tensor(n, n.tensor(arg));
  }

  std::string const &constant_arg(NodeArgs const &n,
                                  std::string const &name) const {
    if (!this->constants.count(name)) {
      n.fail(fmt::format("expected {} to be a parameter or buffer", name));
    }
    return name;
  }

  static positive_int pos(NodeArgs const &n, int64_t v, std::string const &what) {
    if (v <= 0) {
      n.fail(fmt::format("{} must be positive, got {}", what, v));
    }
    return positive_int{static_cast<int>(v)};
  }

  static nonnegative_int nonneg(NodeArgs const &n,
                                int64_t v,
                                std::string const &what) {
    if (v < 0) {
      n.fail(fmt::format("{} must be non-negative, got {}", what, v));
    }
    return nonnegative_int{static_cast<int>(v)};
  }

  static std::vector<std::string> output_names(NodeArgs const &n,
                                               json const &node) {
    std::vector<std::string> names;
    for (json const &o : node.at("outputs")) {
      if (o.contains("as_tensor")) {
        names.push_back(o.at("as_tensor").at("name").get<std::string>());
      } else if (o.contains("as_tensors")) {
        for (json const &t : o.at("as_tensors")) {
          names.push_back(t.at("name").get<std::string>());
        }
      } else if (o.contains("as_none")) {
        continue;
      } else {
        n.fail(fmt::format("unsupported output kind: {}", o.dump()));
      }
    }
    return names;
  }

  void import_node(json const &node) {
    NodeArgs n;
    n.node_name = node.at("name").get<std::string>();
    n.target = node.at("target").get<std::string>();
    for (json const &input : node.at("inputs")) {
      n.args.emplace(input.at("name").get<std::string>(), input.at("arg"));
    }
    std::vector<std::string> outs = output_names(n, node);

    std::vector<std::optional<tensor_guid_t>> results =
        this->translate(n, outs);

    if (results.size() != outs.size()) {
      n.fail(fmt::format("importer produced {} outputs, node has {}",
                         results.size(),
                         outs.size()));
    }
    for (size_t i = 0; i < outs.size(); i++) {
      if (!results.at(i).has_value()) {
        continue; // unused auxiliary output (e.g. batch norm saved stats)
      }
      tensor_guid_t t = results.at(i).value();
      this->check_shape(n, outs.at(i), t);
      this->tensors.emplace(outs.at(i), t);
    }
  }

  void check_shape(NodeArgs const &n,
                   std::string const &name,
                   tensor_guid_t const &t) const {
    // Keep the TensorShape alive: ff_ordered() returns a reference into it.
    TensorShape ff_tensor_shape = this->cgb.get_shape(t);
    Shape ff_shape;
    for (positive_int d : ff_ordered(ff_tensor_shape.dims)) {
      ff_shape.push_back(d.int_from_positive_int());
    }
    Shape const &torch_shape = this->shape_of(name);
    if (ff_shape != torch_shape) {
      n.fail(fmt::format("output {}: FlexFlow inferred shape {} but the .pt2 "
                         "records {}",
                         name,
                         to_str(ff_shape),
                         to_str(torch_shape)));
    }
  }

  // Size of each output along `dim`; used for chunk/split.
  std::vector<positive_int> output_sizes_along(NodeArgs const &n,
                                               std::vector<std::string> const &outs,
                                               int64_t dim) const {
    std::vector<positive_int> sizes;
    for (std::string const &o : outs) {
      sizes.push_back(pos(n, this->shape_of(o).at(dim), "split size"));
    }
    return sizes;
  }

  using Results = std::vector<std::optional<tensor_guid_t>>;

  Results translate(NodeArgs const &n, std::vector<std::string> const &outs) {
    std::string const prefix = "torch.ops.aten.";
    if (n.target.rfind(prefix, 0) != 0) {
      n.fail("unsupported op (not an ATen op)");
    }
    std::string op = n.target.substr(prefix.size());
    std::optional<std::string> name = n.node_name;
    ComputationGraphBuilder &b = this->cgb;

    auto single = [](tensor_guid_t t) { return Results{t}; };

    // ---- no-ops and pass-throughs
    if (op == "_assert_tensor_metadata.default") {
      return {};
    }
    if (op == "detach.default" || op == "detach_.default" ||
        op == "clone.default" || op == "alias.default" ||
        op == "contiguous.default" || op == "lift_fresh_copy.default") {
      return single(this->data_arg(n, "self"));
    }
    if (op == "dropout.default") {
      if (n.get_bool("train")) {
        n.fail("dropout in training mode is not supported; export the model "
               "in eval() mode");
      }
      return single(this->data_arg(n, "input"));
    }
    if (op == "to.dtype" || op == "_to_copy.default") {
      std::string in = n.tensor("self");
      if (this->dtypes.at(in) != this->dtypes.at(outs.at(0))) {
        n.fail("dtype conversion is not supported yet");
      }
      return single(this->data_tensor(n, in));
    }

    // ---- convolution / normalization / linear
    if (op == "conv2d.default") {
      Shape const &w = this->shape_of(this->constant_arg(n, n.tensor("weight")));
      if (w.size() != 4) {
        n.fail(fmt::format("expected a 4D conv weight, got {}", to_str(w)));
      }
      auto [sh, sw] = n.get_int_pair("stride", std::vector<int64_t>{1});
      auto [ph, pw] = n.get_int_pair("padding", std::vector<int64_t>{0});
      auto [dh, dw] = n.get_int_pair("dilation", std::vector<int64_t>{1});
      if (dh != 1 || dw != 1) {
        n.fail("dilated convolution is not supported");
      }
      std::optional<std::string> bias = n.optional_tensor("bias");
      if (bias.has_value()) {
        this->constant_arg(n, *bias);
      }
      return single(b.conv2d(this->data_arg(n, "input"),
                             /*outChannels=*/pos(n, w.at(0), "out_channels"),
                             /*kernelH=*/pos(n, w.at(2), "kernel_h"),
                             /*kernelW=*/pos(n, w.at(3), "kernel_w"),
                             /*strideH=*/pos(n, sh, "stride_h"),
                             /*strideW=*/pos(n, sw, "stride_w"),
                             /*paddingH=*/nonneg(n, ph, "padding_h"),
                             /*paddingW=*/nonneg(n, pw, "padding_w"),
                             /*activation=*/std::nullopt,
                             /*groups=*/pos(n, n.get_int("groups", 1), "groups"),
                             /*use_bias=*/bias.has_value(),
                             /*kernel_initializer=*/std::nullopt,
                             /*bias_initializer=*/std::nullopt,
                             /*kernel_regularizer=*/std::nullopt,
                             name));
    }
    if (op == "batch_norm.default" ||
        op == "_native_batch_norm_legit_no_training.default") {
      bool has_weight = n.has("weight");
      if (has_weight != n.has("bias")) {
        n.fail("batch norm with only one of weight/bias is not supported");
      }
      float eps = static_cast<float>(n.get_float("eps"));
      float momentum = static_cast<float>(n.get_float("momentum"));
      tensor_guid_t out = b.batch_norm(this->data_arg(n, "input"),
                                       /*affine=*/has_weight,
                                       /*activation=*/std::nullopt,
                                       eps,
                                       momentum,
                                       name);
      Results r{out};
      // The decomposed form also returns (save_mean, save_rstd), unused at
      // inference.
      while (r.size() < outs.size()) {
        r.push_back(std::nullopt);
      }
      return r;
    }
    if (op == "linear.default") {
      Shape const &w = this->shape_of(this->constant_arg(n, n.tensor("weight")));
      if (w.size() != 2) {
        n.fail(fmt::format("expected a 2D linear weight, got {}", to_str(w)));
      }
      std::optional<std::string> bias = n.optional_tensor("bias");
      if (bias.has_value()) {
        this->constant_arg(n, *bias);
      }
      std::string in = n.tensor("input");
      return single(b.dense(
          this->data_tensor(n, in),
          /*outDim=*/pos(n, w.at(0), "out_features"),
          /*activation=*/std::nullopt,
          /*use_bias=*/bias.has_value(),
          /*data_type=*/datatype_from_torch_scalar_type(this->dtypes.at(in), in),
          /*projection_initializer=*/std::nullopt,
          /*bias_initializer=*/std::nullopt,
          name));
    }

    // ---- element-wise unary
    if (op == "silu.default" || op == "silu_.default") {
      return single(b.silu(this->data_arg(n, "self"), name));
    }
    if (op == "relu.default" || op == "relu_.default") {
      return single(b.relu(this->data_arg(n, "self"), name));
    }
    if (op == "sigmoid.default" || op == "sigmoid_.default") {
      return single(b.sigmoid(this->data_arg(n, "self"), name));
    }
    if (op == "tanh.default" || op == "tanh_.default") {
      return single(b.tanh(this->data_arg(n, "self"), name));
    }
    if (op == "gelu.default") {
      std::optional<std::string> approx = n.get_string("approximate");
      if (approx.has_value() && *approx != "none") {
        n.fail(fmt::format("gelu approximate='{}' is not supported", *approx));
      }
      return single(b.gelu(this->data_arg(n, "self"), name));
    }
    if (op == "exp.default") {
      return single(b.exp(this->data_arg(n, "self"), name));
    }
    if (op == "rsqrt.default") {
      return single(b.rsqrt(this->data_arg(n, "self"), name));
    }

    // ---- element-wise binary (tensor-tensor or tensor-scalar)
    if (op == "add.Tensor" || op == "sub.Tensor") {
      if (n.has("alpha") && n.get_float("alpha") != 1.0) {
        n.fail("alpha != 1 is not supported");
      }
    }
    if (op == "add.Tensor" || op == "sub.Tensor" || op == "mul.Tensor" ||
        op == "div.Tensor") {
      tensor_guid_t self = this->data_arg(n, "self");
      if (n.is_tensor("other")) {
        tensor_guid_t other = this->data_arg(n, "other");
        if (op == "add.Tensor") {
          return single(b.add(self, other, name));
        } else if (op == "sub.Tensor") {
          return single(b.subtract(self, other, name));
        } else if (op == "mul.Tensor") {
          return single(b.multiply(self, other, name));
        } else {
          return single(b.divide(self, other, name));
        }
      }
      float s = static_cast<float>(n.get_float("other"));
      if (op == "add.Tensor") {
        return single(b.scalar_add(self, s, name));
      } else if (op == "sub.Tensor") {
        return single(b.scalar_sub(self, s, name));
      } else if (op == "mul.Tensor") {
        return single(b.scalar_multiply(self, s, name));
      } else {
        return single(b.scalar_truediv(self, s, name));
      }
    }

    // ---- shape ops
    if (op == "cat.default") {
      std::vector<std::string> ins = n.tensors("tensors");
      std::vector<tensor_guid_t> ts;
      for (std::string const &t : ins) {
        ts.push_back(this->data_tensor(n, t));
      }
      int64_t dim = n.get_int("dim", 0);
      return single(b.concat(ts, relative_ff_dim_t{static_cast<int>(dim)}, name));
    }
    if (op == "chunk.default" || op == "split.Tensor" ||
        op == "split_with_sizes.default") {
      std::string in = n.tensor("self");
      int64_t dim = normalize_dim(n.get_int("dim", 0), this->shape_of(in).size());
      tensor_guid_t input = this->data_tensor(n, in);
      if (outs.size() == 1) {
        return single(input);
      }
      std::vector<tensor_guid_t> parts =
          b.split(input,
                  this->output_sizes_along(n, outs, dim),
                  relative_ff_dim_t{static_cast<int>(dim)},
                  name);
      return Results(parts.begin(), parts.end());
    }
    if (op == "view.default" || op == "reshape.default" ||
        op == "_unsafe_view.default" || op == "flatten.using_ints" ||
        op == "unsqueeze.default" || op == "squeeze.dim" ||
        op == "squeeze.dims") {
      // The target shape (with -1 resolved) is the recorded output shape.
      return single(b.reshape(
          this->data_arg(n, "self"), this->positive_dims(outs.at(0)), name));
    }
    if (op == "transpose.int" || op == "permute.default") {
      std::string in = n.tensor("self");
      size_t rank = this->shape_of(in).size();
      std::vector<int64_t> perm(rank);
      for (size_t i = 0; i < rank; i++) {
        perm.at(i) = static_cast<int64_t>(i);
      }
      if (op == "transpose.int") {
        int64_t d0 = normalize_dim(n.get_int("dim0"), rank);
        int64_t d1 = normalize_dim(n.get_int("dim1"), rank);
        std::swap(perm.at(d0), perm.at(d1));
      } else {
        std::vector<int64_t> dims = n.get_ints("dims");
        if (dims.size() != rank) {
          n.fail("permute dims has the wrong length");
        }
        for (size_t i = 0; i < rank; i++) {
          perm.at(i) = normalize_dim(dims.at(i), rank);
        }
      }
      std::vector<nonnegative_int> ff_perm;
      for (int64_t p : perm) {
        ff_perm.push_back(nonneg(n, p, "permutation entry"));
      }
      return single(b.transpose(this->data_tensor(n, in), ff_perm, name));
    }

    // ---- pooling / resampling
    if (op == "max_pool2d.default") {
      auto [kh, kw] = n.get_int_pair("kernel_size");
      // PyTorch: an empty stride means stride = kernel_size.
      auto [sh, sw] = n.get_ints("stride", std::vector<int64_t>{}).empty()
                          ? std::pair<int64_t, int64_t>{kh, kw}
                          : n.get_int_pair("stride");
      auto [ph, pw] = n.get_int_pair("padding", std::vector<int64_t>{0});
      auto [dh, dw] = n.get_int_pair("dilation", std::vector<int64_t>{1});
      if (dh != 1 || dw != 1) {
        n.fail("dilated max_pool2d is not supported");
      }
      if (n.get_bool("ceil_mode", false)) {
        n.fail("max_pool2d ceil_mode=True is not supported");
      }
      return single(b.pool2d(this->data_arg(n, "self"),
                             pos(n, kh, "kernel_h"),
                             pos(n, kw, "kernel_w"),
                             pos(n, sh, "stride_h"),
                             pos(n, sw, "stride_w"),
                             nonneg(n, ph, "padding_h"),
                             nonneg(n, pw, "padding_w"),
                             PoolOp::MAX,
                             /*activation=*/std::nullopt,
                             name));
    }
    if (op == "adaptive_avg_pool2d.default") {
      auto [oh, ow] = n.get_int_pair("output_size");
      return single(b.adaptive_pool2d(this->data_arg(n, "self"),
                                      pos(n, oh, "output_h"),
                                      pos(n, ow, "output_w"),
                                      PoolOp::AVG,
                                      /*activation=*/std::nullopt,
                                      name));
    }
    if (op == "upsample_nearest2d.vec" || op == "upsample_nearest2d.default") {
      std::string in = n.tensor("input");
      Shape const &is = this->shape_of(in);
      Shape const &os = this->shape_of(outs.at(0));
      if (is.size() != 4 || os.size() != 4 || os.at(2) % is.at(2) != 0 ||
          os.at(3) % is.at(3) != 0 || os.at(2) / is.at(2) != os.at(3) / is.at(3) ||
          os.at(2) / is.at(2) < 2) {
        n.fail(fmt::format("only equal integer scale factors >= 2 are "
                           "supported (input {}, output {})",
                           to_str(is),
                           to_str(os)));
      }
      int factor = static_cast<int>(os.at(2) / is.at(2));
      return single(b.upsample(this->data_tensor(n, in),
                               int_ge_two{factor},
                               UpsampleMode::NEAREST,
                               name));
    }

    // ---- matmul / softmax
    if (op == "matmul.default" || op == "bmm.default") {
      std::string lhs = n.tensor("self");
      std::string rhs = n.is_tensor("other") ? n.tensor("other") : n.tensor("mat2");
      Shape const &ls = this->shape_of(lhs);
      Shape const &rs = this->shape_of(rhs);
      if (ls.size() < 3 || ls.size() != rs.size() ||
          !std::equal(ls.begin(), ls.end() - 2, rs.begin())) {
        n.fail(fmt::format("only batched matmul with equal batch dims is "
                           "supported (lhs {}, rhs {})",
                           to_str(ls),
                           to_str(rs)));
      }
      return single(b.batch_matmul(
          this->data_tensor(n, lhs), this->data_tensor(n, rhs), name));
    }
    if (op == "softmax.int" || op == "_softmax.default") {
      if (n.has("half_to_float") && n.get_bool("half_to_float")) {
        n.fail("half_to_float softmax is not supported");
      }
      if (n.has("dtype")) {
        n.fail("softmax with dtype= is not supported");
      }
      return single(b.softmax(this->data_arg(n, "self"),
                              relative_ff_dim_t{static_cast<int>(n.get_int("dim"))},
                              name));
    }

    n.fail("unsupported op");
  }
};

} // namespace

ComputationGraph
    computation_graph_from_pt2_model_json(nlohmann::json const &model) {
  try {
    return Pt2GraphImporter{model}.run();
  } catch (json::exception const &e) {
    throw Pt2ImportError(
        fmt::format("unexpected .pt2 model.json structure: {}", e.what()));
  }
}

} // namespace FlexFlow
