import numpy as np
import os, torch
from alignment.align_test_utils import *

def convert_hf_filename_to_ff_filename(f, num_layers=12):
    if f.endswith(".lm_head.weight"):
        f_version = f"fwd_step_0_layers_{num_layers-1}_lm_head_shard_0_weight_0"
    elif f == "norm.weight":
        f_version = f"fwd_step_0_layers_{num_layers-1}_norm_shard_0_weight_0"
    else:
        f_version = "fwd_step_0_"
        if f.startswith("layers."):
            layernum = f.split("layers.")[1].split(".")[0]
            f_version += f"layers_{layernum}_"
        f_version += f.split(".weight")[0].replace(".base_layer", "").replace(".default", "")
        weight_index="0"
        if "lora_A" in f_version:
            weight_index="A"
        elif "lora_B" in f_version:
            weight_index="B"
        f_version = f_version.replace("lora_A", "lora").replace("lora_B", "lora")
        f_version += f"_shard_0_weight_{weight_index}"
    return f_version

def check_weights_alignment():
    print("-- Weights alignment --")
    files_list = os.listdir(hf_path)
    num_layers=12
    for f in sorted(files_list):
        if f.endswith(".weight"):
            if "self_attn" in f:
                continue
            f_version = convert_hf_filename_to_ff_filename(f, num_layers=num_layers)
            # print(f, f_version)
            hf_w_path = os.path.join(hf_path, f)
            ff_w_path = os.path.join(ff_path, f_version)
            assert(os.path.isfile(hf_w_path))
            assert(os.path.isfile(ff_w_path))
            # print("\t", os.path.isfile(hf_w_path), os.path.isfile(ff_w_path))
            # print("\t", ff_w_path)

            # check equivalence
            compare_tensors(hf_w_path, ff_w_path, tolerance=1e-5)

def check_fwd_pass(tot_num_layers = 12):
    print("-- FWD pass --")
    # Transfomer head
    hf_embed_input= f"{hf_path}/fwd_step_0_embed_tokens.input_0"
    ff_embed_input = f"{ff_path}/fwd_step_0_layers_0_embed_tokens_shard_0_input_0"
    compare_tensors(hf_embed_input, ff_embed_input)
    hf_embed_output = f"{hf_path}/fwd_step_0_embed_tokens.output_0"
    ff_embed_output = f"{ff_path}/fwd_step_0_layers_0_embed_tokens_shard_0_output_0"
    compare_tensors(hf_embed_output, ff_embed_output)

    # Transformers blocks
    for i in range(tot_num_layers):
        hf_input_ln_in = f"{hf_path}/fwd_step_0_layers.{i}.input_layernorm.input_0"
        ff_input_ln_in = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.input_layernorm_shard_0_input_0"
        if i > 0:
            ff_input_ln_in = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.input_layernorm_shard_0_output_0"
        compare_tensors(hf_input_ln_in, ff_input_ln_in, tolerance=1e-5)
        hf_input_ln_out = f"{hf_path}/fwd_step_0_layers.{i}.input_layernorm.output_0"
        ff_input_ln_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.input_layernorm_shard_0_output_0"
        if i > 0:
            ff_input_ln_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.input_layernorm_shard_0_output_1"
        compare_tensors(hf_input_ln_out, ff_input_ln_out, tolerance=1e-5)
        hf_attn_out = f"{hf_path}/fwd_step_0_layers.{i}.self_attn.o_proj.output_0"
        ff_attn_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.self_attn_shard_0_output_0"
        compare_tensors(hf_attn_out, ff_attn_out, tolerance=1e-5)
        hf_ffn_norm_out = f"{hf_path}/fwd_step_0_layers.{i}.post_attention_layernorm.output_0"
        ff_ffn_norm_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.post_attention_layernorm_shard_0_output_1"
        compare_tensors(hf_ffn_norm_out, ff_ffn_norm_out, tolerance=1e-5)
        # w1
        hf_gate_proj_out = f"{hf_path}/fwd_step_0_layers.{i}.mlp.gate_proj.output_0"
        ff_gate_proj_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.gate_proj_shard_0_output_0"
        compare_tensors(hf_gate_proj_out, ff_gate_proj_out, tolerance=1e-5)
        # w3
        hf_up_proj_out = f"{hf_path}/fwd_step_0_layers.{i}.mlp.up_proj.output_0" 
        ff_up_proj_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.up_proj_shard_0_output_0"
        compare_tensors(hf_up_proj_out, ff_up_proj_out, tolerance=1e-5)
        # w2
        hf_down_proj_in = f"{hf_path}/fwd_step_0_layers.{i}.mlp.down_proj.input_0"
        hf_down_proj_out = f"{hf_path}/fwd_step_0_layers.{i}.mlp.down_proj.output_0"
        ff_down_proj_in = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.down_proj_shard_0_input_0"
        ff_down_proj_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.down_proj_shard_0_output_0"
        compare_tensors(hf_down_proj_in, ff_down_proj_in)
        # compare_tensors(hf_down_proj_out, ff_down_proj_out)
        # LORA input
        hf_lora_A_in = f"{hf_path}/fwd_step_0_layers.{i}.mlp.down_proj.lora_A.default.input_0"
        ff_lora_A_in = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_input_0"
        compare_hf_tensors(hf_down_proj_in, hf_lora_A_in)
        compare_tensors(hf_lora_A_in, ff_lora_A_in)
        # LORA weights
        hf_lora_A_weight_fp = f"{hf_path}/layers.{i}.mlp.down_proj.lora_A.default.weight"
        ff_lora_A_weight_fp = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_weight_A"
        compare_tensors(hf_lora_A_weight_fp, ff_lora_A_weight_fp)
        hf_lora_B_weight_fp = f"{hf_path}/layers.{i}.mlp.down_proj.lora_B.default.weight"
        ff_lora_B_weight_fp = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_weight_B"
        compare_tensors(hf_lora_B_weight_fp, ff_lora_B_weight_fp)
        # LORA intermediate hf
        hf_lora_A_out = f"{hf_path}/fwd_step_0_layers.{i}.mlp.down_proj.lora_A.default.output_0"
        hf_lora_B_in = f"{hf_path}/fwd_step_0_layers.{i}.mlp.down_proj.lora_B.default.input_0"
        compare_hf_tensors(hf_lora_A_out, hf_lora_B_in)
        # LORA output
        hf_lora_out = f"{hf_path}/fwd_step_0_layers.{i}.mlp.down_proj.lora_B.default.output_0"
        ff_lora_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_output_0"
        # compare_tensors(hf_lora_out, ff_lora_out)
        # compare_flexflow_tensors(ff_down_proj_out, ff_lora_out)
        # compare_tensors(hf_down_proj_out, ff_lora_out)
        compare_tensors_difference(hf_lora_out, ff_lora_out, ff_down_proj_out)
        

    # After last layer only
    hf_norm_out = f"{hf_path}/fwd_step_0_norm.output_0"
    ff_norm_out = f"{ff_path}/fwd_step_0_layers_{tot_num_layers-1}_norm_shard_0_output_1"
    compare_tensors(hf_norm_out, ff_norm_out, tolerance=1e-5)
    hf_lm_head_out = f"{hf_path}/fwd_step_0_base_model.model.lm_head.output_0"
    ff_lm_head_out = f"{ff_path}/fwd_step_0_layers_{tot_num_layers-1}_lm_head_shard_0_output_0"
    compare_tensors(hf_lm_head_out, ff_lm_head_out, tolerance=1e-5)

def check_bwd_pass(tot_num_layers = 12):
    # ff_BWD_softmax_in = f"{ff_path}/model_0_bwd-step_0_layer-num_100_layer-name_Softmax_shard-id_0_input_0"
    print("-- LM head --")
    hf_BWD_lm_head_out = f"{hf_path}/bwd_step_0_base_model.model.lm_head.go_0"
    ff_BWD_lm_head_out = f"{ff_path}/bwd_step_0_layers_{tot_num_layers-1}_lm_head_shard_0_output_0"
    compare_tensors(hf_BWD_lm_head_out, ff_BWD_lm_head_out, tolerance=1e-5)
    # compare weights
    # hf_lm_head_weight = f"{hf_path}/base_model.model.lm_head.weight"
    # ff_lm_head_weight = f"{ff_path}/fwd_step_0_layers_{tot_num_layers-1}_output_shard_0_weight_0"
    # compare_tensors(hf_lm_head_weight, ff_lm_head_weight, tolerance=1e-5)
    hf_BWD_lm_head_in = f"{hf_path}/bwd_step_0_base_model.model.lm_head.gi_0"
    ff_BWD_lm_head_in = f"{ff_path}/bwd_step_0_layers_{tot_num_layers-1}_lm_head_shard_0_input_0"
    compare_tensors(hf_BWD_lm_head_in, ff_BWD_lm_head_in, tolerance=1e-5)
    # # Manually check the matmul
    # ff_tensor_out = np.loadtxt(ff_BWD_lm_head_out, delimiter=',')
    # ff_weight = np.loadtxt(ff_lm_head_weight, delimiter=',').reshape((4096,32000), order='F')
    # ff_tensor_out = ff_tensor_out[:32000*24].reshape((32000,24), order='F')
    # print(ff_tensor_out.shape)
    # print(ff_weight.shape)
    # print(np.matmul(ff_weight, ff_tensor_out))
    # compare_tensors(hf_BWD_lm_head_in, ff_BWD_lm_head_in)
    # ff_tensor = np.loadtxt(ff_tensor_filepath, delimiter=',')
    print("-- Final Norm --")
    hf_BWD_norm_out = f"{hf_path}/bwd_step_0_norm.go_0"
    ff_BWD_norm_out = f"{ff_path}/bwd_step_0_layers_{tot_num_layers-1}_norm_shard_0_output_0"
    compare_hf_tensors(hf_BWD_lm_head_in, hf_BWD_norm_out)
    compare_tensors(hf_BWD_norm_out, ff_BWD_norm_out)
    ff_BWD_norm_weight = f"{ff_path}/fwd_step_0_layers_{tot_num_layers-1}_norm_shard_0_weight_0"
    hf_FWD_norm_weight = f"{hf_path}/norm.weight"
    compare_tensors(hf_FWD_norm_weight, ff_BWD_norm_weight, tolerance=1e-5)
    hf_BWD_norm_in = f"{hf_path}/bwd_step_0_norm.gi_0"
    ff_BWD_norm_in = f"{ff_path}/bwd_step_0_layers_{tot_num_layers-1}_norm_shard_0_input_1"
    compare_tensors(hf_BWD_norm_in, ff_BWD_norm_in, tolerance=1e-5)

    print("-- Transformers blocks --")
    for i in range(tot_num_layers-1, -1, -1):
        # HuggingFace filepaths
        hf_BWD_norm_in = f"{hf_path}/bwd_step_0_norm.gi_0"
        hf_BWD_loraB_out = f"{hf_path}/bwd_step_0_layers.{i}.mlp.down_proj.lora_B.default.go_0"
        hf_BWD_loraB_in = f"{hf_path}/bwd_step_0_layers.{i}.mlp.down_proj.lora_B.default.gi_0"
        hf_BWD_loraA_out = f"{hf_path}/bwd_step_0_layers.{i}.mlp.down_proj.lora_A.default.go_0"
        hf_BWD_loraA_in = f"{hf_path}/bwd_step_0_layers.{i}.mlp.down_proj.lora_A.default.gi_0"
        hf_loraA_weight = f"{hf_path}/layers.{i}.mlp.down_proj.lora_A.default.weight"
        hf_loraB_weight = f"{hf_path}/layers.{i}.mlp.down_proj.lora_B.default.weight"
        hf_BWD_w2_out = f"{hf_path}/bwd_step_0_layers.{i}.mlp.down_proj.go_0"
        hf_BWD_w2_in = f"{hf_path}/bwd_step_0_layers.{i}.mlp.down_proj.gi_0"
        hf_w2_weight = f"{hf_path}/layers.{i}.mlp.down_proj.base_layer.weight"
        hf_BWD_w3_out = f"{hf_path}/bwd_step_0_layers.{i}.mlp.up_proj.go_0"
        hf_BWD_w3_in = f"{hf_path}/bwd_step_0_layers.{i}.mlp.up_proj.gi_0"
        hf_BWD_w1_out = f"{hf_path}/bwd_step_0_layers.{i}.mlp.gate_proj.go_0"
        hf_BWD_w1_in = f"{hf_path}/bwd_step_0_layers.{i}.mlp.gate_proj.gi_0"
        hf_BWD_act_fn_in = f"{hf_path}/bwd_step_0_layers.{i}.mlp.act_fn.gi_0"
        hf_BWD_ffn_norm_out = f"{hf_path}/bwd_step_0_layers.{i}.post_attention_layernorm.go_0"
        hf_BWD_ffn_norm_in = f"{hf_path}/bwd_step_0_layers.{i}.post_attention_layernorm.gi_0"
        hf_BWD_attn_out_out = f"{hf_path}/bwd_step_0_layers.{i}.self_attn.o_proj.go_0"
        hf_BWD_attn_q_in = f"{hf_path}/bwd_step_0_layers.11.self_attn.q_proj.gi_0"
        hf_FWD_w1_out = f"{hf_path}/fwd_step_0_layers.{i}.mlp.gate_proj.output_0"
        hf_FWD_w3_out = f"{hf_path}/fwd_step_0_layers.{i}.mlp.up_proj.output_0"
        hf_FWD_act_fn_out = f"{hf_path}/fwd_step_0_layers.{i}.mlp.act_fn.output_0"
        hf_BWD_attn_oproj_in = f"{hf_path}/bwd_step_0_layers.{i}.self_attn.o_proj.gi_0"
        hf_attn_qproj_weight = f"{hf_path}/layers.{i}.self_attn.q_proj.weight"
        hf_attn_kproj_weight = f"{hf_path}/layers.{i}.self_attn.k_proj.weight"
        hf_attn_vproj_weight = f"{hf_path}/layers.{i}.self_attn.v_proj.weight"
        hf_attn_oproj_weight = f"{hf_path}/layers.{i}.self_attn.o_proj.weight"
        
        # FlexFlow filepaths
        ff_BWD_w2_out = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.mlp.down_proj_shard_0_output_0"
        ff_BWD_w2_in = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.mlp.down_proj_shard_0_input_0"
        ff_BWD_w2_in_pre = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.mlp.down_proj_shard_0_pre_input_0"
        ff_w2_weight = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.down_proj_shard_0_weight_0"
        ff_BWD_ssm_out = f"{ff_path}/bwd_step_0_layers_{i}_SigmoidSiluMulti_shard_0_output_0"
        ff_BWD_ssm_in1 = f"{ff_path}/bwd_step_0_layers_{i}_SigmoidSiluMulti_shard_0_input_0"
        ff_BWD_ssm_in2 = f"{ff_path}/bwd_step_0_layers_{i}_SigmoidSiluMulti_shard_0_input_1"
        ff_BWD_w3_out = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.mlp.up_proj_shard_0_output_0"
        ff_BWD_w3_in = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.mlp.up_proj_shard_0_input_0"
        ff_BWD_lora_A_in = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_input_0"
        ff_BWD_lora_B_out = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_output_0"
        ff_lora_A_weight = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_weight_A"
        ff_lora_B_weight = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_weight_B"
        ff_BWD_w1_out = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.mlp.gate_proj_shard_0_output_0"
        ff_BWD_w1_in = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.mlp.gate_proj_shard_0_input_0"
        ff_BWD_w1_in_pre = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.mlp.gate_proj_shard_0_pre_input_0"
        ff_BWD_ffn_norm_in1 = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.post_attention_layernorm_shard_0_input_0"
        ff_BWD_ffn_norm_in2 = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.post_attention_layernorm_shard_0_input_1"
        ff_BWD_ffn_norm_out = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.post_attention_layernorm_shard_0_output_0"
        ff_BWD_attn_out = ff_path + f"/bwd_step_0_layers_{i}_layers.{i}.self_attn_shard_0_output_0"        
        ff_BWD_attn_o_proj_in = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.self_attn_shard_0_o_proj_in_grad"
        ff_attn_oproj_weight = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.self_attn_shard_0_weight_0"

        # HuggingFace checks
        print("\nHuggingface checks:")
        if i == tot_num_layers-1:
            compare_hf_tensors(hf_BWD_norm_in, hf_BWD_loraB_out)
            compare_hf_tensors(hf_BWD_norm_in, hf_BWD_w2_out)
        compare_hf_tensors(hf_BWD_loraB_out, hf_BWD_w2_out)
        compare_hf_tensors(hf_BWD_loraB_in, hf_BWD_loraA_out)

        compare_hf_tensors(hf_BWD_act_fn_in, hf_BWD_w1_out)
        check_hf_sum_tensors(hf_BWD_ffn_norm_out, hf_BWD_w1_in, hf_BWD_w3_in)
        if i == tot_num_layers-1:
            check_hf_sum_tensors(hf_BWD_attn_out_out, hf_BWD_ffn_norm_in, hf_BWD_norm_in)

        # FlexFlow checks
        print("\nFlexFlow checks:")
        compare_flexflow_tensors(ff_BWD_w2_out, ff_BWD_lora_B_out)
        compare_flexflow_tensors(ff_BWD_w2_in_pre, ff_BWD_lora_A_in)
        compare_flexflow_tensors(ff_BWD_w2_in, ff_BWD_ssm_out)
        compare_flexflow_tensors(ff_BWD_ssm_in2, ff_BWD_w3_out)
        compare_flexflow_tensors(ff_BWD_ssm_in1, ff_BWD_w1_out)
        # compare_flexflow_tensors(ff_BWD_w1_in, ff_BWD_ffn_norm_out)
        compare_flexflow_tensors(ff_BWD_w1_in_pre, ff_BWD_w3_in)
        # compare_flexflow_tensors(ff_BWD_ffn_norm_in1, ff_BWD_ffn_norm_in2, max_len=24*768)
        
        # HF-FlexFlow checks
        print("\nHuggingface-FlexFlow checks:")
        print("-- W2 --")
        compare_tensors(hf_BWD_w2_out, ff_BWD_w2_out, tolerance=1e-5)
        compare_tensors(hf_w2_weight, ff_w2_weight, tolerance=1e-5)
        
        print("-- Lora --")
        compare_tensors(hf_loraA_weight, ff_lora_A_weight, tolerance=1e-5)
        compare_tensors(hf_loraB_weight, ff_lora_B_weight, tolerance=1e-5)

        compare_tensors(hf_BWD_loraB_out, ff_BWD_lora_B_out)
        compare_tensors(hf_BWD_loraA_in, ff_BWD_lora_A_in)
        
        print("-- W2/W1/W3 --")
        compare_tensors(hf_BWD_w2_in, ff_BWD_ssm_out)
        compare_tensors(hf_BWD_w2_in, ff_BWD_w2_in)
        compare_tensors(hf_BWD_w1_out, ff_BWD_w1_out)
        compare_tensors_difference(hf_BWD_w1_in, ff_BWD_w1_in, ff_BWD_w1_in_pre)
        compare_tensors(hf_BWD_w3_out, ff_BWD_w3_out)
        compare_tensors(hf_BWD_w3_in, ff_BWD_w3_in)
        compare_tensors(hf_BWD_w1_out, ff_BWD_w1_out)
        
        print("-- Attention --")
        num_tokens = 24
        hidden_size = 768
        qProjSize = 64
        num_heads = 12
        # Check output
        compare_tensors(hf_BWD_attn_out_out, ff_BWD_attn_out)
        
        # Check weights
        ff_attn_weight_tensor = np.loadtxt(ff_attn_oproj_weight, delimiter=',')
        ff_attn_qproj_weight_tensor = ff_attn_weight_tensor[:hidden_size*qProjSize*num_heads].reshape((hidden_size,qProjSize*num_heads), order = 'F')
        ff_attn_kproj_weight_tensor = ff_attn_weight_tensor[hidden_size*qProjSize*num_heads:2*hidden_size*qProjSize*num_heads].reshape((hidden_size,qProjSize*num_heads), order = 'F')
        ff_attn_vproj_weight_tensor = ff_attn_weight_tensor[2*hidden_size*qProjSize*num_heads:3*hidden_size*qProjSize*num_heads].reshape((hidden_size,qProjSize*num_heads), order = 'F')
        ff_attn_oproj_weight_tensor = ff_attn_weight_tensor[3*hidden_size*qProjSize*num_heads:].reshape((qProjSize*num_heads,hidden_size), order='F')
        
        hf_attn_qproj_weight_tensor = torch.load(hf_attn_qproj_weight).T.detach().cpu().numpy()
        hf_attn_kproj_weight_tensor = torch.load(hf_attn_kproj_weight).T.detach().cpu().numpy()
        hf_attn_vproj_weight_tensor = torch.load(hf_attn_vproj_weight).T.detach().cpu().numpy()
        hf_attn_oproj_weight_tensor = torch.load(hf_attn_oproj_weight).T.detach().cpu().numpy()
        
        assert(np.allclose(ff_attn_qproj_weight_tensor, hf_attn_qproj_weight_tensor, atol=1e-5))
        assert(np.allclose(ff_attn_kproj_weight_tensor, hf_attn_kproj_weight_tensor, atol=1e-5))
        assert(np.allclose(ff_attn_vproj_weight_tensor, hf_attn_vproj_weight_tensor, atol=1e-5))
        assert(np.allclose(ff_attn_oproj_weight_tensor, hf_attn_oproj_weight_tensor, atol=1e-5))

        # Compare attn outproj grad in tensors
        compare_tensors(hf_BWD_attn_oproj_in, ff_BWD_attn_o_proj_in)

        # Compare vproj grads
        hf_vproj_grads = f"{hf_path}/bwd_step_0_layers.{i}.self_attn.v_proj.go_0"
        ff_vproj_grads = ff_path + f"/bwd_step_0_layers_{i}_layers.{i}.self_attn_shard_0_v_proj_in_grad"
        hf_vproj_grads = torch.load(hf_vproj_grads).squeeze().detach().cpu().numpy()
        ff_vproj_grads = np.loadtxt(ff_vproj_grads, delimiter=',').reshape((num_tokens, qProjSize*num_heads), order='F')
        compare_loaded_tensors(hf_vproj_grads, ff_vproj_grads)

        # Compare kproj grads
        ff_kproj = ff_path + f"/bwd_step_0_layers_{i}_layers.{i}.self_attn_shard_0_devkproj"
        ff_kproj = np.loadtxt(ff_kproj, delimiter=',').reshape((num_tokens, qProjSize, num_heads), order = 'F')
        hf_kproj_grads = f"{hf_path}/bwd_step_0_layers.{i}.self_attn.k_proj.go_0"
        hf_kproj_grads = torch.load(hf_kproj_grads).squeeze()
        reshaped_tensor = hf_kproj_grads.view(24, 12, 64).transpose(1, 2).contiguous().detach().cpu().numpy()
        assert(np.allclose(ff_kproj, reshaped_tensor, atol=1e-2))
        print("Ok!")

        # Compare qproj grads
        hf_qproj_grads = f"{hf_path}/bwd_step_0_layers.{i}.self_attn.q_proj.go_0"
        hf_qproj_grads = torch.load(hf_qproj_grads).squeeze()
        reshaped_tensor = hf_qproj_grads.view(24, 12, 64).transpose(1, 2).contiguous().detach().cpu().numpy()
        ff_qproj = ff_path + f"/bwd_step_0_layers_{i}_layers.{i}.self_attn_shard_0_devQKVPRojArray"
        ff_qproj = np.loadtxt(ff_qproj, delimiter=',').reshape((num_tokens, qProjSize, num_heads, 3), order = 'F')[:,:,:,0]
        assert(np.allclose(ff_qproj, reshaped_tensor, atol=1e-2))
        print("Ok!")

        # Compare attn grad input 
        hf_attn_in = f"{hf_path}/bwd_step_0_layers.{i}.input_layernorm.go_0"
        ff_attn_in = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.self_attn_shard_0_attn_final_grad_in"
        compare_tensors(hf_attn_in, ff_attn_in)

        # compare input layernorm
        print("-- Input LayerNorm --")
        if i > 0:
            ff_input_ln_out = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.input_layernorm_shard_0_output_1"
            ff_attn_operator_in = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.self_attn_shard_0_input_0"
            compare_flexflow_tensors(ff_attn_operator_in, ff_input_ln_out)
            hf_input_ln_in = f"{hf_path}/bwd_step_0_layers.{i}.input_layernorm.gi_0"
            ff_input_ln_in0 = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.input_layernorm_shard_0_input_0"
            ff_input_ln_in1 = f"{ff_path}/bwd_step_0_layers_{i}_layers.{i}.input_layernorm_shard_0_input_1"
            compare_flexflow_tensors(ff_input_ln_in0, ff_input_ln_in1)
            if i > 1:
                compare_tensors(hf_input_ln_in, ff_input_ln_in0)
        



if __name__ == "__main__":
    check_weights_alignment()
    check_fwd_pass()
    check_bwd_pass()