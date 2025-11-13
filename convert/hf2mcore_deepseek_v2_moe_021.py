# Copyright (c) 2025 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
os.environ['MEGATRON_DISABLE_CUDA_RNG_TRACKER'] = '1'

import re
import torch
from functools import partial
from collections import defaultdict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
if os.getenv("ACCELERATOR_BACKEND") == "musa":
    import musa_patch

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata
from megatron.training.utils import get_ltor_masks_and_position_ids

# path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
# sys.path.append(os.path.join(path_dir, "examples"))
# from deepseek_v2.pretrain_deepseek import model_provider
from megatron_patch.arguments import get_patch_args
print(f"{os.environ['PYTHONPATH']}")
from toolkits.model_checkpoints_convertor.utils import (
    save_state_dict,
    save_hfmodel
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

from contextlib import nullcontext
from typing import List, Optional, Tuple, Union
from megatron.core.transformer.spec_utils import import_module
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)

if os.environ.get("DEBUG", "0") == "1":
    import debugpy
    print("Waiting for debugger to attach...")
    debugpy.listen(('localhost', 5678))
    debugpy.wait_for_client()
    print("Debugger attached!")
        
def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    print(" args is ", args)
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        print(" args.yaml_cfg is not None")
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        print("core_transformer_config_from_args(args)")
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            # args.padded_vocab_size = 102400 ### ds v2 lite
            args.padded_vocab_size = 128256 ### 021-32B
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling
            )

    return model


def add_model_args(parser):

    parser.add_argument(
        "--target-tensor-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-pipeline-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-expert-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-decoder-first-pipeline-num-layers",
        type=int,
        default=None
    )

    parser.add_argument(
        "--hf-ckpt-path",
        type=str
    )

    parser.add_argument(
        "--save-safetensors",
        action='store_false',
    )

    return parser


def load_megatron_model(args):
    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.hf_ckpt_path + "/*config.json " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/*.py " + args.save)

    os.system("cp -rf " + args.hf_ckpt_path + "/*config.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/*.py " + args.load)

    model = model_provider()

    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size

    if args.tensor_model_parallel_size > 1:
        args.sequence_parallel = True

    model_path = args.load
    tracker_filename = get_checkpoint_tracker_filename(model_path)
    iteration, release = read_metadata(tracker_filename)
    q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
    group_per_split = args.num_attention_heads // args.tensor_model_parallel_size
    if args.num_experts is not None:
        if args.moe_grouped_gemm:
            pattern = r'weight(\d+)'
        else:
            pattern = r'local_experts\.(\d+)\.'        
        num_local_experts = args.num_experts // args.expert_model_parallel_size
    state_dict = {}
    mid_state = defaultdict(list)
    if (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size == 1 ###
    ):
        checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, None, None, None, None)
        state_dict = torch.load(checkpoint_name)['model']
    elif (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size > 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):
        for ep_rank in range(args.expert_model_parallel_size):
            checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, None, None, True, ep_rank)
            print(f'load {checkpoint_name}')
            split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
            for k, v in split_state.items():
                if 'local_experts' in k:
                    expert_local_rank = int(re.findall(pattern, k)[0])
                    expert_rank = expert_local_rank + num_local_experts * ep_rank
                    # k = k.replace(f'local_experts.{expert_local_rank}', f'local_experts.{expert_rank}')                  
                    if not args.moe_grouped_gemm:
                        k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')
                    else:
                        # print("Warning: moe_grouped_gemm is True, save all local experts weights in each checkpoint")
                        k = k.replace(f'weight{expert_rank}', f'weight{expert_local_rank}')
                state_dict[k] = v
    elif (
        args.tensor_model_parallel_size >= 1
        and args.pipeline_model_parallel_size >= 1
        and args.expert_model_parallel_size >= 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):
        #assert args.num_layers % args.pipeline_model_parallel_size == 0
        if args.target_decoder_first_pipeline_num_layers is not None:
            remained_layers = args.num_layers - args.target_decoder_first_pipeline_num_layers
            remained_stages = args.pipeline_model_parallel_size - 1
            assert remained_layers % remained_stages == 0
            pp_layers_per_stage = [args.target_decoder_first_pipeline_num_layers] +([remained_layers // remained_stages] * remained_stages)
        else:
            pp_layers_per_stage = [args.num_layers // args.pipeline_model_parallel_size] * args.pipeline_model_parallel_size

        #num_layers = args.num_layers // args.pipeline_model_parallel_size
        layers_to_copy = {}
        key_map = {}
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                for pp_rank in range(args.pipeline_model_parallel_size):
                    layer_offset = sum(pp_layers_per_stage[:pp_rank])
                    for layer in range(pp_layers_per_stage[pp_rank]):
                        pp_layer_id = layer + layer_offset
                        layers_to_copy[(pp_rank, layer)] = pp_layer_id

                    if args.expert_model_parallel_size > 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank, True,
                                                              ep_rank)
                    elif args.expert_model_parallel_size == 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank,
                                                              False)
                    print(f'load {checkpoint_name}')
                    split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
                    #if pp_rank == 0:
                        #with open("keymap.log", "w") as f:  # "a" 代表追加写入
                            #f.write(f"split_state: {split_state}\n")
                    for k, v in split_state.items():                
                        try:
                            if 'local_experts' in k or 'mlp.experts' in k and '_extra_state' not in k:
                                local_expert_rank = int(re.findall(pattern, k)[0])
                                expert_rank = local_expert_rank + num_local_experts * ep_rank
                                if not args.moe_grouped_gemm:
                                    k = k.replace(f'local_experts.{local_expert_rank}', f'local_experts.{expert_rank}')
                                else:
                                    k = k.replace(f'weight{local_expert_rank}', f'weight{expert_rank}')                        
                                    
                            layer_pattern = re.compile(r'\d+')
                            res = layer_pattern.findall(k)
                            tgt = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy[(pp_rank, int(res[0]))]), k)
                            if 'linear_proj' in k or 'linear_q_proj' in k or 'linear_q_down_proj' in k or 'linear_q_up_proj.weight'in k or \
                                    'linear_kv_up_proj.weight' in k or 'linear_kv_down_proj' in k or 'decoder.layers.0.mlp.linear_fc2' in k or \
                                    'decoder.layers.0.mlp.linear_fc1.weight' in k or 'shared_experts.linear_fc1' in k or 'shared_experts.linear_fc2' in k:
                                if ep_rank ==0:
                                    mid_state[tgt].append(v)
                            else:
                                mid_state[tgt].append(v)
                        except:
                            if "word_embeddings" in k:
                                if ep_rank ==0 and pp_rank == 0:
                                    mid_state[k].append(v)
                            elif "output_layer" in k or "final_layernorm" in k:
                                if ep_rank ==0 and pp_rank == args.pipeline_model_parallel_size - 1:
                                    mid_state[k].append(v)
                            else:
                                raise ValueError(f"{k} is missing! ")
        #with open("keymap.log", "w") as f:  # "a" 代表追加写入
            #f.write(f"keymap: {key_map}\n")
        #with open("mid_state.log", "w") as f:  # "a" 代表追加写入
            #f.write(f"mid_state: {mid_state}\n")
        for k, v in mid_state.items():
            if not isinstance(v[0], torch.Tensor) or 'router' in k or 'gate' in k:
                target_v = v[0]
            elif 'extra_state' in k:
                target_v = None
            elif 'word_embeddings' in k or 'output_layer' in k or 'final_layernorm' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k:
                target_v = torch.cat(v, dim=1)
            elif 'linear_q_proj' in k:
                viewed = [x.view(group_per_split, -1, q_head_dim, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
            elif 'linear_kv_up_proj.weight' in k:
                viewed = [x.view(group_per_split, -1, q_head_dim - args.qk_pos_emb_head_dim + args.v_head_dim, args.kv_lora_rank) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.kv_lora_rank)
            elif 'linear_q_up_proj.weight' in k:
                target_v = v[0]
            elif 'linear_q_down_proj' in k:
                target_v = v[0]
            elif 'linear_kv_down_proj' in k:
                target_v = v[0]
            elif 'linear_fc1.weight' in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
            elif 'linear_fc2' in k:
                target_v = torch.cat(v, dim=1)
            elif 'input_layernorm' in k:
                target_v = v[0]
            # elif 'q_layernorm' in k:
            elif 'linear_q_up_proj.layer_norm_weight' in k:
                target_v = v[0]
            # elif 'kv_layernorm' in k:
            elif 'linear_kv_up_proj.layer_norm_weight' in k:
                target_v = v[0]
            elif 'pre_mlp_layernorm' in k:
                target_v = v[0]
            elif 'linear_fc1.layer_norm_weight' in k:
                target_v = v[0]
            else:
                raise ValueError(f"{k} is missing!")
            state_dict[k] = target_v

    else:
        raise ValueError('not support yet')
    with open("state_dict.shape", "w") as f:
        for k, v in state_dict.items():
            if hasattr(v, "shape"):
                shape_str = str(tuple(v.shape))
            else:
                shape_str = f"Not a tensor (type={type(v)})"
            line = f"{k}: {shape_str}\n"
            f.write(line)
    with open("state_dict.log", "w") as f:  # "a" 代表追加写入
        f.write(f"state_dict: {state_dict}\n")
    model.load_state_dict(state_dict, strict=False)

    return model


def convert_checkpoint_from_megatron_to_transformers(mgmodel, hfmodel, args):

    if args.fp16:
        mgmodel = mgmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()

    with torch.no_grad():
        hfmodel.model.embed_tokens.weight.copy_(mgmodel.embedding.word_embeddings.weight)
        for layer_idx, (mglayer, hflayer) in enumerate(zip(mgmodel.decoder.layers, hfmodel.model.layers)):
            hflayer.input_layernorm.weight.copy_(mglayer.input_layernorm.weight)
            #hflayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)
            if layer_idx == 0:
                hflayer.post_attention_layernorm.weight.copy_(mglayer.mlp.linear_fc1.layer_norm_weight) ####
            else:
                if hasattr(mglayer.pre_mlp_layernorm, 'weight'): #### shoud be existed in 1-26 layers
                    hflayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)
            if args.q_lora_rank is not None:
                hflayer.self_attn.q_a_proj.weight.copy_(mglayer.self_attention.linear_q_down_proj.weight)
                hflayer.self_attn.q_b_proj.weight.copy_(mglayer.self_attention.linear_q_up_proj.weight)
                #hflayer.self_attn.q_a_layernorm.weight.copy_(mglayer.self_attention.q_layernorm.weight) ### 021
                hflayer.self_attn.q_a_layernorm.weight.copy_(mglayer.self_attention.linear_q_up_proj.layer_norm_weight)
            else:
                hflayer.self_attn.q_proj.weight.copy_(mglayer.self_attention.linear_q_proj.weight)

            hflayer.self_attn.kv_a_proj_with_mqa.weight.copy_(mglayer.self_attention.linear_kv_down_proj.weight)
            hflayer.self_attn.kv_b_proj.weight.copy_(mglayer.self_attention.linear_kv_up_proj.weight)
            #hflayer.self_attn.kv_a_layernorm.weight.copy_(mglayer.self_attention.kv_layernorm.weight)
            hflayer.self_attn.kv_a_layernorm.weight.copy_(mglayer.self_attention.linear_kv_up_proj.layer_norm_weight)

            hflayer.self_attn.o_proj.weight.copy_(mglayer.self_attention.linear_proj.weight)

            if layer_idx == 0:
                gate_weight, up_weight = torch.split(mglayer.mlp.linear_fc1.weight, split_size_or_sections=args.ffn_hidden_size)
                hflayer.mlp.gate_proj.weight.copy_(gate_weight)
                hflayer.mlp.up_proj.weight.copy_(up_weight)
                hflayer.mlp.down_proj.weight.copy_(mglayer.mlp.linear_fc2.weight)

            else:
                hflayer.mlp.gate.weight.copy_(mglayer.mlp.router.weight)
                
                if not args.moe_grouped_gemm:
                    for mgexpert, hfexpert in zip(mglayer.mlp.experts.local_experts, hflayer.mlp.experts):
                        gate_weight, up_weight = torch.split(mgexpert.linear_fc1.weight,
                                                            split_size_or_sections=args.moe_ffn_hidden_size)
                        hfexpert.gate_proj.weight.copy_(gate_weight)
                        hfexpert.up_proj.weight.copy_(up_weight)
                        hfexpert.down_proj.weight.copy_(mgexpert.linear_fc2.weight)
                else:
                    print("Warning: moe_grouped_gemm is True, skip local experts weight copy")
                    for i, hfexpert in enumerate(hflayer.mlp.experts):
                        mg_fc1_param = getattr(mglayer.mlp.experts.linear_fc1, f'weight{i}')
                        mg_fc2_param = getattr(mglayer.mlp.experts.linear_fc2, f'weight{i}')
                        gate_weight, up_weight = torch.split(mg_fc1_param,
                                                            split_size_or_sections=args.moe_ffn_hidden_size)
                        hfexpert.gate_proj.weight.copy_(gate_weight)
                        hfexpert.up_proj.weight.copy_(up_weight)
                        hfexpert.down_proj.weight.copy_(mg_fc2_param)
                        

                shared_expert_gate_weight, shared_expert_up_weight = \
                    torch.split(mglayer.mlp.shared_experts.linear_fc1.weight,
                                split_size_or_sections=args.moe_shared_expert_intermediate_size)
                hflayer.mlp.shared_experts.gate_proj.weight.copy_(shared_expert_gate_weight)
                hflayer.mlp.shared_experts.up_proj.weight.copy_(shared_expert_up_weight)
                hflayer.mlp.shared_experts.down_proj.weight.copy_(mglayer.mlp.shared_experts.linear_fc2.weight)

        hfmodel.model.norm.weight.copy_(mgmodel.decoder.final_layernorm.weight)
        hfmodel.lm_head.weight.copy_(mgmodel.output_layer.weight)
        #with open("hf_state_dict.log", "w") as f:  # "a" 代表追加写入
            #f.write(f"hf_state_dict: {hfmodel.state_dict()}\n")


def convert_checkpoint_from_transformers_to_megatron(hfmodel, mgmodel, args):

    if args.fp16:
        mgmodel = mgmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
    
    with torch.no_grad():
        mgmodel.embedding.word_embeddings.weight.copy_(hfmodel.model.embed_tokens.weight)
        for layer_idx, (mglayer, hflayer) in enumerate(zip(mgmodel.decoder.layers, hfmodel.model.layers)):
            print(layer_idx)
            mglayer.input_layernorm.weight.copy_(hflayer.input_layernorm.weight)
            if layer_idx == 0:
                mglayer.mlp.linear_fc1.layer_norm_weight.copy_(hflayer.post_attention_layernorm.weight) ####
            else:
                if hasattr(mglayer.pre_mlp_layernorm, 'weight'): #### shoud be existed in 1-26 layers
                    mglayer.pre_mlp_layernorm.weight.copy_(hflayer.post_attention_layernorm.weight)
            if args.q_lora_rank is not None:
                mglayer.self_attention.linear_q_down_proj.weight.copy_(hflayer.self_attn.q_a_proj.weight)
                mglayer.self_attention.linear_q_up_proj.weight.copy_(hflayer.self_attn.q_b_proj.weight)
                mglayer.self_attention.q_layernorm.weight.copy_(hflayer.self_attn.q_a_layernorm.weight)
            else:
                mglayer.self_attention.linear_q_proj.weight.copy_(hflayer.self_attn.q_proj.weight)
            mglayer.self_attention.linear_kv_down_proj.weight.copy_(hflayer.self_attn.kv_a_proj_with_mqa.weight)
            mglayer.self_attention.linear_kv_up_proj.weight.copy_(hflayer.self_attn.kv_b_proj.weight)
                # mglayer.self_attention.kv_layernorm.weight.copy_(hflayer.self_attn.kv_a_layernorm.weight)
            if hasattr(mglayer.self_attention.linear_kv_up_proj, 'layer_norm_weight'): #### should be existed in all layers
                mglayer.self_attention.linear_kv_up_proj.layer_norm_weight.copy_(hflayer.self_attn.kv_a_layernorm.weight)

            mglayer.self_attention.linear_proj.weight.copy_(hflayer.self_attn.o_proj.weight)

            if layer_idx == 0:
                mglayer.mlp.linear_fc1.weight.copy_(
                    torch.cat([hflayer.mlp.gate_proj.weight, hflayer.mlp.up_proj.weight]))
                mglayer.mlp.linear_fc2.weight.copy_(hflayer.mlp.down_proj.weight)
            else:
                mglayer.mlp.router.weight.copy_(hflayer.mlp.gate.weight) # checked
                if not args.moe_grouped_gemm:
                    for hf_expert, expert in zip(hflayer.mlp.experts, mglayer.mlp.experts.local_experts):
                        fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                        expert.linear_fc1.weight.copy_(fc1_weight)
                        expert.linear_fc2.weight.copy_(hf_expert.down_proj.weight)
                else:
                    print("Warning: moe_grouped_gemm is True, skip local experts weight copy")
                    for i, hf_expert in enumerate(hflayer.mlp.experts):
                        fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                        mg_fc1_param = getattr(mglayer.mlp.experts.linear_fc1, f'weight{i}')
                        mg_fc2_param = getattr(mglayer.mlp.experts.linear_fc2, f'weight{i}')
                        mg_fc1_param.copy_(fc1_weight)
                        mg_fc2_param.copy_(hf_expert.down_proj.weight)

                shared_fc1_weight = torch.cat(
                    [hflayer.mlp.shared_experts.gate_proj.weight, hflayer.mlp.shared_experts.up_proj.weight])
                mglayer.mlp.shared_experts.linear_fc1.weight.copy_(shared_fc1_weight)
                mglayer.mlp.shared_experts.linear_fc2.weight.copy_(hflayer.mlp.shared_experts.down_proj.weight)

        mgmodel.decoder.final_layernorm.weight.copy_(hfmodel.model.norm.weight)
        if args.untie_embeddings_and_output_weights:
            mgmodel.output_layer.weight.copy_(hfmodel.lm_head.weight)


def check_layer(layers_to_copy, k):
    pattern = re.compile(r"decoder.layers.\d+")
    res = pattern.findall(k)
    return res and res[0] in layers_to_copy.keys()

def save_mgmodel(mgmodel, args):

    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size

    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.load + "/*config.json " + args.save)
    os.system("cp -rf " + args.load + "/tokenizer* " + args.save)

    tracker_filepath = os.path.join(args.save, 'latest_checkpointed_iteration.txt')
    with open(tracker_filepath, "w") as f:
        f.write("release")

    full_model = mgmodel.state_dict_for_save_checkpoint()

    for k in list(full_model.keys()):
        if full_model[k] is None and '_extra_state' not in k:
            full_model.pop(k)
            continue
        if '_extra_state' in k and isinstance(full_model[k], torch.Tensor):
            full_model[k] = None

    if args.num_experts is not None:
        if args.moe_grouped_gemm:
            pattern = r'weight(\d+)'
        else:
            pattern = r'local_experts\.(\d+)\.'
        num_local_experts = args.num_experts // args.expert_model_parallel_size if args.num_experts else 0

    if (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size == 1
    ):
        checkpoint_name = get_checkpoint_name(args.save, 0, True)
        save_state_dict(args, [full_model], checkpoint_name)
    elif (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size >1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):
        for ep_rank in range(args.expert_model_parallel_size):
            model_split = {}
            checkpoint_name = get_checkpoint_name(args.save, 0, True, None, None, None, True, ep_rank)
            print(f'save ep_rank {ep_rank} model to {checkpoint_name}')
            for k, v in full_model.items():
                if 'local_experts' in k or 'mlp.experts' in k and '_extra_state' not in k:
                    expert_rank = int(re.findall(pattern, k)[0])
                    if expert_rank // num_local_experts != ep_rank:
                        continue
                    expert_local_rank = expert_rank % num_local_experts
                    if not args.moe_grouped_gemm:
                        k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')
                    else:
                        # print("Warning: moe_grouped_gemm is True, save all local experts weights in each checkpoint")
                        k = k.replace(f'weight{expert_rank}', f'weight{expert_local_rank}')
                                            
                model_split[k] = v
            save_state_dict(args, [model_split], checkpoint_name)
    if (
        args.pipeline_model_parallel_size > 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):

        if args.target_decoder_first_pipeline_num_layers is not None:
            remained_layers = args.num_layers - args.target_decoder_first_pipeline_num_layers
            remained_stages = args.pipeline_model_parallel_size - 1
            assert remained_layers % remained_stages == 0
            pp_layers_per_stage = [ args.target_decoder_first_pipeline_num_layers] +([remained_layers // remained_stages] * remained_stages)
        else:
            pp_layers_per_stage = [args.num_layers // args.pipeline_model_parallel_size] * args.pipeline_model_parallel_size

        #num_layers = args.num_layers // args.pipeline_model_parallel_size
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                for pp_rank in range(args.pipeline_model_parallel_size):
                    model_split = {}
                    #layer_offset = pp_rank * num_layers
                    layer_offset = sum(pp_layers_per_stage[:pp_rank])
                    layers_to_copy = {}
                    #for layer in range(num_layers):
                    for layer in range(pp_layers_per_stage[pp_rank]):
                        pp_layer_id = layer + layer_offset
                        layers_to_copy[f"decoder.layers.{pp_layer_id}"] = layer
                    if args.expert_model_parallel_size > 1:
                        checkpoint_name = get_checkpoint_name(args.save, 0, True, True, tp_rank, pp_rank, True, ep_rank)
                    elif args.expert_model_parallel_size == 1:
                        checkpoint_name = get_checkpoint_name(args.save, 0, True, True, tp_rank, pp_rank, False)
                    print(f'tensor_parallel & pipeline_parallel & expert_parallel, save model to {checkpoint_name}')
                    for k, v in full_model.items():

                        if check_layer(layers_to_copy, k):
                            layer_pattern = re.compile(r'\d+')
                            res = layer_pattern.findall(k)
                            k = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy["decoder.layers." + res[0]]), k)
                        elif not ("word_embeddings" in k or "output_layer" in k or "final_layernorm" in k):
                            continue

                        if not isinstance(v, torch.Tensor):
                            if 'local_experts' in k:
                                expert_rank = int(re.findall(pattern, k)[0])
                                if expert_rank // num_local_experts != ep_rank:
                                    continue
                                expert_local_rank = expert_rank % num_local_experts
                                k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')
                            target_v = v
                        elif 'linear_q_proj' in k or 'linear_q_down_proj' in k or 'linear_kv_down_proj' in k:
                            seg = v.shape[0] // args.tensor_model_parallel_size
                            target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                        elif 'linear_q_up_proj' in k:
                            seg = v.shape[0] // args.tensor_model_parallel_size
                            target_v = v[seg * tp_rank:seg * (tp_rank + 1)]
                        elif 'q_layernorm' in k:
                            seg = v.shape[0] // args.tensor_model_parallel_size
                            target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                        elif 'linear_kv_up_proj' in k:
                            seg = v.shape[0] // args.tensor_model_parallel_size
                            target_v = v[seg * tp_rank:seg * (tp_rank + 1)]
                        elif 'linear_proj' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                        elif 'decoder.layers.0.mlp.linear_fc2' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                        elif 'decoder.layers.0.mlp.linear_fc1' in k:
                            viewed = v.view(-1, args.ffn_hidden_size, args.hidden_size)
                            seg = args.ffn_hidden_size // args.tensor_model_parallel_size
                            target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                        elif 'local_experts' in k:
                            expert_rank = int(re.findall(pattern, k)[0])
                            if expert_rank // num_local_experts != ep_rank:
                                continue
                            expert_local_rank = expert_rank % num_local_experts
                            if 'linear_fc1' in k:
                                viewed = v.view(-1, args.moe_ffn_hidden_size, args.hidden_size)
                                seg = args.moe_ffn_hidden_size // args.tensor_model_parallel_size
                                target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                            elif 'linear_fc2' in k:
                                seg = v.shape[1] // args.tensor_model_parallel_size
                                target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                            k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')

                        elif 'shared_expert' in k and 'gate' not in k:
                            if 'linear_fc1' in k:
                                viewed = v.view(-1, args.moe_shared_expert_intermediate_size,
                                                args.hidden_size)
                                seg = args.moe_shared_expert_intermediate_size // args.tensor_model_parallel_size
                                target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                            elif 'linear_fc2' in k:
                                seg = v.shape[1] // args.tensor_model_parallel_size
                                target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]

                        elif "word_embeddings" in k or "output_layer" in k or "final_layernorm" in k:
                            seg = v.shape[0] // args.tensor_model_parallel_size
                            target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                        else:
                            target_v = v

                        if "word_embeddings" in k:
                            if pp_rank == 0:
                                model_split[k] = target_v
                        elif "output_layer" in k or "final_layernorm" in k:
                            if pp_rank == args.pipeline_model_parallel_size - 1:
                                model_split[k] = target_v
                        else:
                            model_split[k] = target_v
                    save_state_dict(args, [model_split], checkpoint_name)

    else:
        raise ValueError('Something is wrong, please check your tp/pp/ep size')

    print(f'megatron model is save to {args.save}')

def add_extra_args(parser):
    parser = get_patch_args(parser)
    parser = add_model_args(parser)
    return parser

def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()

    if args.convert_checkpoint_from_megatron_to_transformers:
        config = AutoConfig.from_pretrained(args.hf_ckpt_path, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=config.torch_dtype)
        with open("hf_model.txt", "w") as f:  # "a" 代表追加写入
            f.write(f"hf_model\n: {hf_model}\n")
        mg_model = load_megatron_model(args)
        with open("mg_model.txt", "w") as f:  # "a" 代表追加写入
            f.write(f"mg_model\n: {mg_model}\n")
        #print(hf_model)
        #print(mg_model)
        convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
        save_hfmodel(args, hf_model)
    else:
        config = AutoConfig.from_pretrained(args.load, trust_remote_code=True)
        # config.vocab_size = 102400
        # print_rank_0(f"vocab_size: {config.vocab_size}")
        hf_model = AutoModelForCausalLM.from_pretrained(args.load, trust_remote_code=True, torch_dtype=config.torch_dtype)
        mg_model = model_provider()
        convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
        print("done convert")
        save_mgmodel(mg_model, args)

if __name__ == "__main__":
    main()
