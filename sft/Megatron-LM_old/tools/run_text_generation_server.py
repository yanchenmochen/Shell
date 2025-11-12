# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT"""
import os
import sys

import torch
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
#                                              os.path.pardir)))

import musa_patch
sys.path.append("/mnt/seed-program-nas/001688/haoran.huang/Megatron-LM_1014")
from megatron.training import print_rank_0
from megatron.core.models.gpt import GPTModel
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
# from megatron.inference.text_generation_server import MegatronServer
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from typing import AsyncIterator, List
from contextlib import nullcontext
from typing import Union
import megatron

import os
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
import sys
from argparse import Namespace
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import SimpleTextGenerationController
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.module import MegatronModule
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
#                                              os.path.pardir, os.path.pardir)))

from megatron.training import get_args
from megatron.training import get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.core import mpu
from megatron.training.initialize import initialize_megatron
from megatron.training import get_model
import time

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the core GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """

    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')

    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=False,
            pre_process=pre_process,
            post_process=post_process
        )
    else:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te)
            else:
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=False,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            rope_scaling_factor=args.rope_scaling_factor,
        )

    return model


def get_inference_engine(args: Namespace, model: MegatronModule) -> AbstractEngine:
    """Get the relevant backend for running inference

    This function will automatically choose the TRTLLMBackend when possible, and default to Mcore backend if the user does not specify any backends. TRTLLMBackend is not implmented yet.

    Args:
        args (Namespace): The user arguments parsed from command line
        model (MegatronModule): The megatron model.

    Returns:
        AbstractBackend: The chosen backend
    """
    tokenizer = get_tokenizer()

    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=args.hidden_size,
        inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
        fp32_residual_connection=args.fp32_residual_connection,
        params_dtype=args.params_dtype,
        padded_vocab_size=args.padded_vocab_size,
        inference_max_seq_length=args.inference_max_seq_length,
        inference_max_requests=args.inference_max_batch_size
    )

    inference_wrapped_model = GPTInferenceWrapper(model, inference_wrapper_config)
    text_generation_controller = SimpleTextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer)
    return MCoreEngine(text_generation_controller=text_generation_controller)


def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--port", type=int, default=5000,
                       help='port for text generation server to run on')
    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--top_k", type=int, default=1,
                       help='Top k sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--return-log-probs", action='store_true', default=True,
                       help='Return the log probabilities of the final output tokens')
    group.add_argument("--num-tokens-to-generate", type=int, default=30,
                       help='Number of tokens to generate for each prompt')
    group.add_argument("--prompts", metavar='N', type=str, nargs='+',
                       help='Input prompts with each prompt within quotes and seperated by space')
    group.add_argument("--max-batch-size", type=int, default=8,
                       help='Max number of prompts to process at once')
    return parser


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True,
                                       'exit_on_missing_checkpoint': True})

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text "
                 "generation.")
    args.exit_on_missing_checkpoint = True

    # Set up model and load checkpoint
    load_context = nullcontext()
    if args.fp8:
        from transformer_engine.pytorch.fp8 import fp8_model_init
        load_context = fp8_model_init()
    with load_context:
        model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    model.eval()
    def print_inf_name(name):
        def check_inf_nan(grad):
            if torch.isinf(grad).any():
                print("Inf detected in gradients!", name)
                print(grad)
            elif torch.isnan(grad).any():
                print("NaN detected in gradients!",name)
                print(grad)
            else:
                print(name, grad)
        return check_inf_nan

    def forward_output(name):
        def forward_hook(module, input, output):
            print(f"Inside {module.__class__.__name__} forward hook")
            print(f"Input: {input}")  # 假设输入是个张量
            print(f"Output: {output}")
            try:
                print("is_inf1:", torch.isinf(output[0]).any(), "is_nan1:",torch.isnan(output[0]).any(), output)
                print("is_inf1:", torch.isinf(input[0]).any(), "is_nan1:",torch.isnan(input[0]).any(), input)
            except:
                pass
            try:
                print("weight:", module.weight)
            except:
                pass
            index = 0
            
        return forward_hook

    def backward_output(name):
       def print_backward_hook(module, grad_input, grad_output):
           #torch.set_printoptions(profile='full')
            if len(grad_output) > 0 and grad_output[0] != None :
                for idx, output in enumerate(grad_output):
                    if output is not None and (torch.isinf(output).any() or torch.isnan(output).any()):
                        global_rank = torch.distributed.get_rank()
                        print(module.__class__, 'backward ends', name, len(grad_output), len(grad_input))
                        print("is_inf1:", torch.isinf(output).any(), "is_nan1:",torch.isnan(output).any(), output,'global_rank', global_rank)
            if len(grad_input) > 0 and grad_input[0] !=None:
                for idx, input in enumerate(grad_input):
                    print(module.__class__, 'backward ends', name, grad_output, grad_input)
                    # if input is not None and (torch.isinf(input).any() or torch.isnan(input).any()):
                    #     global_rank = torch.distributed.get_rank()
                    #     print(module.__class__, 'backward ends', name, len(grad_output), len(grad_input))
                    #     print("is_inf2:", torch.isinf(input).any(), "is_nan2:",torch.isnan(input).any(),input, 'global_rank', global_rank, "idx:", idx)
                    #     for idx1, output in enumerate(grad_output):
                    #         torch.save(input.cpu(), f'global-{global_rank}.{name}.{idx1}.nan1.grad_output.pt')
                    #     for idx2, input in enumerate(grad_input):
                    #         torch.save(input.cpu(), f'global-{global_rank}.{name}.{idx2}.nan2.grad_input.pt')
       return print_backward_hook
    # print(model)
    # for name, module in model.named_modules():

    #     # print(name, model_module)
    #    # module.register_forward_pre_hook(print_pre_forward_hook)
    #    # module.register_forward_hook(print_forward_hook)
    # #    if name == 'model.layers.2.self_attn.core_attention.pv' or name == 'model.layers.2.self_attn.core_attention.softmax':
    #        #module.register_full_backward_pre_hook(print_pre_backward_hook)
    #        #if name == 'model.layers.25.input_layernorm':
    #     module.register_forward_hook(forward_output(name))

    inference_engine = get_inference_engine(args, model)
    sampling_params = SamplingParams(
        temperature=0,
        top_k=1,
        top_p=0.0,
        return_log_probs=True,
        num_tokens_to_generate=10,
    )

    if args.enable_cuda_graph:
        print(f"Running warmup for CUDA graphs...")
        inference_engine.generate(
                prompts=args.prompts, sampling_params=sampling_params
            )
    import asyncio
    from typing import AsyncIterator, List
    start_time = time.perf_counter()
    
    results: List[InferenceRequest] = inference_engine.generate(
            # prompts=['Once'], sampling_params=sampling_params,
            prompts=["<|start_header_id|>user<|end_header_id|>Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Vertical integration forwards is when a firm mergers or acquires another\n\nA) Towards the source of supply\nB) Towards the consumer\nC) At the same stage of the supply chain\nD) In another industry<|start_header_id|>assistant<|end_header_id|>", "<|start_header_id|>user<|end_header_id|>The future of machine learning depends on?<|start_header_id|>assitant<|end_header_id|>", "<|start_header_id|>user<|end_header_id|>Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD.\n\nWhich State ordinarily exercises jurisdiction in respect of crimes committed on board vessels?\n\nA) The coastal State\nB) The flag State\nC) All States enjoy such jurisdiction\nD) The International Tribunal for the Law of the Sea<|start_header_id|>assitant<|end_header_id|>"], sampling_params=sampling_params,
        )
    end_time = time.perf_counter()
    latency = end_time - start_time

    if torch.distributed.get_rank() == 0:
        for idx, result in enumerate(results):
            print(f' \n------------- RESULT FOR PROMPT {idx} --------------- ')
            result = {
                'id': result.request_id,
                'input_prompt': result.prompt,
                'generated_text': result.generated_text,
                'generated_tokens': result.generated_tokens,
                'latency': latency,
                'probabilities': result.generated_log_probs,
            }
            print(result)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("/mnt/seed-program-nas/001688/haoran.huang/llama3_tokenizer_align")
            print(tokenizer.decode(result["generated_tokens"].cpu().numpy()))
    
    torch.distributed.destroy_process_group()
    # if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
    #     server = MegatronServer(inference_engine, args)
    #     server.run("0.0.0.0",port=args.port)
