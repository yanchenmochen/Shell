# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial
from contextlib import nullcontext
import inspect

from typing import List, Optional, Tuple, Union

# if os.getenv("ACCELERATOR_BACKEND", "musa") == "musa":
if os.getenv("ACCELERATOR_BACKEND") == "musa":
    import musa_patch
else:
    import cuda_patch
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from data.utils import (
    get_batch_on_this_tp_rank_original,
    get_batch_on_this_tp_rank_idxmap_sft,
    get_position_id_on_this_tp_rank_idxmap_sft_packing
)
from data import train_valid_test_datasets_provider

SPIKY_LOSS_PERC = 0.3  # percentage threshold for spiky loss

stimer = StragglerDetector()


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
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
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
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
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

    def forward_output(name):
        def forward_hook(module, input, output):
            print(f"Inside {module.__class__.__name__} forward hook")
            print(f"Input: {input}")  # 假设输入是个张量
            print(f"Output: {output}")
            if len(input) > 0 and input[0] != None:
                try:
                    print("is_inf1:", torch.isinf(input[0]).any(), "is_nan1:", torch.isnan(input[0]).any())
                    print("is_inf1:", torch.isinf(output[0]).any(), "is_nan1:", torch.isnan(output[0]).any())
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
            # torch.set_printoptions(profile='full')
            if len(grad_output) > 0 and grad_output[0] != None:
                for idx, output in enumerate(grad_output):
                    if output is not None and (torch.isinf(output).any() or torch.isnan(output).any()):
                        global_rank = torch.distributed.get_rank()
                        print(module.__class__, 'backward ends', name, len(grad_output), len(grad_input))
                        print("is_inf1:", torch.isinf(output).any(), "is_nan1:", torch.isnan(output).any(), output,
                              'global_rank', global_rank)
            if len(grad_input) > 0 and grad_input[0] != None:
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
    #     module.register_full_backward_hook(backward_output(name))

    return model


def get_batch(data_iterator):
    """Generate a batch."""
    args = get_args()

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        packed_seq_params = None
        if args.dataset == 'MMAP' and args.train_mode == "finetune" and args.reset_position_ids:
            position_ids = get_position_id_on_this_tp_rank_idxmap_sft_packing(data_iterator)
            position_ids = position_ids[0]  # shape: [seq_length]
            start_indices = (position_ids == 0).nonzero(as_tuple=True)[0]
            seqlens = start_indices[1:] - start_indices[:-1]
            # NOTE: cu_seqlens: [0, A1, A1+A2, A1+A2+A3, ..., seq_len]
            cu_seqlens = torch.zeros(start_indices.shape[0] + 1, device=position_ids.device, dtype=torch.int)
            cu_seqlens[1:-1] = torch.cumsum(seqlens, dim=0)
            cu_seqlens[-1] = position_ids.shape[0]
            max_seqlen = torch.max(seqlens.max(), position_ids.max() + 1)
            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                qkv_format='thd',
                max_seqlen_q=max_seqlen,
                max_seqlen_kv=max_seqlen,
            )

        return None, None, None, None, None, None, packed_seq_params

    if args.dataset == 'JSON-SFT':
        if args.train_mode == "pretrain":
            raise ValueError('The JSON-SFT dataset should only be used for finetuning!')
        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank_original(data_iterator, per_seq_average=False)
        # slice batch along sequence dimension for context parallelism
        num_seqs = batch.pop('num_seqs')
        batch = get_batch_on_this_cp_rank(batch)

        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            None
        )
    elif args.dataset == 'MMAP':
        # get batches based on the TP rank you are on
        if args.train_mode == "pretrain":
            batch = get_batch_on_this_tp_rank(data_iterator)
        else:
            batch = get_batch_on_this_tp_rank_idxmap_sft(data_iterator, per_seq_average=False)

        packed_seq_params = None
        if args.reset_position_ids:
            # sequence-packing, build cu_seqlens
            position_ids = batch.get('position_ids', None)
            if position_ids is not None:
                # mbs = 1
                position_ids = position_ids[0]  # shape: [seq_length]
                start_indices = (position_ids == 0).nonzero(as_tuple=True)[0]
                seqlens = start_indices[1:] - start_indices[:-1]
                # NOTE: cu_seqlens: [0, A1, A1+A2, A1+A2+A3, ..., seq_len]
                cu_seqlens = torch.zeros(start_indices.shape[0] + 1, device=position_ids.device, dtype=torch.int)
                cu_seqlens[1:-1] = torch.cumsum(seqlens, dim=0)
                cu_seqlens[-1] = position_ids.shape[0]
                max_seqlen = torch.max(seqlens.max(), position_ids.max() + 1)
                packed_seq_params = PackedSeqParams(
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    qkv_format='thd',
                    max_seqlen_q=max_seqlen,
                    max_seqlen_kv=max_seqlen,
                )

        if packed_seq_params is not None and args.context_parallel_size > 1:
            raise ValueError('Sequence Packing is not supported when CP>1 !')
        # slice batch along sequence dimension for context parallelism
        num_seqs = batch.pop('num_seqs', None)
        batch = get_batch_on_this_cp_rank(batch)
        if batch['tokens'].max() > 128255:
            print(batch['tokens'])
            # print(tokenizer.decode(batch['tokens'][0].cpu().numpy()))
            # batch['tokens'] = torch.where(batch['tokens'] > 128255, 128255, batch['tokens'])
            # from transformers import AutoTokenizer
            # tokenizer = AutoTokenizer.from_pretrained("/mnt/seed-program-nas/001688/xiechunhong/1B_Loss_Align/Meta-Llama-3-tokenizer")
        # print("labels:", batch['labels'][0])
        # print("label all tokens:", (batch['labels'][0] != -100).sum())
        # print("loss mask:", batch['loss_mask'][0])
        # batch['labels'] = torch.where(batch['labels'] == -100, 128006, batch['tokens'])
        # print(tokenizer.decode(batch['labels'][0].cpu().numpy()))
        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            packed_seq_params
        )
    else:
        raise ValueError("please set correct --dataset ")


# def get_batch(data_iterator):
#     """Generate a batch."""

#     # TODO: this is pretty hacky, find a better way
#     if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
#         return None, None, None, None, None

#     # get batches based on the TP rank you are on
#     batch = get_batch_on_this_tp_rank(data_iterator)

#     # slice batch along sequence dimension for context parallelism
#     batch = get_batch_on_this_cp_rank(batch)
#     # batch['tokens'] = torch.where(batch['tokens'] >= 128000, 5713, batch['tokens'])
#     # batch['labels'] = torch.where(batch['labels'] >= 128000, 5713, batch['labels'])
#     # print("tokens:", batch['tokens'][0])
#     # print("labels:", batch['labels'][0])
#     # from transformers import AutoTokenizer
#     # tokenizer = AutoTokenizer.from_pretrained("/mnt/seed-program-nas/001688/haoran.huang/llama3_tokenizer_align")
#     # print(tokenizer.decode(batch['tokens'][0].cpu().numpy()))
#     # print(tokenizer.decode(batch['labels'][0].cpu().numpy()))
#     return batch.values()

def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    # print("losses shape:", losses)
    # print("loss_mask:", loss_mask.sum())
    # if loss_mask.sum() == 0:
    #     loss_mask = torch.ones_like(loss_mask)
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    # y = losses.view(-1) * loss_mask
    # print(losses.view(-1), losses.view(-1).shape)
    # print(loss_mask, loss_mask.shape)
    # print(y[torch.isin(y, losses.view(-1))])
    # print(y[y > 0], torch.mean(losses.view(-1)), torch.sum(losses.view(-1) * loss_mask), "tokens contributing to loss")
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(rerun_state_machine.is_spiky_loss, threshold=SPIKY_LOSS_PERC),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    if not int(os.getenv("NO_LOSS_REDUCE", 0)):  # TODO:(huang.huang) will influence the loss reported Now!
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

        if int(os.getenv("USE_EPX", 0)):
            from musa_patch.fault_tolerance_epx.epx_sync_tensor import epx_sync_tensor_across_replicas
            epx_sync_tensor_across_replicas(reporting_loss, opts=torch.distributed.ReduceOp.AVG, assemble=False)

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids, _, _ = get_batch(
            data_iterator)
    timers('batch-generator').stop()
    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
            mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


# def train_valid_test_datasets_provider(train_val_test_num_samples):
#     """Build the train test and validation datasets.

#     Args:
#         train_val_test_num_samples : A list containing the number of samples in train test and validation.
#     """
#     args = get_args()

#     config = core_gpt_dataset_config_from_args(args)

#     if args.mock_data:
#         dataset_type = MockGPTDataset
#     else:
#         dataset_type = GPTDataset

#     print_rank_0("> building train, validation, and test datasets for GPT ...")

#     train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
#         dataset_type,
#         train_val_test_num_samples,
#         is_dataset_built_on_rank,
#         config
#     ).build()

#     print_rank_0("> finished creating GPT datasets ...")

#     return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
