# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import torch
from tqdm import tqdm
import types

# >>>
sys.path.append("/lustre/fs1/portfolios/adlr/users/lmcafee/llama/2/llama")

from lutil import pax, tp as _tp
# <<<

def add_arguments(parser):
    group = parser.add_argument_group(title='Llama-2 loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--tokenizer-model', required=True,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')

def load_args_from_checkpoint(args):

    # Read Llama args.
    llama_args_path = os.path.join(args.load, "params.json")
    with open(llama_args_path) as f:
        llama_args = json.load(f)

    # pt_path = os.path.join(args.load, "

    # Update Megatron args.
    args.seq_length = 4096
    args.max_position_embeddings = 4096
    args.hidden_size = llama_args["dim"]
    # args.make_vocab_size_divisible_by = llama_args["multiple_of"]
    args.num_attention_heads = llama_args["n_heads"]
    args.num_layers = llama_args["n_layers"]
    args.global_batch_size = 1024
    args.layernorm_epsilon = llama_args["norm_eps"]
    args.iteration = 0
    args.add_position_embedding = False
    args.use_rotary_position_embeddings = True
    # args.rotary_percent = 0.5
    args.swiglu = True
    args.tokenizer_type = "Llama2"
    # args.tokenizer_type = "SentencePieceTokenizer"
    args.bf16 = True
    args.norm_type = "rms"

    args.untie_embeddings_and_output_weights = True # to get 'output_layer'
    args.vocab_size = -1 # 32000 # ... set from tokenizer
    args.padded_vocab_size = -1 # 32000 # ... set from tokenizer
    # args.llama_num_kv_heads = llama_args["n_kv_heads"]
    # args.llama_ffn_dim_multiplier = llama_args["ffm_dim_multiplier"]
    # args.llama_multiple_of = llama_args["multiple_of"]
    args.llama = llama_args

    ffn_dim_multiplier = llama_args.get("ffn_dim_multiplier", 1.)
    ffn_multiple_of = llama_args["multiple_of"]
    ffn_hidden_size = 4 * int(2 * args.hidden_size / 3)
    if ffn_dim_multiplier is not None:
        ffn_hidden_size = int(ffn_dim_multiplier * ffn_hidden_size)
    ffn_hidden_size = ffn_multiple_of * ((ffn_hidden_size + ffn_multiple_of - 1) // ffn_multiple_of)
    args.ffn_hidden_size = ffn_hidden_size

    if "n_kv_heads" in llama_args:
        args.group_query_attention = True
        args.num_query_groups = llama_args["n_kv_heads"]

    # >>>
    # pax({
    #     "ffn_dim_multiplier" : ffn_dim_multiplier,
    #     "ffn_multiple_of" : ffn_multiple_of,
    #     "ffn_hidden_size" : ffn_hidden_size,
    # })
    # <<<

    model_name = os.path.basename(args.load).split("-")[-1]
    args.tensor_model_parallel_size = {
        "7b" : 1,
        "13b" : 2,
        "70b" : 8,
    }[model_name]
    args.pipeline_model_parallel_size = 1

    # >>>
    # pax({
    #     "args": args,
    #     "num_attention_heads" : args.num_attention_heads,
    #     "n_kv_heads" : llama_args["n_kv_heads"],
    # })
    # <<<

def load_vocab_size(args):
    from megatron.tokenizer import build_tokenizer
    tokenizer = build_tokenizer(args)
    args.vocab_size = tokenizer.vocab_size
    args.padded_vocab_size = args.vocab_size # llama doesn't pad

    # >>>
    # pax({
    #     "args / vocab_size": args.vocab_size,
    #     "args / padded_vocab_size": args.padded_vocab_size,
    # })
    # <<<

def concatenate_embeddings(args):

    # >>>
    return None
    # <<<

    # Load & concatenate embeddings.
    embedding_shards = []
    for rank in tqdm(range(args.tensor_model_parallel_size), "embedding shards"):
        filename = os.path.join(args.load, f"consolidated.0{rank}.pth")
        assert os.path.isfile(filename), f"missing checkpoint file '{filename}'."
        state_dict = torch.load(filename)
        embedding_shards.append(state_dict["tok_embeddings.weight"])
    embeddings = torch.cat(embedding_shards, dim=1)

    # pax({
    #     "embedding_shards" : [ str(t.shape) for t in embedding_shards ],
    #     "embeddings" : _tp(embeddings),
    # })

    return embeddings

# >>>
# def get_megatron_rotary_freqs(args):

#     from torch import einsum, nn

#     class RotaryEmbedding(nn.Module):
#         def __init__(self, dim):
#             super().__init__()
#             inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
#             # inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2)[:(dim//2)].float() / dim))
#             # >>>
#             # pax({"dim": dim, "inv_freq": _tp(inv_freq)})
#             # <<<
#             self.register_buffer('inv_freq', inv_freq)

#         def forward(self, max_seq_len, offset=0):
#             seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
#             freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
#             # >>>
#             # pax({"inv_freq": _tp(self.inv_freq)})
#             # return self.inv_freq
#             # return freqs
#             # <<<
#             # first part even vector components, second part odd vector components,
#             #  2 * dim in dimension size
#             emb = torch.cat((freqs, freqs), dim=-1)
#             # emb [seq_length, .., dim]
#             return emb[:, None, None, :]

#     rotary_dim = args.hidden_size // args.num_attention_heads \
#         if args.kv_channels is None else args.kv_channels
#     return RotaryEmbedding(rotary_dim)(args.seq_length)

#     # def _rotate_half(x):
#     #     """
#     #     change sign so the last dimension becomes [-odd, +even]
#     #     """
#     #     x1, x2 = torch.chunk(x, 2, dim=-1)
#     #     return torch.cat((-x2, x1), dim=-1)

#     # def apply_rotary_pos_emb(t, freqs):
#     #     """
#     #     input tensor t is of shape [seq_length, ..., dim]
#     #     rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
#     #     check https://kexue.fm/archives/8265 for detailed formulas
#     #     """
#     #     rot_dim = freqs.shape[-1]
#     #     # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
#     #     t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

#     #     # first part is cosine component
#     #     # second part is sine component, need to change signs with _rotate_half method
#     #     t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
#     #     return torch.cat((t, t_pass), dim=-1)

# def get_llama_rotary_freqs(args):

#     # from llama.model import precompute_freqs_cis
#     def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
#         freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#         # pax({"dim": dim, "freqs": _tp(freqs)})
#         t = torch.arange(end, device=freqs.device)  # type: ignore
#         # >>>
#         # return freqs
#         # <<<
#         freqs = torch.outer(t, freqs).float()  # type: ignore
#         # >>>
#         # return freqs
#         # <<<
#         freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
#         return freqs_cis

#     # def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
#     #     ndim = x.ndim
#     #     assert 0 <= 1 < ndim
#     #     assert freqs_cis.shape == (x.shape[1], x.shape[-1])
#     #     shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
#     #     return freqs_cis.view(*shape)

#     # def apply_rotary_emb(
#     #     xq: torch.Tensor,
#     #     xk: torch.Tensor,
#     #     freqs_cis: torch.Tensor,
#     # ) -> Tuple[torch.Tensor, torch.Tensor]:
#     #     xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
#     #     xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
#     #     freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
#     #     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
#     #     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
#     #     return xq_out.type_as(xq), xk_out.type_as(xk)

#     freqs_cis = precompute_freqs_cis(
#         args.hidden_size // args.num_attention_heads, args.seq_length * 2)

#     # pax({
#     #     "freqs_cis" : _tp(freqs_cis),
#     # })

#     return freqs_cis
# <<<

def set_rmsnorm_state(rmsnorm, tensor):
    rmsnorm.weight.data.copy_(tensor)
    # pax({
    #     "rmsnorm" : _tp(rmsnorm.weight.clone()),
    #     "tensor" : _tp(tensor),
    # })

# def set_preprocess_state(model, state_dict, args, rank, embeddings):
def set_preprocess_state(args, rank, model, embeddings, state_dict):

    # >>>
    # from megatron.core.models.common.rotary_pos_embedding import RotaryEmbedding
    # megatron_freqs = get_megatron_rotary_freqs(args)
    # llama_freqs = get_llama_rotary_freqs(args)

    # rotary_pos_emb = model.language_model.rotary_pos_emb
    # pax({
    #     "rotary_pos_emb" : rotary_pos_emb,
    #     "rope.freqs" : _tp(state_dict["rope.freqs"]),
    #     "megatron_freqs" : _tp(megatron_freqs),
    #     "llama_freqs" : _tp(llama_freqs),
    # })
    # <<<

    padded_vocab_size = args.padded_vocab_size
    world_size = args.tensor_model_parallel_size
    assert padded_vocab_size % world_size == 0
    shard_size = args.padded_vocab_size // world_size
    start_idx = rank * shard_size
    end_idx = min(embeddings.shape[0], start_idx + shard_size)

    model.language_model.embedding.word_embeddings.weight[0:(end_idx-start_idx)].data.copy_(embeddings[start_idx:end_idx])
    
    if rank == 7:
        pax({
            "word_embeddings" : _tp(model.language_model.embedding.word_embeddings.weight.clone()),
            "embeddings" : _tp(embeddings),
            "padded_vocab_size" : args.padded_vocab_size,
            "rank" : rank,
            "world_size" : world_size,
            "shard_size" : shard_size,
            "start_idx" : start_idx,
            "end_idx" : end_idx,
        })

# def set_postprocess_state(model, state_dict):
def set_postprocess_state(args, model, state_dict):

    model_norm = model.language_model.encoder.final_layernorm
    model_output = model.language_model.output_layer
    model_norm.weight.data.copy_(state_dict["norm.weight"])
    model_output.weight.data.copy_(state_dict["output.weight"])

    # set_rmsnorm_state(model.language_model.encoder.final_layernormm, state_dict["norm.weight"])
    
    # pax({
    #     "model_norm" : _tp(model_norm.weight.clone()),
    #     "model_output" : _tp(model_output.weight.clone()),
    #     "norm.weight" : _tp(state_dict["norm.weight"]),
    #     "output.weight" : _tp(state_dict["output.weight"]),
    # })

def set_attn_state(args, layer, layer_state_dict):

    # Get attention layer & state.
    attn = layer.self_attention
    attn_state_dict = {k.split(".")[1]:v for k,v in layer_state_dict.items()
                       if k.startswith("attention.")}

    # Reshape loaded weights.
    tp = args.tensor_model_parallel_size
    nh = args.num_attention_heads // tp
    ng = (args.num_query_groups if args.group_query_attention \
        else args.num_attention_heads) // tp
    dim = args.kv_channels
    assert nh % ng == 0

    # Copy weights.
    # attn.query_key_value.weight.data.copy_(torch.cat([
    #     attn_state_dict["wq"],
    #     attn_state_dict["wk"],
    #     attn_state_dict["wv"],
    # ], dim=0))
    attn.query_key_value.weight.data.copy_(torch.cat([ 
        attn_state_dict["wq"].reshape((ng, dim*nh//ng, -1)),
        attn_state_dict["wk"].reshape((ng, dim, -1)),
        attn_state_dict["wv"].reshape((ng, dim, -1)),
    ], dim=1).reshape((-1, args.hidden_size)))
    attn.dense.weight.data.copy_(attn_state_dict["wo"])

    # pax({
    #     "num_attention_heads" : args.num_attention_heads,
    #     "num_query_groups" : args.num_query_groups,
    #     "kv_channels" : args.kv_channels,
    #     # "attn" : attn,
    #     "attn / query_key_value" : _tp(attn.query_key_value.weight.clone()),
    #     "attn / dense" : _tp(attn.dense.weight.clone()),
    #     "attn_state_dict": {k:str(v.shape) for k,v in attn_state_dict.items()},
    #     "kv_channels" : args.kv_channels,
    # })
    # pax({
    #     "tp" : tp,
    #     "nh" : nh,
    #     "ng" : ng,
    #     "dim" : dim,
    #     # "wq" : _tp(wq),
    #     # "wk" : _tp(wk),
    #     # "wv" : _tp(wv),
    #     "naive" : _tp(torch.cat([attn_state_dict[k]
    #                              for k in ("wq", "wk", "wv")], dim=0)),
    #     "query_key_value" : _tp(query_key_value),
    # })

def set_mlp_state(args, layer, layer_state_dict):

    mlp = layer.mlp
    mlp_state_dict = {k.split(".")[1]:v for k,v in layer_state_dict.items() if k.startswith("feed_forward")}

    mlp.dense_h_to_4h.weight.data.copy_(torch.cat([
        mlp_state_dict["w1"],
        mlp_state_dict["w3"],
    ], dim=0))
    mlp.dense_4h_to_h.weight.data.copy_(mlp_state_dict["w2"])

    # pax({
    #     "mlp" : mlp,
    #     "mlp / dense_h_to_4h" : _tp(mlp.dense_h_to_4h.weight.clone()),
    #     "mlp / dense_4h_to_h" : _tp(mlp.dense_4h_to_h.weight.clone()),
    #     "mlp_state_dict" : {k:str(v.shape) for k,v in mlp_state_dict.items()},
    #     "h_to_4h" : _tp(h_to_4h),
    #     "4h_to_h" : _tp(_4h_to_h),
    # })

# def set_layer_state(model, model_state_dict, layer_idx):
def set_layer_state(args, model, model_state_dict, layer_idx):

    layer = model.language_model.encoder.layers[layer_idx]

    layer_state_dict = {".".join(k.split(".")[2:]):v
                        for k,v in model_state_dict.items()
                        if k.startswith(f"layers.{layer_idx}")}
    # layer_state_dict["rope.freqs"] = model_state_dict["rope.freqs"]

    set_attn_state(args, layer, layer_state_dict)
    set_mlp_state(args, layer, layer_state_dict)
    set_rmsnorm_state(layer.input_layernorm,
                      layer_state_dict["attention_norm.weight"])
    set_rmsnorm_state(layer.post_attention_layernorm,
                      layer_state_dict["ffn_norm.weight"])

    # pax({
    #     "layer" : layer,
    #     "layer / mlp" : layer.mlp,
    #     "layer / mlp / h->4h" : _tp(layer.mlp.dense_h_to_4h.weight.clone()),
    #     "layer / mlp / 4h->h" : _tp(layer.mlp.dense_4h_to_h.weight.clone()),
    #     "layer_state_dict" : {k:str(v.shape) for k,v in layer_state_dict.items()},
    # })

def load_checkpoint_to_model(args, rank, model, embeddings):

    # Load state dict.
    filename = os.path.join(args.load, f"consolidated.0{rank}.pth")
    assert os.path.isfile(filename), f"missing checkpoint file '{filename}'."
    state_dict = torch.load(filename)

    # Set model state.
    # set_preprocess_state(args, rank, model, embeddings, state_dict)
    set_postprocess_state(args, model, state_dict)
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        set_layer_state(args, model, state_dict, layer_idx)

    # pax({
    #     "args" : args,
    #     "rank" : rank,
    #     "model" : model,
    #     "filename" : filename,
    #     "state_dict" : list(state_dict.keys()),
    # })

def _load_checkpoint(queue, args):

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.arguments import parse_args, validate_args
        from megatron.global_vars import set_args, set_global_variables
        # from megatron.checkpointing import load_args_from_checkpoint, load_checkpoint
        from megatron.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron import fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--load', args.load_dir
                ]

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    load_args_from_checkpoint(margs)
    load_vocab_size(margs)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('disable_bias_linear', False)
    check_for_arg('params_dtype')
    check_for_arg('swiglu', False)

    # Determine how to make our models
    assert args.model_type == 'GPT', 'Llama-2 is a GPT model.'
    from pretrain_gpt import model_provider
    margs.model_type = ModelType.encoder_or_decoder

    # suppress warning about torch.distributed not being initialized
    module.MegatronModule.embedding_warning_printed = True

    # Concatenate word embeddings. (Llama uses different sharding than Megatron.)
    embeddings = concatenate_embeddings(margs)

    def get_models(count, dtype):
        models = []
        for rank in range(count):
            print("loading rank %d / %d." % (rank, count))
            mpu.set_tensor_model_parallel_rank(rank)
            model = model_provider(True, True).to(dtype)
            # >>>
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print(model)
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # exit()
            # <<<
            load_checkpoint_to_model(margs, rank, model, embeddings)
            models.append(model)
            # pax({"model": model})
        return models

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    fused_kernels.load(margs)

    # >>>
    # # Get true (non-padded) vocab size
    # if args.true_vocab_size is not None:
    #     true_vocab_size = args.true_vocab_size
    # elif args.vocab_file is not None:
    #     vocab = json.load(open(args.vocab_file))
    #     true_vocab_size = len(vocab)
    #     if args.true_vocab_size is not None and true_vocab_size != args.true_vocab_size:
    #         print("Both --true-vocab-size and --vocab-file specified and the vocab size does not match, aborting.")
    #         queue.put("exit")
    #         exit(1)
    # else:
    #     true_vocab_size = None
    # <<<

    # short aliases
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # metadata
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    # md.true_vocab_size = true_vocab_size
    # md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    # md.checkpoint_args = checkpoint_args

    # pax({"margs": margs, "md": md, "tp_size": tp_size})

    # Get first pipe stage
    mpu.set_pipeline_model_parallel_rank(0)
    all_models = [get_models(tp_size, md.params_dtype)]
    models = all_models[0][0]

    pax({"models": models})

    md.consumed_train_samples = 0 # consumed_train_samples
    md.consumed_valid_samples = 0 # consumed_valid_samples
    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings
    message = {
        "word embeddings": torch.cat(
            [models[tp_rank].language_model.embedding.word_embeddings.weight.data for tp_rank in range(tp_size)],
            dim = 0)
    }
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = models[0].language_model.embedding.position_embeddings.weight.data
    else:
        assert not hasattr(models[0].language_model.embedding, 'position_embeddings')

    queue_put("embeddings", message)

    total_layer_num = 0
    for vp_rank in range(vp_size):
        mpu.set_virtual_pipeline_model_parallel_rank(vp_rank)
        for pp_rank in range(pp_size):
            if pp_rank > 0:
                mpu.set_pipeline_model_parallel_rank(pp_rank)
                if vp_rank == 0:
                    all_models.append(get_models(tp_size, md.params_dtype))
            models = all_models[pp_rank][vp_rank]
            for layer_num in range(len(models[0].language_model.encoder.layers)):
                message = {}

                # Get non-parallel tensors from tp_rank 0
                layer = models[0].language_model.encoder.layers[layer_num]
                message["input layernorm weight"] = layer.input_layernorm.weight.data
                message["input layernorm bias"] = layer.input_layernorm.bias.data
                message["post layernorm weight"] = layer.post_attention_layernorm.weight.data
                message["post layernorm bias"] = layer.post_attention_layernorm.bias.data
                if md.linear_bias:
                    message["dense bias"] = layer.self_attention.dense.bias.data
                    message["mlp l1 bias"] = layer.mlp.dense_4h_to_h.bias.data

                # Grab all parallel tensors for this layer
                qkv_weight = []
                qkv_bias = []
                dense_weight = []
                mlp_l0_weight = []
                mlp_l0_bias = []
                mlp_l1_weight = []
                for tp_rank, model in enumerate(models):
                    layer = model.language_model.encoder.layers[layer_num]
                    qkv_weight.append(layer.self_attention.query_key_value.weight.data)
                    dense_weight.append(layer.self_attention.dense.weight.data)
                    mlp_l0_weight.append(layer.mlp.dense_h_to_4h.weight.data)
                    mlp_l1_weight.append(layer.mlp.dense_4h_to_h.weight.data)
                    if md.linear_bias:
                        qkv_bias.append(layer.self_attention.query_key_value.bias.data)
                        mlp_l0_bias.append(layer.mlp.dense_h_to_4h.bias.data)

                # Handle gated linear units
                if md.swiglu:
                    # concat all the first halves ('W's) and all the second halves ('V's)
                    for tp_rank in range(tp_size):
                        mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
                    message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
                    message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
                else:
                    message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

                # simple concat of the rest
                message["qkv weight"] = torch.cat(qkv_weight, dim=0)
                message["dense weight"] = torch.cat(dense_weight, dim=1)
                message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
                if md.linear_bias:
                    message["qkv bias"] = torch.cat(qkv_bias, dim=0)
                    if md.swiglu:
                        for tp_rank in range(tp_size):
                            mlp_l0_bias[tp_rank] = torch.chunk(mlp_l0_bias[tp_rank], 2, dim=0)
                        message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias],dim=0)
                        message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias],dim=0)
                    else:
                        message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)

                queue_put(f"transformer layer {total_layer_num}", message)

                total_layer_num = total_layer_num + 1

    # Send final layernorm from tp_rank 0
    message = {
        "weight": models[0].language_model.encoder.final_layernorm.weight.data,
        "bias": models[0].language_model.encoder.final_layernorm.bias.data
    }
    queue_put("final layernorm", message)

    if md.output_layer:
        message = {
            "weight": torch.cat(
                [models[tp_rank].language_model.output_layer.weight.data for tp_rank in range(tp_size)],
                dim = 0)
        }
        queue_put("output layer", message)


    # Send BERT lm head and binary head if it exists
    if md.model_type == 'BERT':
        message = {
            "weight": models[0].language_model.pooler.dense.weight.data,
            "bias": models[0].language_model.pooler.dense.bias.data
        }
        queue_put("pooler", message)

        message = {
            "dense weight": models[0].lm_head.dense.weight.data,
            "dense bias": models[0].lm_head.dense.bias.data,
            "layernorm weight": models[0].lm_head.layernorm.weight.data,
            "layernorm bias": models[0].lm_head.layernorm.bias.data
        }
        queue_put("lm head", message)

        if md.bert_binary_head:
            message = {
                "weight": models[0].binary_head.weight.data,
                "bias": models[0].binary_head.bias.data
            }
            queue_put("binary head", message)
    queue.put("done")

def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise