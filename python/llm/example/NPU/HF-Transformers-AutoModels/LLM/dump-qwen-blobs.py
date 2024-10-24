import torch
from ipex_llm.transformers.npu_models.qwen2_mp import LowBitQwenMultiDecoderlayer
import uuid
import numpy as np
from ipex_llm.transformers.npu_model import AutoModelForCausalLM
from openvino.runtime import Core, serialize
import os
from ipex_llm.transformers.npu_models.kv import DynamicFusedNormalCache
from ipex_llm.transformers.npu_models.mp_models_base import LLMBaseNNFactory
from typing import Sequence
from intel_npu_acceleration_library.backend.factory import NNFactory


def update_names_of_IR_and_export_blob(xml_path, new_ir_path=None, blob_path=None):
    core = Core()
    core.set_property("NPU",
                      {"NPU_COMPILATION_MODE_PARAMS": "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add"})
    core.set_property("NPU", {"PERFORMANCE_HINT": "LATENCY"})

    model = core.read_model(xml_path)
    inputs = model.inputs
    for idx, input in enumerate(inputs):
        # print(f"Input {idx} : {input.names}, shape: {input.shape}, precision: {input.element_type}")
        if len(input.names) == 0:
            model.inputs[idx].set_names({f"input_{idx}"})
    outputs = model.outputs
    for idx, input in enumerate(outputs):
        #print(f"Output {idx}: {input.names}, shape: {input.shape}, precision: {input.element_type}")
        if len(input.names) == 0:
            model.outputs[idx].set_names({f"output_{idx}"})
    # rewrite this model to a new IR path
    if new_ir_path is not None:
        serialize(model, new_ir_path)

    if blob_path is not None:
        compiledModel = core.compile_model(model, device_name="NPU")
        model_stream = compiledModel.export_model()
        with open(blob_path, 'wb') as f:
            f.write(model_stream)


class LowBitQwenLMHead(LLMBaseNNFactory):
    def __init__(
        self,
        # batch_size: int,
        # seq_len: int,
        # hidden_size: int,
        hidden_shape: Sequence[int],
        *shapes,
        num_heads: int,
        num_key_value_heads: int,
        mode: str = "decode",
        dtype: np.dtype = np.int8,
        max_seq_len: int = 1024,
        transpose_value: bool = False,
        profile: bool = False,
        device: str = "NPU",
        rms_norm_eps,
        model_norm_weight,
        vocab_size,
    ):
        super().__init__(max_seq_len=max_seq_len,
                         transpose_value=transpose_value,
                         dtype=dtype,
                         profile=profile,
                         device=device)
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.batch_size, self.seq_len, self.hidden_size = hidden_shape
        self.mode = mode
        self.rms_norm_eps = rms_norm_eps
        self.transpose_value = transpose_value
        self.vocab_size = vocab_size

        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads

        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # define input, the order self.parameter matters
        input = self.create_input_op((self.batch_size, self.seq_len, self.hidden_size))

        hidden_states = input

        # model norm and lm head
        model_norm_weight = self.constant(model_norm_weight)
        hidden_states = self.layer_norm(hidden_states, model_norm_weight)
        hidden_states = self.linear(
            hidden_states, self.vocab_size, self.hidden_size, bias=False, wt_dtype=self.dtype
        )

        # define outputs
        hidden_states = self.convert_to_fp32(hidden_states)

        print("start compiling")
        self.compile()


class QwenEmbedding(NNFactory):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        padding_idx,
        dtype,  # fp16
        device: str = "NPU",
    ):
        super().__init__(False, device)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.dtype = dtype

        # define input
        weight = self.parameter((vocab_size, embedding_dim))
        input = self.parameter((1, 1), dtype=np.int32)

        if padding_idx == -1:
            padding_idx += vocab_size

        if padding_idx is not None:
            masked_embeddings = np.ones(weight.shape, dtype='int64')
            masked_embeddings[padding_idx, :] = 0  # mask

            node_mask = self.constant(masked_embeddings)
            node_masked_w = self.matmul(weight, node_mask, False, True)

        axis_node = self.constant(np.array([0], dtype=np.int64))
        res = self.gather(node_masked_w if padding_idx else weight, input, axis_node, 0)

        # define outputs
        res = self.convert_to_fp16(res)

        print("start compiling")
        self.compile()


if __name__ == "__main__":

    model = AutoModelForCausalLM.from_pretrained(
        r"D:\llm-models\Qwen2-7B-Instruct",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
        load_in_low_bit="sym_int4",
        mixed_precision=True
    )

    weight_dir = os.path.join(".", "model_layer_weights")
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)

    for layer_idx in range(0, 28):
        num_heads = model.model.layers[layer_idx].self_attn.num_heads
        num_key_value_heads = model.model.layers[layer_idx].self_attn.num_key_value_heads
        head_dim = model.model.layers[layer_idx].self_attn.head_dim
        rms_norm_eps = model.config.rms_norm_eps
        intermediate_size = model.config.intermediate_size
        # layer_weights = []
        # input_layer_norm_weights = []
        # post_attn_layernorm_weights = []
        # q_biases = []
        # k_biases = []
        # v_biases = []
        curr_layer = model.model.layers[layer_idx]
        attn_layer = curr_layer.self_attn
        mlp_layer = curr_layer.mlp

        weights = [
            attn_layer.q_proj.weight, attn_layer.q_proj.scale,
            attn_layer.k_proj.weight, attn_layer.k_proj.scale,
            attn_layer.v_proj.weight, attn_layer.v_proj.scale,
            attn_layer.o_proj.weight, attn_layer.o_proj.scale,
            mlp_layer.gate_proj.weight, mlp_layer.gate_proj.scale,
            mlp_layer.up_proj.weight, mlp_layer.up_proj.scale,
            mlp_layer.down_proj_0.weight, mlp_layer.down_proj_0.scale,
            mlp_layer.down_proj_1.weight, mlp_layer.down_proj_1.scale,
        ]

        q_bias = attn_layer.q_proj.bias.to(torch.float16)   # Seems already fp16
        k_bias = attn_layer.k_proj.bias.to(torch.float16)
        v_bias = attn_layer.v_proj.bias.to(torch.float16)
        cached_cos = curr_layer.self_attn.rotary_emb.cos_cached.to(torch.float16)
        cached_sin = curr_layer.self_attn.rotary_emb.sin_cached.to(torch.float16)
        layer_norm_0 = curr_layer.input_layernorm.weight.to(torch.float16)
        layer_norm_1 = curr_layer.post_attention_layernorm.weight.to(torch.float16)

        # layer_weights.extend(weights)
        # q_biases.append(q_bias)
        # k_biases.append(k_bias)
        # v_biases.append(v_bias)
        # input_layer_norm_weights.append(layer_norm_0)
        # post_attn_layernorm_weights.append(layer_norm_1)

        # parameters = layer_weights
        # op_parameters = []
        # for w in parameters:
        #     if isinstance(w, tuple):  # from QuantizedLinear
        #         op_parameters.append((w[0].numpy(), w[1].numpy()))
        #     else:
        #         op_parameters.append(w.to(torch.float16).numpy())
        # op_id = str(uuid.uuid4())
        # if isinstance(parameters[0], tuple):
        np_dtype = np.int8 if weights[0].dtype == torch.int8 else np.uint8
        # else:  # FP16 Linear
        #     np_dtype = np.float16

        if layer_idx == 0:
            single_decoder = LowBitQwenMultiDecoderlayer(
                [1, 1, num_heads * head_dim],
                input_layernorm_weights=None,
                post_attn_layernorm_weights=None,
                q_biases=None,
                k_biases=None,
                v_biases=None,
                cached_cos=cached_cos,
                cached_sin=cached_sin,
                num_heads=num_heads, 
                num_key_value_heads=num_key_value_heads,
                num_layers=1,
                max_seq_len=1023,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
                mode="decode",
                transpose_value=True,
                dtype=np_dtype,
            )
            # save IR for current Decoder Layer 0
            xml_path = "Qwen2-7B-rest.xml"
            single_decoder.save(xml_path)
            new_ir_path = "Qwen2-7B-rest-new.xml"
            blob_path = "Qwen2-7B-rest.blob"
            # update names for IR and export to a blob
            update_names_of_IR_and_export_blob(xml_path, new_ir_path, blob_path)
            # single_decoder.set_weights(op_id, op_parameters[0 * 7: 1 * 7])

        # save weights bins files
        weight_numpy = [weight.data.numpy() for weight in weights]

        # 0, 1, 2 are input_embed/attention_mask/position_id
        # 3, 4 are layernorms
        input_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_3.bin")  # input layernorm
        post_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_4.bin")  # post layernorm
        layer_norm_0.data.numpy().tofile(input_lm_bin_file)
        layer_norm_1.data.numpy().tofile(post_lm_bin_file)
        # 5, 6, 7 are biases
        q_bias_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_5.bin")
        k_bias_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_6.bin")
        v_bias_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_7.bin")
        q_bias.data.numpy().tofile(q_bias_bin_file)
        k_bias.data.numpy().tofile(k_bias_bin_file)
        v_bias.data.numpy().tofile(v_bias_bin_file)
        # 8, 9 are past k/v

        for idx, weight in enumerate(weight_numpy):
            bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{10+idx}.bin")
            weight.tofile(bin_file)

    model_norm = model.model.norm
    lm_head = model.lm_head.lm_heads[0]   # After adding optimize_pre and mixed_precision, becomes SlicedLMHead with only one slice

    num_heads = model.model.layers[0].self_attn.num_heads
    num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    rms_norm_eps = model.config.rms_norm_eps
    vocab_size = model.config.vocab_size

    weights = [(lm_head.weight, lm_head.scale),]
    parameters = weights
    op_parameters = []
    # for w in parameters:
    #     if isinstance(w, tuple):  # from QuantizedLinear
    #         op_parameters.append((w[0].numpy(), w[1].numpy()))
    #     else:
    #         op_parameters.append(w.to(torch.float16).numpy())
    # op_id = str(uuid.uuid4())
    if isinstance(parameters[0], tuple):
        np_dtype = np.int8 if parameters[0][0].dtype == torch.int8 else np.uint8
    else:  # FP16 Linear
        np_dtype = np.float16

    new_lm_head = LowBitQwenLMHead(
        [1, 1, num_heads * head_dim],
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        max_seq_len=1023,
        rms_norm_eps=rms_norm_eps,
        mode="decode",
        transpose_value=False,
        dtype=np_dtype,
        model_norm_weight=model_norm.weight.to(torch.float16),
        vocab_size=vocab_size,
    )
    print('success')

    # save IR for model norm and lm head
    xml_path = "Qwen2-7B-lm_head.xml"
    new_lm_head.save(xml_path)
    new_ir_path = "Qwen2-7B-lm_head_new.xml"
    blob_path = "Qwen2-7B-lm_head.blob"
    # update names for IR and export to a blob
    update_names_of_IR_and_export_blob(xml_path, new_ir_path, blob_path)

    # save weights bins files
    weight_numpy = [
        lm_head.weight.data.numpy(), lm_head.scale.data.numpy(),
    ]

    # input 0 is hidden states
    for idx, weight in enumerate(weight_numpy):
        bin_file = os.path.join(weight_dir, f"model_lm_head_input_{1+idx}.bin")
        weight.tofile(bin_file)

    embedding_layer = model.model.embed_tokens
    new_embedding = QwenEmbedding(
        vocab_size=model.config.vocab_size,
        embedding_dim=model.config.hidden_size,
        padding_idx=model.config.pad_token_id,
        dtype=np.float16,
    )

    # save IR for current embedding
    xml_path = "Qwen2-7B-embedding.xml"
    new_embedding.save(xml_path)
    new_ir_path = "Qwen2-7B-embedding_new.xml"
    blob_path = "Qwen2-7B-embedding.blob"
    # update names for IR and export to a blob
    update_names_of_IR_and_export_blob(xml_path, new_ir_path, blob_path)

    # save weights bins files
    weight_numpy = [
        embedding_layer.weight.to(torch.float16).detach().numpy()
    ]
    for idx, weight in enumerate(weight_numpy):
        bin_file = os.path.join(weight_dir, f"model_embedding_input_{0+idx}.bin")
        weight.tofile(bin_file)
