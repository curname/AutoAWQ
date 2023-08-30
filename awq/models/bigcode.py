from .base import BaseAWQForCausalLM


class BigCodeAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "gpt_bigcode"
    max_new_tokens_key = "n_positions"

    @staticmethod
    def get_model_layers(model):
        return model.transformer.h

    @staticmethod
    def get_act_for_scaling(module):
        # return dict(
        #     is_scalable=False
        # )
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.c_fc.out_features
        )

    @staticmethod
    def move_embed(model, device):
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.drop = model.transformer.drop.to(device)

    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.ln_1,
            layers=[module.attn.c_attn],
            inp=input_feat['attn.c_attn'],
            module2inspect=module.attn,
            kwargs=module_kwargs
        ))

        # attention output
        # layers.append(dict(
        #     prev_op=module.attn.c_attn,
        #     layers=[module.attn.c_proj],
        #     inp=input_feat['attn.c_proj']
        # ))

        # linear 1
        layers.append(dict(
            prev_op=module.ln_2,
            layers=[module.mlp.c_fc],
            inp=input_feat['mlp.c_fc'],
            module2inspect=module.mlp
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.mlp.act,
            layers=[module.mlp.c_proj],
            inp=input_feat['mlp.c_proj']
        ))

        return layers