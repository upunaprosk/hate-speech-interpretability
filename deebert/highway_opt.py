import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

# ---------- utilities ----------
def entropy(logits: torch.Tensor):
    p = torch.softmax(logits, dim=-1)
    return -(p * (p.clamp_min(1e-8)).log()).sum(dim=-1)

class HighwayException(Exception):
    def __init__(self, message, exit_layer: int):
        self.message = message  # e.g., (logits, ...)
        self.exit_layer = exit_layer  # 1-based index


# ---------- highway heads for OPT (decoder-only) ----------
class OPTHighway(nn.Module):
    """
    Per-layer highway head for OPT. Pools the LAST token (decoder) then linear->tanh->dropout->classifier.
    Uses hidden_size (decoder internal width), not word_embed_proj_dim.
    """
    def __init__(self, config):
        super().__init__()
        hid = config.hidden_size
        self.pooler = nn.Linear(hid, hid)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", getattr(config, "dropout", 0.1)))
        self.classifier = nn.Linear(hid, config.num_labels)

    def forward(self, hidden_states: torch.Tensor):
        # decoder: use last token representation
        last_token = hidden_states[:, -1, :]         # (B, H)
        pooled = self.act(self.pooler(last_token))   # (B, H)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)             # (B, C)
        return logits, pooled


class OPTEncoderHighway(nn.Module):
    """
    Wraps the OPT decoder stack and adds a highway head after each layer.
    Matches BERT highway surface: has set_early_exit_entropy and raises HighwayException on exit.
    """
    def __init__(self, config, decoder):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.layers = decoder.layers  # nn.ModuleList[OPTDecoderLayer]
        self.final_layer_norm = decoder.final_layer_norm

        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.highway = nn.ModuleList([OPTHighway(config) for _ in range(config.num_hidden_layers)])
        self.early_exit_entropy = [-1.0 for _ in range(config.num_hidden_layers)]  # disabled by default

    def set_early_exit_entropy(self, x):
        if isinstance(x, (float, int)):
            self.early_exit_entropy = [float(x) for _ in self.early_exit_entropy]
        else:
            assert len(x) == len(self.early_exit_entropy)
            self.early_exit_entropy = list(map(float, x))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        # Mirror OPTDecoder.forward prologue
        output_attentions = output_attentions or self.output_attentions
        output_hidden_states = output_hidden_states or self.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        if input_ids is not None:
            input_shape = input_ids.size()
            inputs_embeds = self.decoder.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("input_ids or inputs_embeds required.")

        bsz, seq_len = input_shape
        pkv_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        mask_seq_len = pkv_len + seq_len

        if attention_mask is None:
            attention_mask = torch.ones(bsz, mask_seq_len, device=inputs_embeds.device)
        elif attention_mask.shape[1] != mask_seq_len:
            raise ValueError(
                f"attention_mask length {attention_mask.shape[1]} != {mask_seq_len} (past+current)"
            )

        causal_attn_mask = self.decoder._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, pkv_len
        )
        pos_embeds = self.decoder.embed_positions(attention_mask, pkv_len)

        if self.decoder.project_in is not None:
            inputs_embeds = self.decoder.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        # caches & collectors
        all_hidden_states = () if output_hidden_states else None
        all_highway_exits = ()

        # Per-layer loop with highway & (optional) early exit
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_attn_mask,
                layer_head_mask=(head_mask[i] if head_mask is not None else None),
                past_key_value=(past_key_values[i] if past_key_values is not None else None),
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]

            # Highway head on current layer output
            h_logits, _ = self.highway[i](hidden_states)
            if not self.training:
                h_ent = entropy(h_logits).mean()  # batch-mean; simple/global stopping
                all_highway_exits += ((h_logits, hidden_states, h_ent),)

                if h_ent.item() < self.early_exit_entropy[i]:
                    # Pack minimal payload (logits + collectors) like BERT code does
                    new_output = (h_logits,)
                    if output_hidden_states:
                        new_output += (all_hidden_states,)
                    # raise to signal early exit at (i+1)
                    raise HighwayException(new_output, i + 1)
            else:
                all_highway_exits += ((h_logits, hidden_states),)

        # Final norm/projection like OPT
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.decoder.project_out is not None:
            hidden_states = self.decoder.project_out(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, all_hidden_states, all_highway_exits


class OPTForSequenceClassificationHighway(OPTPreTrainedModel):
    """
    Decoder-only OPT with per-layer highway exits (names/APIs aligned with BERT highway code).
    Returns (loss), logits, (hidden_states), (attentions=None), ((orig_entropy, highway_entropies), exit_layer).
    """
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.decoder = OPTDecoder(config)                 # reuse HF decoder
        self.encoder = OPTEncoderHighway(config, self.decoder)

        # final classifier for "no early-exit" path; use *word_embed_proj_dim* since decoder projects out
        out_dim = config.word_embed_proj_dim
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", getattr(config, "dropout", 0.1)))
        self.classifier = nn.Linear(out_dim, config.num_labels)

        self.post_init()

    # keep same API as BERT highway model
    def set_early_exit_entropy(self, x):
        self.encoder.set_early_exit_entropy(x)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        output_layer: int = -1,
        train_highway: bool = False,
        output_hidden_states: bool = None,
        output_attentions: bool = None,
        use_cache: bool = None,
        return_dict: bool = True,
    ):
        exit_layer = self.config.num_hidden_layers
        # Try early exit inside encoder
        try:
            hidden_states, all_hidden, all_highways = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache or False,
                output_attentions=output_attentions or False,
                output_hidden_states=output_hidden_states or False,
            )
            # final logits (no exit)
            last_token = hidden_states[:, -1, :]
            logits = self.classifier(self.dropout(last_token))
            outputs = (logits, all_hidden, None, all_highways)  # align tuple positions
        except HighwayException as e:
            # Use highway logits from the exit layer
            outputs = e.message
            exit_layer = e.exit_layer
            logits = outputs[0]  # highway logits

        # Compute entropies for reporting
        if not self.training:
            orig_entropy = entropy(logits).mean()
            highway_ents = []
            for hx in outputs[-1]:  # all_highways
                if len(hx) == 3:
                    highway_ents.append(hx[2].item())
            entropies = (orig_entropy.item(), highway_ents)
        else:
            entropies = None

        # Loss (optionally train highway heads)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if train_highway and outputs[-1] is not None:
                h_losses = []
                for hx in outputs[-1]:  # (logits, hidden[, entropy])
                    h_logits = hx[0]
                    if self.num_labels == 1:
                        loss_fct = MSELoss()
                        h_losses.append(loss_fct(h_logits.view(-1), labels.view(-1)))
                    else:
                        loss_fct = CrossEntropyLoss()
                        h_losses.append(loss_fct(h_logits.view(-1, self.num_labels), labels.view(-1)))
                if len(h_losses) > 1:
                    loss = sum(h_losses[:-1])  # exclude last-layer highway

        # Match BERT-highway return structure
        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": outputs[1],
                "attentions": None,
                "entropies": entropies,     # (orig, [per-layer highway])
                "exit_layer": exit_layer,   # 1..num_layers
            }
        else:
            return (loss, logits, outputs[1], None, entropies, exit_layer)
