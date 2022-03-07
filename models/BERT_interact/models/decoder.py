"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np
import copy

from models.encoder import PositionalEncoding
from models.neural import MultiHeadedAttention, PositionwiseFeedForward, DecoderState

MAX_SIZE = 5000


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerDecoderLayer, self).__init__()


        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.cross_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        #inter_mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)
        #self.register_buffer('inter_mask', inter_mask)
        self.activate = nn.Tanh()
        self.gate = nn.Linear(3*d_model, 1, bias=False)
        #self.mid_gate = nn.Linear(2 * d_model, 1, bias=False)
        self.mlp = nn.Linear(3 * d_model, d_model)
        #self.mlp = nn.Linear(2 * d_model, d_model)

    def self_att(self, inputs, tgt_pad_mask, previous_input=None, layer_cache=None, step=None):
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                            :tgt_pad_mask.size(1)], 0)
        # print(dec_mask)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        query_self, _ = self.self_attn(all_input, all_input, input_norm,
                                    mask=dec_mask,
                                    layer_cache=layer_cache,
                                    type="self")
        query_self = self.drop(query_self) + inputs
        query_norm = self.layer_norm_2(query_self)
        return query_norm

    # def forward(self, inputs, memory_bank, inter_memory_bank, src_pad_mask, tgt_pad_mask, inter_pad_mask,
    #             merge_type, inter_weight, previous_input=None, layer_cache=None, inter_layer_cache=None, step=None):
    #     """
    #     Args:
    #         inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
    #         memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
    #         src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
    #         tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`
    #
    #     Returns:
    #         (`FloatTensor`, `FloatTensor`, `FloatTensor`):
    #
    #         * output `[batch_size x 1 x model_dim]`
    #         * attn `[batch_size x 1 x src_len]`
    #         * all_input `[batch_size x current_step x model_dim]`
    #
    #     """
    #     dec_mask = torch.gt(tgt_pad_mask +
    #                         self.mask[:, :tgt_pad_mask.size(1),
    #                                   :tgt_pad_mask.size(1)], 0)
    #     #print(dec_mask)
    #     if inter_memory_bank is not None:
    #         inter_dec_mask = torch.gt(inter_pad_mask +
    #                             self.mask[:, :inter_pad_mask.size(1),
    #                             :inter_pad_mask.size(1)], 0)
    #         inter_input_norm = self.layer_norm_1(inter_memory_bank)
    #         #print(inter_dec_mask)
    #     input_norm = self.layer_norm_1(inputs)
    #     all_input = input_norm
    #     if previous_input is not None:
    #         all_input = torch.cat((previous_input, input_norm), dim=1)
    #         dec_mask = None
    #
    #     query_self = self.self_attn(all_input, all_input, input_norm,
    #                                  mask=dec_mask,
    #                                  layer_cache=layer_cache,
    #                                  type="self")
    #
    #     # modify cache option here
    #     if inter_memory_bank is not None:
    #         query_cross = self.cross_attn(inter_input_norm, inter_input_norm, input_norm,
    #                                      mask=inter_dec_mask,
    #                                      layer_cache=inter_layer_cache,
    #                                      type="role")
    #         # TODO: parameter setting / gate mechanism?
    #         if merge_type == 'tanh':
    #             query = query_self + self.activate(query_cross) * inter_weight
    #         elif merge_type == 'linear':
    #             query = query_self + query_cross * inter_weight
    #         elif merge_type == 'gate':
    #             cat_query = torch.cat([query_self, query_cross], dim=-1).view(-1, query_self.shape[-1] * 2)
    #             query = query_self + query_cross * nn.Sigmoid()(self.gate(cat_query)).view(query_self.shape[0], query_self.shape[1], 1)
    #         else:
    #             assert True
    #         query = self.drop(query) + inputs
    #     else:
    #         query = self.drop(query_self) + inputs
    #     #query = self.drop(query_self) + inputs
    #
    #     query_norm = self.layer_norm_2(query)
    #     mid = self.context_attn(memory_bank, memory_bank, query_norm,
    #                                   mask=src_pad_mask,
    #                                   layer_cache=layer_cache,
    #                                   type="context")
    #     output = self.feed_forward(self.drop(mid) + query)
    #
    #     return output, all_input
    #     # return output

    # modified decoder layer
    def forward(self, inputs, memory_bank, inter_memory_bank, src_pad_mask, tgt_pad_mask, inter_pad_mask,
                merge_type, inter_weight, role_mask, previous_input=None, layer_cache=None, inter_layer_cache=None, step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0)
        #print(dec_mask)
        if inter_memory_bank is not None:
            inter_dec_mask = torch.gt(inter_pad_mask +
                                self.mask[:, :inter_pad_mask.size(1),
                                :inter_pad_mask.size(2)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        query_self, _ = self.self_attn(all_input, all_input, input_norm,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     type="self")
        query_self = self.drop(query_self) + inputs
        query_norm = self.layer_norm_2(query_self)


        src_pad_mask_user = src_pad_mask | role_mask.ne(1).unsqueeze(1).expand(role_mask.shape[0], src_pad_mask.shape[1], role_mask.shape[1])
        mid_user, user_score = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask_user,
                                      layer_cache=layer_cache,
                                      type="context")
        src_pad_mask_agent = src_pad_mask | role_mask.ne(2).unsqueeze(1).expand(role_mask.shape[0], src_pad_mask.shape[1], role_mask.shape[1])
        mid_agent, agent_score= self.context_attn(memory_bank, memory_bank, query_norm,
                                     mask=src_pad_mask_agent,
                                     layer_cache=layer_cache,
                                     type="context")

        # mid = self.context_attn(memory_bank, memory_bank, query_norm,
        #                              mask=src_pad_mask,
        #                              layer_cache=layer_cache,
        #                              type="context")
        # mid_query = torch.cat([mid_user, mid_agent], dim=-1).view(-1, mid_user.shape[-1] * 2)
        # mid_gate = nn.Sigmoid()(self.mid_gate(mid_query)).view(mid_user.shape[0], mid_user.shape[1], 1)
        # mid = mid_user * mid_gate + mid_agent * (1 - mid_gate)

        # modify cache option here
        if inter_memory_bank is not None:
            query_cross, _ = self.cross_attn(inter_memory_bank, inter_memory_bank, query_norm,
                                          mask=inter_dec_mask,
                                          layer_cache=inter_layer_cache,
                                          type="role")

            if merge_type == 'tanh':
                mid = mid + self.activate(query_cross) * inter_weight
            elif merge_type == 'linear':
                mid = mid + query_cross * inter_weight
            elif merge_type == 'gate':
                cat_query = torch.cat([mid_user, mid_agent, query_cross], dim=-1).view(-1, mid_user.shape[-1] * 3)
                # mid = mid + query_cross * nn.Sigmoid()(self.gate(cat_query)).view(mid_user.shape[0],
                #                                                                            mid_user.shape[1], 1)
                # cat_query = torch.cat([mid_user, mid_agent, query_cross], dim=-1).view(-1, mid_user.shape[-1] * 3)
                mid = self.mlp(cat_query).view(mid_user.shape[0], mid_user.shape[1], -1)
                # cat_query = torch.cat([mid, query_cross], dim=-1).view(-1, mid.shape[-1] * 2)
                # mid = self.mlp(cat_query).view(mid.shape[0], mid.shape[1], -1)

            else:
                assert True


        output = self.feed_forward(self.drop(mid) + query_self)

        return output, all_input, user_score, agent_score

    def _get_attn_subsequent_mask(self, size, k=1):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=k).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask




class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
       attn_type (str): if using a seperate copy attention
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout,self.embeddings.embedding_dim)


        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, memory_bank, inter_memory_bank, state, memory_lengths=None,
                step=None, cache=None,memory_masks=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """

        src_words = state.src
        tgt_words = tgt
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Run the forward pass of the TransformerDecoder.
        # emb = self.embeddings(tgt, step=step)
        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = self.pos_emb(emb, step)

        src_memory_bank = memory_bank
        padding_idx = self.embeddings.padding_idx
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        if (not memory_masks is None):
            src_len = memory_masks.size(-1)
            src_pad_mask = memory_masks.expand(src_batch, tgt_len, src_len)

        else:
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, tgt_len, src_len)

        if state.cache is None:
            saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            output, all_input, user_score, agent_score \
                = self.transformer_layers[i](
                    output, src_memory_bank, inter_memory_bank,
                    src_pad_mask, tgt_pad_mask, None,
                    None, None,
                    previous_input=prev_layer_input,
                    layer_cache=state.cache["layer_{}".format(i)]
                    if state.cache is not None else None,
                    step=step)
            if state.cache is None:
                saved_inputs.append(all_input)

        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)

        # Process the result and update the attentions.

        if state.cache is None:
            state = state.update_state(tgt, saved_inputs)

        return output, state, user_score, agent_score

    def init_decoder_state(self, src, memory_bank,
                           with_cache=False):
        """ Init decoder state """
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state

class RoleDecoder(nn.Module):
    def __init__(self, user_decoder, agent_decoder):
        super(RoleDecoder, self).__init__()

        # Basic attributes.
        self.user_decoder = user_decoder
        self.agent_decoder = agent_decoder

    def forward(self, tgt_user, tgt_agent, memory_bank, state_user, state_agent, merge_type, inter_weight, role_mask,
                memory_lengths=None, step=None, cache=None, memory_masks=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """

        src_words = state_user.src
        tgt_words_user = tgt_user
        tgt_words_agent = tgt_agent
        src_batch, src_len = src_words.size()
        tgt_batch_user, tgt_len_user = tgt_words_user.size()
        tgt_batch_agent, tgt_len_agent = tgt_words_agent.size()
        src_memory_bank = memory_bank
        user_scores, agent_scores = [[], []], [[], []]

        # user:
        emb_user = self.user_decoder.embeddings(tgt_user)
        assert emb_user.dim() == 3  # len x batch x embedding_dim

        output_user = self.user_decoder.pos_emb(emb_user, step)
        padding_idx = self.user_decoder.embeddings.padding_idx
        tgt_pad_mask_user = tgt_words_user.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch_user, tgt_len_user, tgt_len_user)
        inter_pad_mask_user = tgt_words_user.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch_user, tgt_len_agent, tgt_len_user)

        if (not memory_masks is None):
            src_len = memory_masks.size(-1)
            src_pad_mask_user = memory_masks.expand(src_batch, tgt_len_user, src_len)
        else:
            src_pad_mask_user = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, tgt_len_user, src_len)

        if state_user.cache is None:
            saved_inputs_user = []

        # agent:
        emb_agent = self.agent_decoder.embeddings(tgt_agent)
        assert emb_agent.dim() == 3  # len x batch x embedding_dim

        output_agent = self.agent_decoder.pos_emb(emb_agent, step)
        padding_idx = self.agent_decoder.embeddings.padding_idx
        tgt_pad_mask_agent = tgt_words_agent.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch_agent, tgt_len_agent, tgt_len_agent)
        inter_pad_mask_agent = tgt_words_agent.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch_agent, tgt_len_user, tgt_len_agent)

        if (not memory_masks is None):
            src_len = memory_masks.size(-1)
            src_pad_mask_agent = memory_masks.expand(src_batch, tgt_len_agent, src_len)
        else:
            src_pad_mask_agent = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, tgt_len_agent, src_len)

        if state_agent.cache is None:
            saved_inputs_agent = []

        kl_mask_user = tgt_words_user.data.ne(padding_idx).unsqueeze(2).expand(src_batch, tgt_len_user, src_len)
        kl_mask_agent = tgt_words_agent.data.ne(padding_idx).unsqueeze(2).expand(src_batch, tgt_len_user, src_len)  # (tgt_batch_user, tgt_len_user)

        # loop
        # TODO: need to modify into output_users and output_agents
        # memory_bank_user, memory_bank_agent = None, None
        memory_bank_user, memory_bank_agent = output_user, output_agent
        for i in range(self.user_decoder.num_layers):
            prev_layer_input_user, prev_layer_input_agent = None, None

            # print(i)
            # print('----------------------------')
            # print(state_user.cache["layer_{}".format(i)])
            # print(state_agent.cache["layer_{}".format(i)])
            # if state_user.cache["layer_{}".format(i)]['self_keys'] is not None:
            #     print(state_user.cache["layer_{}".format(i)]['self_keys'].shape)
            # if state_agent.cache["layer_{}".format(i)]['self_keys'] is not None:
            #     print(state_agent.cache["layer_{}".format(i)]['self_keys'].shape)

            # calculate user layer
            if state_user.cache is None:
                if state_user.previous_input is not None:
                    prev_layer_input_user = state_user.previous_layer_inputs[i]
            # TODO: change the cache layer into others
            self_query_user = self.user_decoder.transformer_layers[i].self_att(output_user, tgt_pad_mask_user, previous_input=prev_layer_input_user,
                    layer_cache=state_user.cache["layer_{}".format(i)]
                    if state_user.cache is not None else None, step=step)

            self_query_agent = self.agent_decoder.transformer_layers[i].self_att(output_agent, tgt_pad_mask_agent,
                                                                               previous_input=prev_layer_input_agent,
                                                                               layer_cache=state_agent.cache[
                                                                                   "layer_{}".format(i)]
                                                                               if state_agent.cache is not None else None,
                                                                               step=step)

            output_user, all_input_user, user_score_u, user_score_a \
                = self.user_decoder.transformer_layers[i](
                    output_user, src_memory_bank, self_query_agent,
                    src_pad_mask_user, tgt_pad_mask_user, inter_pad_mask_agent,
                    merge_type, inter_weight, role_mask,
                    previous_input=prev_layer_input_user,
                    layer_cache=state_user.cache["layer_{}".format(i)]
                    if state_user.cache is not None else None,
                    inter_layer_cache=state_agent.cache["layer_{}".format(i)]
                    if state_agent.cache is not None else None,
                    step=step)
            user_scores[0].append(torch.mean(user_score_u, dim=1))
            agent_scores[0].append(torch.mean(user_score_a, dim=1))
            if state_user.cache is None:
                saved_inputs_user.append(all_input_user)

            # calculate agent layer
            if state_agent.cache is None:
                if state_agent.previous_input is not None:
                    prev_layer_input_agent = state_agent.previous_layer_inputs[i]
            output_agent, all_input_agent, agent_score_u, agent_score_a \
                = self.agent_decoder.transformer_layers[i](
                output_agent, src_memory_bank, self_query_user,
                src_pad_mask_agent, tgt_pad_mask_agent, inter_pad_mask_user,
                merge_type, inter_weight, role_mask,
                previous_input=prev_layer_input_agent,
                layer_cache=state_agent.cache["layer_{}".format(i)]
                if state_agent.cache is not None else None,
                inter_layer_cache=state_user.cache["layer_{}".format(i)]
                if state_user.cache is not None else None,
                step=step)
            user_scores[1].append(torch.mean(user_score_u, dim=1))
            agent_scores[1].append(torch.mean(user_score_a, dim=1))
            if state_agent.cache is None:
                saved_inputs_agent.append(all_input_agent)

            # update inter memory
            memory_bank_user = output_user.clone()
            memory_bank_agent = output_agent.clone()


        if state_user.cache is None:
            saved_inputs_user = torch.stack(saved_inputs_user)
        if state_agent.cache is None:
            saved_inputs_agent = torch.stack(saved_inputs_agent)

        output_user = self.user_decoder.layer_norm(output_user)
        output_agent = self.agent_decoder.layer_norm(output_agent)

        # Process the result and update the attentions.

        if state_user.cache is None:
            state_user = state_user.update_state(tgt_user, saved_inputs_user)
        if state_agent.cache is None:
            state_agent = state_agent.update_state(tgt_agent, saved_inputs_agent)

        return output_user, output_agent, state_user, state_agent, user_scores, agent_scores, kl_mask_user, kl_mask_agent


class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if (self.previous_input is not None
                and self.previous_layer_inputs is not None):
            return (self.previous_input,
                    self.previous_layer_inputs,
                    self.src)
        else:
            return (self.src,)

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        self.cache = {}

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None
            }
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            layer_cache["inter_keys"] = None
            layer_cache["inter_values"] = None
            self.cache["layer_{}".format(l)] = layer_cache

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 0)
        if self.cache is not None:
            _recursive_map(self.cache)



