import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertModel, BertPreTrainedModel
from transformers import BertConfig

from .crf import CRF, to_crf_pad, unpad_crf


class BertCRFForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCRFForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.num_labels + 2)

        self.ner_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ner_classifier1 = nn.Linear(config.hidden_size, 2048)
        self.ner_classifier2 = nn.Linear(2048, self.num_labels + 2)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)

        self.crf = CRF(self.num_labels)

        self.init_weights()

    def _get_features(self, input_ids=None, attention_mask=None, token_type_ids=None,
                      position_ids=None, head_mask=None, inputs_embeds=None,
                      output_attentions=None, output_hidden_states=None,
                      return_dict=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict
                            )

        # sequence_output = outputs[0]

        # sequence_output = self.dropout(sequence_output)
        # logits = self.classifier(sequence_output)
        sequence_output = outputs[0]
        sequence_output = self.ner_dropout(sequence_output)
        ner_logits = self.ner_classifier1(sequence_output)
        ner_logits = self.dropout1(ner_logits)
        ner_logits = self.ner_classifier2(ner_logits)

        return ner_logits, outputs

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                pad_token_label_id=-100,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        logits, outputs = self._get_features(input_ids, attention_mask, token_type_ids, position_ids,
                                             head_mask, inputs_embeds, output_attentions,
                                             output_hidden_states, return_dict)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            pad_mask = (labels != pad_token_label_id)

            # Only keep active parts of the loss
            if attention_mask is not None:
                loss_mask = ((attention_mask == 1) & pad_mask)
            else:
                loss_mask = ((torch.ones(logits.shape) == 1) & pad_mask)

            crf_labels, crf_mask = to_crf_pad(labels, loss_mask, pad_token_label_id)
            crf_logits, _ = to_crf_pad(logits, loss_mask, pad_token_label_id)

            loss = self.crf.neg_log_likelihood(crf_logits, crf_mask, crf_labels)
            # removing mask stuff from the output path is done later in my_crf_ner but it should be kept away
            # when calculating loss
            best_path = self.crf(crf_logits, crf_mask)
            best_path = unpad_crf(best_path, crf_mask, labels, pad_mask)
            outputs = (loss,) + outputs + (best_path,)
        else:
            if attention_mask is not None:
                mask = (attention_mask == 1)
            else:
                mask = torch.ones(logits.shape).bool()
            crf_logits, crf_mask = to_crf_pad(logits, mask, pad_token_label_id)
            crf_mask = crf_mask.sum(axis=2) == crf_mask.shape[2]
            best_path = self.crf(crf_logits, crf_mask)
            temp_labels = torch.ones(mask.shape) * pad_token_label_id
            temp_labels = temp_labels.type(best_path.dtype).cuda()
            best_path = unpad_crf(best_path, crf_mask, temp_labels, mask)
            outputs = outputs + (best_path,)

        return outputs



class BertForBilevelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.max_seq_length = config.max_seq_length
        self.final_dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.final_classifier1 = nn.Linear(config.hidden_size + self.max_seq_length * self.num_labels, 2048)
        self.final_dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.final_classifier2 = nn.Linear(2048, self.num_labels - 1)

        self.bert = BertModel(config)

        self.seq_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ner_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier1 = nn.Linear(config.hidden_size, 2048)
        self.ner_classifier2 = nn.Linear(2048, self.num_labels)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                seq_labels=None,
                ner_labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict
                            )

        sequence_output = outputs[0]
        sequence_output = self.ner_dropout(sequence_output)
        ner_logits = self.ner_classifier1(sequence_output)
        ner_logits = self.dropout1(ner_logits)
        ner_logits = self.ner_classifier2(ner_logits)


        pad_pred = torch.zeros([self.num_labels], device=ner_logits.device, dtype=ner_logits.dtype)
        pad_pred[-1] = 1
        ner_logits[attention_mask==0]=pad_pred

        final_input = ner_logits.view([ner_logits.shape[0], -1])
        final_input = torch.cat((outputs[1], final_input), dim=1)

        seq_logits = self.final_dropout1(final_input)
        seq_logits = self.final_classifier1(seq_logits)
        seq_logits = self.final_dropout2(seq_logits)
        seq_logits = self.final_classifier2(seq_logits)

        outputs = (ner_logits,) + (seq_logits,) + outputs[2:]  # add hidden states and attention if they are here
        if ner_labels is not None:
            # calculate ner loss
            ner_loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = ner_logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, ner_labels.view(-1), torch.tensor(ner_loss_fct.ignore_index).type_as(ner_labels)
                )
                ner_loss = ner_loss_fct(active_logits, active_labels)
            else:
                ner_loss = ner_loss_fct(ner_logits.view(-1, self.num_labels), ner_labels.view(-1))

            # calculate sequence classification loss
            seq_loss_fct = BCEWithLogitsLoss()
            seq_loss = seq_loss_fct(seq_logits, seq_labels[:, :-1])
            loss = ner_loss + seq_loss

            outputs = (loss,) + outputs

        return outputs


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.seq_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier1 = nn.Linear(config.hidden_size, 2048)
        self.ner_classifier2 = nn.Linear(2048, self.num_labels)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict
                            )

        pooled_output = outputs[1]
        pooled_output = self.seq_dropout(pooled_output)
        seq_logits = self.ner_classifier1(pooled_output)
        seq_logits = self.dropout2(seq_logits)
        seq_logits = self.ner_classifier2(seq_logits)

        outputs = (seq_logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            # calculate sequence classification loss
            seq_loss_fct = BCEWithLogitsLoss()
            loss = seq_loss_fct(seq_logits, labels)

            outputs = (loss,) + outputs

        return outputs


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.ner_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.ner_classifier1 = nn.Linear(config.hidden_size, 2048)
        self.ner_classifier2 = nn.Linear(2048, self.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # sequence_output = self.dropout(sequence_output)
        # logits = self.classifier(sequence_output)

        sequence_output = self.ner_dropout(sequence_output)
        ner_logits = self.ner_classifier1(sequence_output)
        ner_logits = self.dropout1(ner_logits)
        ner_logits = self.ner_classifier2(ner_logits)

        output = (ner_logits,) + outputs[2:]
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = ner_logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(ner_logits.view(-1, self.num_labels), labels.view(-1))

        return ((loss,) + output) if loss is not None else output
