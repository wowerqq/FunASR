import onnxruntime
import yaml
from typing import Any, Dict, Iterable, List, NamedTuple, Set, Tuple, Union
import soundfile as sf
import os
import torch
import numpy as np
from utils.frontend import WavFrontend
from typing import List, Union, Tuple
from pathlib import Path
from utils.utils import (CharTokenizer, Hypothesis,
                          TokenIDConverter, get_logger,
                          read_yaml)
from utils.postprocess_utils import sentence_postprocess




def decode(am_scores: np.ndarray, token_nums: int) -> List[str]:
    return [decode_one(am_score, token_num)
            for am_score, token_num in zip(am_scores, token_nums)]

def decode_one(
               am_score: np.ndarray,
               valid_token_num: int) -> List[str]:
    yseq = am_score.argmax(axis=-1)
    score = am_score.max(axis=-1)
    score = np.sum(score, axis=-1)

    # pad with mask tokens to ensure compatibility with sos/eos tokens
    # asr_model.sos:1  asr_model.eos:2
    yseq = np.array([1] + yseq.tolist() + [2])
    hyp = Hypothesis(yseq=yseq, score=score)

    # remove sos/eos and get results
    last_pos = -1
    token_int = hyp.yseq[1:last_pos].tolist()

    # remove blank symbol id, which is assumed to be 0
    token_int = list(filter(lambda x: x not in (0, 2), token_int))

    # Change integer-ids to tokens
    token = converter.ids2tokens(token_int)
    token = token[:valid_token_num - pred_bias]
    # texts = sentence_postprocess(token)
    return token


#@staticmethod
def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
    def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
        pad_width = ((0, max_feat_len - cur_len), (0, 0))
        return np.pad(feat, pad_width, 'constant', constant_values=0)

    feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
    feats = np.array(feat_res).astype(np.float32)
    return feats


def extract_feat( waveform_list: List[np.ndarray]
                 ) -> Tuple[np.ndarray, np.ndarray]:
    feats, feats_len = [], []
    for waveform in waveform_list:
        speech, _ = frontend.fbank(waveform)
        feat, feat_len = frontend.lfr_cmvn(speech)
        feats.append(feat)
        feats_len.append(feat_len)

    feats = pad_feats(feats, np.max(feats_len))
    feats_len = np.array(feats_len).astype(np.int32)
    feats = torch.from_numpy(feats).type(torch.float32)
    feats_len = torch.from_numpy(feats_len).type(torch.int32)
    return feats, feats_len


def rec_wav():
    waveform_list = []
    wavformer, sr = sf.read("asr_example.wav")
    waveform_list.append(wavformer)


    feats, feats_len = extract_feat(waveform_list) 
    encoder = {'speech': feats.numpy(), 'speech_lengths': feats_len.numpy() } 
    output_name = ['enc', 'enc_len']
    enc_output = encoder_sess.run(output_name, input_feed=encoder)
    pre_feed = {'speech': enc_output[0], 'speech_lengths': enc_output[1]}
    output_name = ['pre_acoustic_embeds', 'pre_token_length']
    pre_output = prc_sess.run(output_name, input_feed=pre_feed)
    dec_feed = {'speech': pre_output[0], 'speech_lengths': pre_output[1], 'x1':enc_output[0], 'y1':enc_output[1]}
    output_name = ['logits', 'token_num']
    dec_output = decoder_sess.run(output_name, input_feed=dec_feed)
    #print (output[0].shape)
    #print (output[1].shape)
    #for item in output:
    #    print(item)

    #decode(am_scores, valid_token_lens)
    preds = decode(dec_output[0], dec_output[1])
    for pred in preds: 
        pred = sentence_postprocess(pred)
        print (pred)
   
if __name__ == '__main__':
    config_file = os.path.join("onnx1", 'config.yaml')
    config = read_yaml(config_file)
    cmvn_file = os.path.join("onnx1", 'am.mvn')
    pred_bias = config['model_conf']['predictor_bias']
    converter = TokenIDConverter(config['token_list'])
    frontend = WavFrontend(
        cmvn_file=cmvn_file,
        **config['frontend_conf']
    )
    enc_onnx_path = "onnx1/encoder.onnx"
    encoder_sess = onnxruntime.InferenceSession(enc_onnx_path)
    pre_onnx_path = "onnx1/predictor.onnx"
    prc_sess = onnxruntime.InferenceSession(pre_onnx_path)
    dec_onnx_path = "onnx1/decoder.onnx"
    decoder_sess = onnxruntime.InferenceSession(dec_onnx_path)
    rec_wav()

