python -m funasr.export.export_model \
    --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
    --export-dir ./ybs \
    --type onnx \
    --quantize true \
    --model_type encoder  #[encoder, predictor, decoder]
  #  --fallback-num [fallback_num]
mv ybs/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.onnx onnx2/encoder.onnx
python -m funasr.export.export_model \
    --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
    --export-dir ./ybs \
    --type onnx \
    --quantize true \
    --model_type predictor  #[encoder, predictor, decoder]
mv ybs/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.onnx onnx2/predictor.onnx

python -m funasr.export.export_model \
    --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
    --export-dir ./ybs \
    --type onnx \
    --quantize true \
    --model_type decoder  #[encoder, predictor, decoder]
mv ybs/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.onnx onnx2/decoder.onnx
