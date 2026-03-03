# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import logging
import tempfile
import subprocess
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def _collect_pcm_bytes(model_output) -> bytes:
    chunks = []
    for i in model_output:
        chunks.append((i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes())
    return b''.join(chunks)


def _save_upload_as_wav_16k(upload: UploadFile) -> str:
    raw_suffix = os.path.splitext(upload.filename or '')[1].lower()
    input_suffix = raw_suffix if raw_suffix else '.bin'
    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix)
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_in_path = temp_in.name
    temp_out_path = temp_out.name
    temp_in.close()
    temp_out.close()

    try:
        upload.file.seek(0)
        with open(temp_in_path, 'wb') as f:
            f.write(upload.file.read())

        cmd = ['ffmpeg', '-y', '-i', temp_in_path, '-ac', '1', '-ar', '16000', temp_out_path]
        run = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if run.returncode != 0:
            raise HTTPException(status_code=400, detail=f'音频解码失败，请上传可读音频文件。{run.stderr}')
        return temp_out_path
    finally:
        if os.path.exists(temp_in_path):
            os.unlink(temp_in_path)


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    audio_bytes = _collect_pcm_bytes(model_output)
    return Response(content=audio_bytes, media_type='application/octet-stream')


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    temp_wav = _save_upload_as_wav_16k(prompt_wav)
    try:
        model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, temp_wav)
        audio_bytes = _collect_pcm_bytes(model_output)
        return Response(content=audio_bytes, media_type='application/octet-stream')
    finally:
        if os.path.exists(temp_wav):
            os.unlink(temp_wav)


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    temp_wav = _save_upload_as_wav_16k(prompt_wav)
    try:
        model_output = cosyvoice.inference_cross_lingual(tts_text, temp_wav)
        audio_bytes = _collect_pcm_bytes(model_output)
        return Response(content=audio_bytes, media_type='application/octet-stream')
    finally:
        if os.path.exists(temp_wav):
            os.unlink(temp_wav)


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    audio_bytes = _collect_pcm_bytes(model_output)
    return Response(content=audio_bytes, media_type='application/octet-stream')


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    temp_wav = _save_upload_as_wav_16k(prompt_wav)
    try:
        model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, temp_wav)
        audio_bytes = _collect_pcm_bytes(model_output)
        return Response(content=audio_bytes, media_type='application/octet-stream')
    finally:
        if os.path.exists(temp_wav):
            os.unlink(temp_wav)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
