import gradio as gr
import json
import pandas as pd
import collections
import scipy.signal
import numpy as np
from functools import partial
import sys
import tensorflow as tf
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op 
import matplotlib.pyplot as plt
import altair as alt
import librosa

def generate_features_for_clip1(clip):
    #mfcc = librosa.feature.mfcc(y=clip, sr=16000, n_mfcc=40)
    sample_data = (clip/32767).astype(np.float32)
    S = librosa.feature.melspectrogram(y=sample_data, win_length=int(480), 
                                        hop_length=int(0.020*16000), n_fft=512, center=True,
                                        sr=16000, n_mels=40, fmin=125, fmax=7500, power=2)#, norm=None)

    S = librosa.power_to_db(S).squeeze()[:, 1:-1] 
    return S.T
    
def generate_features_for_clip(clip):
    micro_frontend = frontend_op.audio_microfrontend(
        tf.convert_to_tensor(clip),
        sample_rate=16000,
        window_size=30,
        window_step=20,
        num_channels=40,
        upper_band_limit=7500,
        lower_band_limit=125,
        enable_pcan=False,
        min_signal_remaining=0.05,
        out_scale=1,
        out_type=tf.float32)
    output = tf.multiply(micro_frontend, 0.0390625)
    return output.numpy()


infer_model = tf.lite.Interpreter(model_path="/home/lior/devel/esphome-on-device-wake-word/trained_models/okay_nabu_v5_1/tflite_stream_state_internal_quant/stream_state_internal_quantize.tflite", num_threads=1)
infer_model.resize_tensor_input(0, [1,1,40], strict=True)  # initialize with fixed input size
infer_model.allocate_tensors()
input_details = infer_model.get_input_details()
output_details = infer_model.get_output_details()
print()
print("Input details:")
print(input_details)
print()
print("Output details:")
print(output_details)
print()



# Define function to process audio

detection_state = np.random.rand(100)
#def process_audio(audio, state=collections.defaultdict(partial(collections.deque, maxlen=60))):
def process_audio(audio):
    # Resample audio to 16khz if needed
    if audio[0] != 16000:
        data = scipy.signal.resample(audio[1], int(float(audio[1].shape[0])/audio[0]*16000))
    
    data = data.astype(np.int16)
    
    res = generate_features_for_clip1(data)
    # Get predictions
    for row in res:
        row1 = row.astype(np.int8)
        row2 = row1.reshape([1,1,40])
        infer_model.set_tensor(input_details[0]['index'], row2)
        infer_model.invoke()
        pred = infer_model.get_tensor(output_details[0]['index'])

        # Add prediction
        detection_state[:-1] = detection_state[1:] 
        detection_state[99] = pred[0,0]
        
    # Make line plot
    df = pd.DataFrame({"x": np.arange(len(detection_state)), "y": detection_state, "Model": "wakeword"})
    plot = gr.LinePlot(value = df, x='x', y='y', color="Model", y_lim = (0,1), tooltip="Model",
                                width=600, height=300, x_title="Time (frames)", y_title="Model Score", color_legend_position="bottom")
    


    return plot


features_state = np.random.rand(40,100)

def process_audio1(audio):
    # Resample audio to 16khz if needed
    if audio[0] != 16000:
        data = scipy.signal.resample(audio[1], int(float(audio[1].shape[0])/audio[0]*16000))
    
    data = data.astype(np.int16)
    #print(data.shape)
    res = generate_features_for_clip1(data)
    # Get predictions


    # if state == None:
    #     state = np.random.rand(40,100)

    for row in res:
        row1 = row.astype(np.int8)
        row2 = row1.reshape([40,1])
            
        features_state[:,:-1] = features_state[:,1:] 
        features_state[:,99] = row2[:,0]

        # Add prediction
        # state = np.append(state, row2, 0)
        # state = np.delete(state, 0)
         
    # Convert this grid to columnar data expected by Altair
    x, y = np.meshgrid(np.arange(0,100, 1), range(0, 40))

    source = pd.DataFrame({'x': x.ravel(),
                            'y': y.ravel(),
                            'z': features_state.ravel()})
    return alt.Chart(source).mark_rect().encode(
            x='x:O',
            y='y:O',
            color='z:Q'
        )


# Create Gradio interface and launch

desc = """
This is a demo of the pre-trained models included in the latest release
"""

gr_int_mfcc = gr.Interface(
    title = "openWakeWord Live Demo",
    description = desc,
    css = ".flex {flex-direction: column} .gr-panel {width: 100%}",
    fn=process_audio1,
    inputs=[
        gr.Audio(sources=["microphone"], type="numpy", streaming=True, show_label=True)
    ],
    outputs=[
        gr.Plot(show_label=False)
    ],
    live=True)

gr_int_detect = gr.Interface(
    title = "openWakeWord Live Demo",
    description = desc,
    css = ".flex {flex-direction: column} .gr-panel {width: 100%}",
    fn=process_audio,
    inputs=[
        gr.Audio(sources=["microphone"], type="numpy", streaming=True, show_label=True)
    ],
    outputs=[
        gr.LinePlot(show_label=False)
    ],
    live=True)

#gr_int_mfcc.launch(share=True)
gr_int_detect.launch(share=True)