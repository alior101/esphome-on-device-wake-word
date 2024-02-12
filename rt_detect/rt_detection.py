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

def generate_features_for_clip(clip):
    micro_frontend = frontend_op.audio_microfrontend(
        tf.convert_to_tensor(clip),
        sample_rate=16000,
        window_size=30,
        window_step=20,
        num_channels=40,
        upper_band_limit=7500,
        lower_band_limit=125,
        enable_pcan=True,
        min_signal_remaining=0.05,
        out_scale=1,
        out_type=tf.float32)
    output = tf.multiply(micro_frontend, 0.0390625)
    return output.numpy()

def features_generator(generator):
    for data in generator:
        for clip in data:
            yield generate_features_for_clip(clip)

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


# def tflite_predict(model,x):
#     if x.shape[1] != 1280:
#         model.resize_tensor_input(0, [1, x.shape[1]], strict=True)  # initialize with fixed input size
#         model.allocate_tensors()
#         input_size = x.shape[1]
#     elif input_size != 1280:
#         model.resize_tensor_input(0, [1, 1280], strict=True)  # initialize with fixed input size
#         model.allocate_tensors()
#         input_size = 1280

#     model.set_tensor(melspec_input_index, x)
#     model.invoke()
#     return model.get_tensor(output_index)



# Define function to process audio
 
#def process_audio(audio, state=collections.defaultdict(partial(collections.deque, maxlen=60))):
def process_audio(audio, state=list[np.float32]):
    # Resample audio to 16khz if needed
    if audio[0] != 16000:
        data = scipy.signal.resample(audio[1], int(float(audio[1].shape[0])/audio[0]*16000))
    
    data = data.astype(np.int16)
    print(data.shape)
    res = generate_features_for_clip(data)
    # Get predictions
    for row in res:
        row1 = row.astype(np.int8)
        row2 = row1.reshape([1,1,40])
        infer_model.set_tensor(input_details[0]['index'], row2)
        infer_model.invoke()
        pred = infer_model.get_tensor(output_details[0]['index'])


        #Fill deque with zeros if it's empty
        if len(state) == 0:
            state.extend(np.zeros(60))
            
        # Add prediction
        state.append(pred[0][0])
        state.pop(0)
        
    
    # Make line plot
    dfs = []
    df = pd.DataFrame({"x": np.arange(len(state)), "y": state, "Model": "wakeword"})
    dfs.append(df)
    
    df = pd.concat(dfs)
    plot = gr.LinePlot(value = df, x='x', y='y', color="Model", y_lim = (0,1), tooltip="Model",
                                width=600, height=300, x_title="Time (frames)", y_title="Model Score", color_legend_position="bottom")
    


    return plot, state


def process_audio1(audio, state=list[np.float32]):
    # Resample audio to 16khz if needed
    if audio[0] != 16000:
        data = scipy.signal.resample(audio[1], int(float(audio[1].shape[0])/audio[0]*16000))
    
    data = data.astype(np.int16)
    print(data.shape)
    res = generate_features_for_clip(data)
    # Get predictions
    for row in res:
        row1 = row.astype(np.int8)
        row2 = row1.reshape([1,1,40])
 
        #Fill deque with zeros if it's empty
        if len(state) == 0:
            state.extend(np.zeros(60))
            
        # Add prediction
        state.append(pred[0][0])
        state.pop(0)
         
    # Make line plot
    dfs = []
    df = pd.DataFrame({"x": np.arange(len(state)), "y": state, "Model": "wakeword"})
    dfs.append(df)
    
    df = pd.concat(dfs)
    plot = gr.LinePlot(value = df, x='x', y='y', color="Model", y_lim = (0,1), tooltip="Model",
                                width=600, height=300, x_title="Time (frames)", y_title="Model Score", color_legend_position="bottom")
    

    
    return plot, state
# Create Gradio interface and launch

desc = """
This is a demo of the pre-trained models included in the latest release
"""

gr_int = gr.Interface(
    title = "openWakeWord Live Demo",
    description = desc,
    css = ".flex {flex-direction: column} .gr-panel {width: 100%}",
    fn=process_audio,
    inputs=[
        gr.Audio(sources=["microphone"], type="numpy", streaming=True, show_label=False),
        gr.State(value=[])
    ],
    outputs=[
        gr.LinePlot(show_label=False),
        gr.State()
    ],
    live=True)

gr_int.launch(share=True)