# To run: streamlit run streamlit_quran_app.py

from datetime import datetime

import logging
import logging.handlers
import queue
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import List
import io
from scipy.io import wavfile
from PIL import Image

import av
import numpy as np
import pydub
import streamlit as st

from RNN_model import *
from postprocessing import *
from mistakes_detection import *

from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


torch.manual_seed(7)
device = torch.device("cpu")
def main():
    im = Image.open("logo/app_logo.jpg")
    st.set_page_config(
        page_title="Quran Recitation Recognition",
        page_icon=im
    )
    col1, mid, col2 = st.columns([20, 1, 15])
    with col1:
        # st.header("Quran Recitation Recognition and Mistakes Detection")
        st.markdown("""<h1 style='text-align: justify;'> 
        Quran Recitation Recognition and Mistakes Detection
        </h1>""",unsafe_allow_html=True)
        st.markdown("""<p style='text-align: justify;'> 
        This is a demo app of Quran recitation recognition and mistakes detection project by Sarah Alrumiah. 
        This demo uses deep learning models trained with expert reciters' recitations.
        </p>""", unsafe_allow_html=True)
        st.markdown("""<p style='text-align: justify;'> 
                To use it, please select a verse from the drop down menu, than choose the suitable mode (recording, uploading, or using a pre-uploaded sample file).
                Please ensure a quite environment before recording to obtain better recognition.
        </p>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<h3>    </h3>""", unsafe_allow_html=True)
        st.image('logo/app_logo.jpg')
        st.markdown("""<h1>    </h1>""", unsafe_allow_html=True)
        # st.markdown("""<br></br>""", unsafe_allow_html=True)
        st.markdown("""<p style='text-align: right;'> 
        هذه واجهة تجريبية للتعرف على تلاوة القرآن الكريم واكتشاف أخطاء التلاوة إن وجدت
            </p>""", unsafe_allow_html=True)
        st.markdown("""<h1>    </h1>""", unsafe_allow_html=True)
        st.markdown("""<h5>    </h5>""", unsafe_allow_html=True)
        st.markdown("""<p style='text-align: right;'> 
        للاستخدام، فضلا اختر آية من القائمة. ثم اختر الطريقة التي تناسبك، سواء كانت تسجيل صوتك أو تحميل ملف صوتي أو استخدام نموذج صوتي مسبق.
        مع مراعاة أن يكون التسجيل في بيئة خالية من الازعاج.
        </p>""", unsafe_allow_html=True)
#     st.header("Quran Recitation Recognition and Mistakes Detection")
#     st.markdown(
#         """
#         <h2 style='text-align: center;'>
#     هذه واجهة تجريبية للتعرف على تلاوة القرآن الكريم واكتشاف أخطاء التلاوة إن وجدت

# للاستخدام، فضلًا اختر آية من القائمة
  
# wav ثم اختر الطريقة التي تناسبك، سواء كانت تسجيل صوتك أو تحميل ملف صوتي بصيغة  

  
#   مع مراعاة أن يكون التسجيل بشكل واضح وفي بيئة خالية من الازعاج
  
  
# This is a demo app of Quran recitation recognition and mistakes detection project by Sarah Alrumiah. 
# This demo uses deep learning models trained with expert reciters' recitations.

# </h2>
# """,unsafe_allow_html=True)
    # if you want link in the markdown: [text](link)

    # select verse
    v112001 = "قُلْ هُوَ ٱللَّهُ أَحَدٌ"
    v112002 = "ٱللَّهُ ٱلصَّمَدُ"
    v112003 = "لَمْ يَلِدْ وَلَمْ يُولَدْ"
    v112004 = "وَلَمْ يَكُن لَّهُۥ كُفُوًا أَحَدٌۢ"
    v113001 = "قُلْ أَعُوذُ بِرَبِّ ٱلْفَلَقِ"
    v113002 = "مِن شَرِّ مَا خَلَقَ"
    v113003 = "وَمِن شَرِّ غَاسِقٍ إِذَا وَقَبَ"
    v113004 = "وَمِن شَرِّ ٱلنَّفَّٰثَٰتِ فِى ٱلْعُقَدِ"
    v113005 = "وَمِن شَرِّ حَاسِدٍ إِذَا حَسَدَ"
    v114001 = "قُلْ أَعُوذُ بِرَبِّ ٱلنَّاسِ"
    v114002 = "مَلِكِ ٱلنَّاسِ"
    v114003 = "إِلَٰهِ ٱلنَّاسِ"
    v114004 = "مِن شَرِّ ٱلْوَسْوَاسِ ٱلْخَنَّاسِ"
    v114005 = "ٱلَّذِى يُوَسْوِسُ فِى صُدُورِ ٱلنَّاسِ"
    v114006 = "مِنَ ٱلْجِنَّةِ وَٱلنَّاسِ"

    verse = st.selectbox("Choose a verse", [v112001, v112002, v112003, v112004, v113001, v113002, v113003, v113004, v113005, v114001, v114002, v114003, v114004, v114005, v114006])

    # select audio mode
    record_page = "Record your audio"
    upload_page = "Upload a wav file"
    app_mode = st.selectbox("Choose the app mode", [record_page, upload_page])

    if app_mode == record_page:
        col1, mid, col2 = st.columns([20, 1, 15])
        with col1:
            st.markdown("""<p style='text-align: justify;'> 
                        To start recording, please press the "Start" button. 
                        To end recording, please press the "Stop" button.
                </p>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
                 <p style='text-align: right;'> "Start"
            للبدء بالتسجيل، فضلًا اضغط  
                </p>
                 """, unsafe_allow_html=True)
            st.markdown("""
                     <p style='text-align: right;'> "Stop"
                وعندما تنتهي من التسجيل، فضلًا اضغط  
                    </p>
                     """, unsafe_allow_html=True)
        app_sst("record", verse)
    elif app_mode == upload_page:
        app_sst("upload", verse)


def handle_uploaded_audio_file(uploaded_file):
    a = pydub.AudioSegment.from_file(
        file=uploaded_file, format=uploaded_file.name.split(".")[-1]
    )

    channel_sounds = a.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max

    return fp_arr[:, 0], a.frame_rate

def audio_processing(waveform, sample_rate, mode):
    transform = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = transform(waveform)
    if mode == "record":
       waveform = torch.mean(waveform, dim=0).unsqueeze(0)
    elif mode == "upload":
        waveform = torch.from_numpy(waveform)
    valid_audio_transforms = torchaudio.transforms.MelSpectrogram()
    spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
    spectrograms = []
    spectrograms.append(spec)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    spectrograms = spectrograms.to(device)
    return spectrograms

def speech_to_text(spectrograms):
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 63,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1
    }

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)

    model_path = "models/RNN_Quran_model.pt"

    model.load_state_dict(torch.load(model_path, map_location ='cpu'))
    model.eval()

    prediction = model(spectrograms)
    prediction = F.log_softmax(prediction, dim=2)
    prediction = prediction.transpose(0, 1)  # (time, batch, n_class)

    decoded_preds = GreedyDecoderNew(prediction.transpose(0, 1))
    text = decoded_preds[0]
    return text

def postprocessing(text):
    max_target_len = 100  # = 200 all transcripts in out data are < 200 characters. SARAH CHange to 100
    vectorizer = VectorizeChar(max_target_len)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1,
    )

    optimizer = keras.optimizers.Adam(0.00001)

    model_new = Transformer(
        num_hid=200,
        num_head=2,
        num_feed_forward=400,
        target_maxlen=300,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=64,
    )
    model_new.compile(optimizer=optimizer, loss=loss_fn)

    model_new.load_weights('models/transformer_test_weights2.tf')

    idx_to_char = vectorizer.get_vocabulary()
    target_start_token_idx = 0
    target_end_token_idx = 1

    source = vectorizer(text)

    source = tf.expand_dims(source, axis=0)

    preds = model_new.generate(source, target_start_token_idx)

    preds = preds.numpy()

    prediction = ""
    for idx in preds[0]:
        prediction += idx_to_char[idx]
        if idx == target_end_token_idx:
            break
    prediction = prediction.replace('<', '')
    prediction = prediction.replace('>', '')
    return prediction

def app_sst(mode, verse):
    status_indicator = st.empty()

    #-----record----#
    if mode == "record":
        # to record user audio
        webrtc_ctx = webrtc_streamer(
            key="sendonly-audio",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            client_settings=ClientSettings(
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={
                    "audio": True
                },
            ),
        )

        if "audio_buffer" not in st.session_state:
            st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

        while True:
            if webrtc_ctx.audio_receiver:
                # start recording
                try:
                    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                except queue.Empty:
                    status_indicator.write("No frame arrived.")
                    continue
                # status_indicator.write("Recording is on. Start reciting!")
                status_indicator.write("تم تفعيل التسجيل، إبدأ بالتلاوة")
                sound_chunk = pydub.AudioSegment.empty()
                for audio_frame in audio_frames:
                    sound = pydub.AudioSegment(
                        data=audio_frame.to_ndarray().tobytes(),
                        sample_width=audio_frame.format.bytes,
                        frame_rate=audio_frame.sample_rate,
                        channels=len(audio_frame.layout.channels),
                    )
                    sound_chunk += sound

                if len(sound_chunk) > 0:
                    st.session_state["audio_buffer"] += sound_chunk
            else:
                # stop recording
                # status_indicator.write("Recording is off.")
                status_indicator.write(" خاصية التسجيل غير مفعلة")
                break

        audio_buffer = st.session_state["audio_buffer"]

        if not webrtc_ctx.state.playing and len(audio_buffer) > 0:
            # # save recording as wav file
            # date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
            # file_name = "wav_"+date
            # audio_buffer.export(file_name+".wav", format="wav")
            # waveform, sample_rate = torchaudio.load(file_name+".wav")

            # save recording in a virtual file
            virtualfile = io.BytesIO()
            audio_buffer.export(virtualfile, format="wav")
            waveform, sample_rate = torchaudio.load(virtualfile)
            recorded_voice = st.empty()
            recorded_voice.markdown("Here is the recorded audio:")
            st.audio(virtualfile)

            # speech to text
            st.info("Recognizing..")
            text_output = st.empty()
            spectrograms = audio_processing(waveform, sample_rate, mode)
            text = speech_to_text(spectrograms)
            text_output.markdown(f"**Recognized Verse:** {text}")

            # postprocessing: handling model's generated mistakes
            st.info("Postprocessing..")
            postprocessed_text = postprocessing(text)
            postprocess_text_output = st.empty()
            postprocess_text_output.markdown(f"**Postprocessed Verse:** {postprocessed_text}")

            # recitation verification
            st.info("Verifying recitation..")
            mistake_report = mistake_detection(postprocessed_text, verse)
            recitation_report_output = st.empty()
            recitation_report_output.markdown(f"**Recitation report:** {mistake_report}")

            # Reset
            st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

    #-----upload----#
    elif mode == "upload":
        st.sidebar.markdown("Upload a WAV audio file here:")
        file_uploader = st.sidebar.file_uploader(
            label="", type=[".wav"]
        )
        st.sidebar.markdown("Or select a sample file here:")
        selected_provided_file = st.sidebar.selectbox(
            label="", index = 0, options=["Select one", "Online Woman", "Online Man", "Author", "Alhusary", "Alakhdhar"]
        )

        if file_uploader is not None:
            # speech to text
            status_indicator.write("File uploaded!")
            waveform, sample_rate = handle_uploaded_audio_file(file_uploader)

            # present audio so the user can listen to
            virtualfile = io.BytesIO()
            wavfile.write(virtualfile, rate=sample_rate, data=waveform)

            upload_voice = st.empty()
            upload_voice.markdown(f"Here is the uploaded audio file:")
            st.audio(virtualfile)

            st.info("Recognizing..")
            text_output = st.empty()
            spectrograms = audio_processing(waveform, sample_rate, mode)
            text = speech_to_text(spectrograms)
            text_output.markdown(f"**Recognized Verse:** {text}")

            # postprocessing: handling model's generated mistakes
            st.info("Postprocessing..")
            postprocessed_text = postprocessing(text)
            postprocess_text_output = st.empty()
            postprocess_text_output.markdown(f"**Postprocessed Verse:** {postprocessed_text}")

            # recitation verification
            st.info("Verifying recitation..")
            mistake_report = mistake_detection(postprocessed_text, verse)
            recitation_report_output = st.empty()
            recitation_report_output.markdown(f"**Recitation report:** {mistake_report}")
        # select from sample file
        else:
            if selected_provided_file == "Select one":
                return
            verse = "قُلْ هُوَ ٱللَّهُ أَحَدٌ"
            if selected_provided_file == "Online Woman":
                file_name = "sample_wav/112001_onlinewoman.wav"
            elif selected_provided_file == "Online Man":
                file_name = "sample_wav/112001_onlineman.wav"
            elif selected_provided_file == "Author":
                file_name = "sample_wav/112001_sarah.wav"
            elif selected_provided_file == "Alhusary":
                file_name = "sample_wav/112001_Husary_64kbps.wav"
            elif selected_provided_file == "Alakhdhar":
                file_name = "sample_wav/112001_Ibrahim_Akhdar_32kbps.wav"

            # speech to text
            waveform, sample_rate = torchaudio.load(file_name)

            # present audio so the user can listen to
            virtualfile = io.BytesIO()
            torchaudio.save(virtualfile, waveform, sample_rate, format="wav")

            selected_voice = st.empty()
            selected_voice.markdown(f"Here is the selected audio file that is recited by {selected_provided_file}:")
            st.audio(virtualfile)

            spectrograms = audio_processing(waveform, sample_rate, "none")
            text = speech_to_text(spectrograms)
            text_output = st.empty()
            text_output.markdown(f"**Recognized Verse:** {text}")

            # postprocessing: handling model's generated mistakes
            st.info("Postprocessing..")
            postprocessed_text = postprocessing(text)
            postprocess_text_output = st.empty()
            postprocess_text_output.markdown(f"**Postprocessed Verse:** {postprocessed_text}")

            # recitation verification
            st.info("Verifying recitation..")
            mistake_report = mistake_detection(postprocessed_text, verse)
            recitation_report_output = st.empty()
            recitation_report_output.markdown(f"**Recitation report:** {mistake_report}")

if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
