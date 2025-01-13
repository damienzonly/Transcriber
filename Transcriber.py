import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QTextEdit, QProgressBar, QLabel, QFileDialog, QComboBox, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from pydub import AudioSegment
import whisper
import traceback
import tempfile
from pprint import pprint
from pydub.silence import detect_nonsilent
import os
import platform
import subprocess

segment_length_ms = 60000

def split_audio_on_silence(file_path, max_segment_length=60_000, silence_thresh=-40, min_silence_len=500):
    """
    Splits an audio file into segments of up to max_segment_length (in ms), cutting only at silent parts.

    Args:
        file_path (str): Path to the audio file.
        max_segment_length (int): Maximum length of each segment in milliseconds (default 60,000 ms = 60 seconds).
        silence_thresh (int): Threshold in dB below which sound is considered silent (default -40 dB).
        min_silence_len (int): Minimum duration of silence in milliseconds to be considered a split point (default 500 ms).

    Returns:
        List[AudioSegment]: A list of audio segments.
    """
    audio = AudioSegment.from_file(file_path)
    nonsilent_ranges = detect_nonsilent(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )
    segments = []
    current_segment = AudioSegment.empty()

    for start, end in nonsilent_ranges:
        chunk = audio[start:end]
        if len(current_segment) + len(chunk) > max_segment_length:
            segments.append(current_segment)
            current_segment = AudioSegment.empty()
        current_segment += chunk

    if len(current_segment) > 0:
        segments.append(current_segment)
    
    return segments


def open_in_editor(filepath):
    if platform.system() == 'Darwin': 
        subprocess.call(('open', filepath))
    elif platform.system() == 'Windows': 
        os.startfile(filepath)
    else:
        subprocess.call(('xdg-open', filepath))

class WhisperWorker(QThread):
    output = pyqtSignal(str)
    logging = pyqtSignal(str)
    status_log = pyqtSignal(str)
    open_editor_enable = pyqtSignal(bool)

    def __init__(self, filepath, model, language):
        super().__init__()
        self.filepath = filepath
        self.model = model
        self.language = language

    def run(self):
        try:
            self.open_editor_enable.emit(False)
            processed_seconds = 0
            self.status_log.emit(f"Detecting silences in audio file... This may take a while")
            audio_segments = split_audio_on_silence(self.filepath, max_segment_length=segment_length_ms)
            self.status_log.emit(f"Silence detection done. Found {len(audio_segments)} segments")
            for idx, segment in enumerate(audio_segments):
                self.logging.emit(f"Transcribing segment {idx + 1} of {len(audio_segments)}...")
                tmpfile = tempfile.TemporaryDirectory()
                filename = tmpfile.name + ".wav"
                try:
                    segment.export(filename, format="wav")
                    self.logging.emit(f"Segment {idx + 1} transcription started...")
                    result = self.model.transcribe(filename, language=self.language)
                    for segment in result['segments']:
                        processed_seconds += segment['end'] - segment['start']
                        self.logging.emit(f"[{(segment['start'] + processed_seconds):.2f} - {(segment['end'] + processed_seconds):.2f}] {segment['text']}")
                    self.output.emit(result["text"])
                    self.open_editor_enable.emit(True)
                except Exception as e:
                    self.status_log.emit(str(e))
                    return
                finally:
                    tmpfile.cleanup()
            self.status_log.emit("Transcription completed")
            self.open_editor_enable.emit(True)
        except Exception as e:
            traceback.print_exc()
            self.status_log.emit(str(e))


class MainWindow(QMainWindow):
    tmpdir = None
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transcriber")
        self.setGeometry(200, 200, 800, 600)

        self.layout = QVBoxLayout()

        self.load_file_button = QPushButton("Load audio file")
        self.load_file_button.clicked.connect(self.load_audio_file)
        self.load_file_button.setEnabled(False)
        self.layout.addWidget(self.load_file_button)

        self.language_selector = QComboBox()
        self.language_selector.addItems([
            "Italian (it)", "English (en)", "French (fr)", "Dutch (de)", "Spanish (es)"
        ])
        self.layout.addWidget(self.language_selector)

        self.model_selector = QComboBox()
        self.model_selector.addItems(['turbo', 'large', 'medium', 'small', 'base', 'tiny'])
        self.model_selector.setCurrentText("turbo")
        self.model_selector.currentTextChanged.connect(self.load_model)
        self.layout.addWidget(self.model_selector)

        self.splitter = QSplitter(Qt.Vertical)

        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.splitter.addWidget(self.text_box)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.splitter.addWidget(self.log_box)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.layout.addWidget(self.status_label)

        self.layout.addWidget(self.splitter)

        self.open_in_editor = QPushButton("Open in editor...")
        self.open_in_editor.setEnabled(False)
        self.open_in_editor.clicked.connect(lambda: open_in_editor(self.tmpdir.name + ".txt"))
        self.footer = QHBoxLayout()
        self.footer.addWidget(self.open_in_editor)
        self.layout.addLayout(self.footer)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        self.model = None
        self.load_model()
        self.set_tmp_file()

    def load_model(self):
        def load():
            setattr(self, 'model', whisper.load_model(model_name))
            self.load_file_button.setEnabled(True)
        self.load_file_button.setEnabled(False)
        model_name = self.model_selector.currentText()
        self.status_log(f"Loading model: {model_name}...")
        self.progress_bar.setVisible(True)

        self.model_loader = QThread()
        self.model_loader.run = load
        self.model_loader.finished.connect(lambda: self.status_log("Model loaded"))
        self.model_loader.start()

    def status_log(self, log):
        self.status_label.setText(log)

    def load_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select audio file", "", "Audio Files (*.mp3 *.wav *.m4a)")
        self.set_tmp_file()
        self.text_box.clear()
        self.log_box.clear()
        if file_path:
            self.status_log("Splitting audio into segments...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            selected_language = self.language_selector.currentText().split("(")[-1].strip(")")

            self.worker = WhisperWorker(file_path, self.model, selected_language)
            self.worker.output.connect(self.display_transcription)
            self.worker.logging.connect(self.display_log)
            self.worker.open_editor_enable.connect(self.set_save_as_enabled)
            self.worker.status_log.connect(lambda log: self.status_log(log))
            self.worker.finished.connect(self.on_worker_finished)
            self.open_in_editor.setEnabled(False)
            self.worker.start()

    def set_save_as_enabled(self, enabled):
        self.open_in_editor.setEnabled(enabled)

    def set_tmp_file(self):
        if (self.tmpdir):
            self.tmpdir.cleanup()
        self.tmpdir = tempfile.TemporaryDirectory()

    def display_transcription(self, text):
        self.text_box.append(text)
        with open(self.tmpdir.name + ".txt", "a") as f:
            f.write(text)
        # self.status_log("Transcription completed")

    def display_error(self, error_message):
        self.status_log(f"Error: {error_message}")

    def display_log(self, log):
        self.log_box.append(log)

    def on_worker_finished(self):
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
