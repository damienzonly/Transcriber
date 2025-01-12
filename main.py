import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QTextEdit, QProgressBar, QLabel, QFileDialog, QComboBox, QSplitter, QHBoxLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import whisper
import io
import threading


class ModelLoaderWorker(QThread):
    model_loaded = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            self.progress_updated.emit(10)
            model = whisper.load_model(self.model_name)
            self.progress_updated.emit(100)
            self.model_loaded.emit(model)
        except Exception as e:
            self.error_occurred.emit(str(e))


class WhisperWorker(QThread):
    transcription_done = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    log_updated = pyqtSignal(str)

    def __init__(self, model, file_path, language):
        super().__init__()
        self.model = model
        self.file_path = file_path
        self.language = language

    def run(self):
        try:
            log_stream = io.StringIO()
            sys.stdout = log_stream

            result = self.model.transcribe(self.file_path, language=self.language, verbose=True)

            sys.stdout = sys.__stdout__
            self.transcription_done.emit(result["text"])
            self.log_updated.emit(log_stream.getvalue())
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            sys.stdout = sys.__stdout__


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech-to-Text con Whisper")
        self.setGeometry(200, 200, 800, 600)

        # Layout principale
        self.layout = QVBoxLayout()

        # Pulsante per caricare file audio
        self.load_button = QPushButton("Carica File Audio")
        self.load_button.clicked.connect(self.load_audio_file)
        self.layout.addWidget(self.load_button)

        # Selettore di lingua
        self.language_selector = QComboBox()
        self.language_selector.addItems([
            "Italiano (it)", "Inglese (en)", "Francese (fr)", "Tedesco (de)", "Spagnolo (es)"
        ])
        # self.layout.addWidget(self.language_selector)

        # Selettore di modello
        self.model_selector = QComboBox()
        self.model_selector.addItems(["turbo", "tiny", "base", "small", "medium", "large"])
        self.model_selector.setCurrentText("turbo")
        self.model_selector.currentTextChanged.connect(self.load_model)
        self.model_tooltip = QLabel("Multilingual models: turbo and large")
        self.model_tooltip.setStyleSheet("color: gray; font-size: 10px;")
        self.layout.addWidget(self.model_tooltip)
        dropdown_layout = QHBoxLayout()
        dropdown_layout.addWidget(self.language_selector)
        dropdown_layout.addWidget(self.model_selector)
        self.layout.addLayout(dropdown_layout)
        # self.layout.addWidget(self.model_selector)


        # Barra di progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

        # Splitter orizzontale
        self.splitter = QSplitter(Qt.Vertical)

        # Text box per la trascrizione
        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.splitter.addWidget(self.text_box)

        # Text box per lo standard output
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.splitter.addWidget(self.log_box)

        # Layout principale
        self.layout.addWidget(self.splitter)

        # Etichetta di stato
        self.status_label = QLabel("Pronto")
        self.layout.addWidget(self.status_label)

        # Contenitore centrale
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Modello iniziale
        self.model = None
        self.load_model()

    def load_model(self):
        model_name = self.model_selector.currentText()
        self.status_label.setText(f"Caricamento modello: {model_name}...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # Crea e avvia il worker per il caricamento del modello
        self.model_loader = ModelLoaderWorker(model_name)
        self.model_loader.progress_updated.connect(self.progress_bar.setValue)
        self.model_loader.model_loaded.connect(self.on_model_loaded)
        self.model_loader.error_occurred.connect(self.on_model_error)
        self.model_loader.start()

    def on_model_loaded(self, model):
        self.model = model
        self.status_label.setText(f"Modello caricato: {self.model_selector.currentText()}")
        self.progress_bar.setVisible(False)

    def on_model_error(self, error_message):
        self.status_label.setText(f"Errore nel caricamento del modello: {error_message}")
        self.progress_bar.setVisible(False)

    def load_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona file audio", "", "Audio Files (*.mp3 *.wav *.m4a)"
        )
        if file_path:
            self.status_label.setText(f"Elaborazione: {file_path}")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.text_box.clear()
            self.log_box.clear()

            selected_language = self.language_selector.currentText().split("(")[-1].strip(")")

            self.worker = WhisperWorker(self.model, file_path, selected_language)
            self.worker.transcription_done.connect(self.display_transcription)
            self.worker.error_occurred.connect(self.display_error)
            self.worker.log_updated.connect(self.display_log)
            self.worker.finished.connect(self.on_worker_finished)
            self.worker.start()

    def display_transcription(self, text):
        self.text_box.setText(text)
        self.status_label.setText("Trascrizione completata")

    def display_error(self, error_message):
        self.status_label.setText(f"Errore: {error_message}")

    def display_log(self, log):
        self.log_box.setText(log)

    def on_worker_finished(self):
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
