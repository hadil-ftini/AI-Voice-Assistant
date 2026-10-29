import os
import sys
import time
import random
import subprocess
import tempfile
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors

from PySide6.QtCore import QThread, Signal, Qt, QUrl
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QFrame,
    QProgressBar, QMessageBox
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer


# ---------------- CONFIGURATION ----------------
# Set GEMINI_API_KEY in your Windows environment before running this app.
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# This voice name works with Piper versions that support downloaded voice names.
# If you have a local model file, set PIPER_MODEL to its full .onnx path and
# place its matching .onnx.json file beside it.
PIPER_MODEL = os.getenv("PIPER_MODEL", "en_US-lessac-medium").strip()

client = genai.Client(api_key=API_KEY) if API_KEY else None


# ---------------- GEMINI WORKER ----------------
class GeminiWorker(QThread):
    answer_ready = Signal(str)
    error_occurred = Signal(str)
    status_changed = Signal(str)

    def __init__(self, question):
        super().__init__()
        self.question = question

    def run(self):
        if client is None:
            self.error_occurred.emit(
                "Gemini API key not found.\n\n"
                "In PowerShell, set it with:\n"
                '$env:GEMINI_API_KEY="YOUR_API_KEY"\n\n'
                "Then restart the application."
            )
            return

        # A few extra attempts help with temporary 503/429/5xx service errors.
        # The Google SDK may also perform its own retries for transient errors.
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.status_changed.emit("Contacting Gemini...")
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=self.question,
                )
                answer = getattr(response, "text", None)
                if not answer or not answer.strip():
                    answer = "Gemini returned an empty response. Please try again."
                self.answer_ready.emit(answer.strip())
                return

            except Exception as error:
                message = str(error)
                retryable = any(
                    token in message.upper()
                    for token in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                                  "500", "INTERNAL", "504", "DEADLINE_EXCEEDED")
                )
                if retryable and attempt < max_attempts - 1:
                    wait_seconds = 2 ** (attempt + 1)
                    self.status_changed.emit(
                        f"Gemini is busy. Retrying in {wait_seconds} seconds..."
                    )
                    time.sleep(wait_seconds)
                    continue

                if "503" in message or "UNAVAILABLE" in message.upper():
                    message = (
                        "Gemini is temporarily busy (503 UNAVAILABLE). This is a "
                        "Google service/model availability issue, not a Piper error.\n\n"
                        "Please wait a little and try again, or choose another model "
                        "by setting GEMINI_MODEL to a model available to your API key.\n\n"
                        f"Technical details: {message}"
                    )
                elif "429" in message or "RESOURCE_EXHAUSTED" in message.upper():
                    message = (
                        "Gemini rate limit or quota reached. Wait before trying again, "
                        "check your API quota, or use a model with available quota.\n\n"
                        f"Technical details: {message}"
                    )
                self.error_occurred.emit(message)
                return


# ---------------- PIPER TTS WORKER ----------------
class PiperWorker(QThread):
    audio_ready = Signal(str)
    error_occurred = Signal(str)
    status_changed = Signal(str)

    def __init__(self, text_to_speak, output_path):
        super().__init__()
        self.text_to_speak = text_to_speak
        self.output_path = output_path

    def run(self):
        try:
            self.status_changed.emit("Generating speech with Piper...")
            command = [
                sys.executable, "-m", "piper",
                "-m", PIPER_MODEL,
                "-f", self.output_path,
                "--", self.text_to_speak,
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if result.returncode != 0:
                details = (result.stderr or result.stdout or "Unknown Piper error").strip()
                raise RuntimeError(
                    "Piper could not generate the audio. Check that piper-tts is installed "
                    "and that the selected voice/model is available.\n\n"
                    f"Command used: {' '.join(command[:-1])} [text]\n\n{details}"
                )
            if not os.path.exists(self.output_path) or os.path.getsize(self.output_path) == 0:
                raise RuntimeError("Piper finished but did not create a valid WAV file.")
            self.audio_ready.emit(self.output_path)
        except subprocess.TimeoutExpired:
            self.error_occurred.emit("Piper took too long to generate speech. Please try again.")
        except Exception as error:
            self.error_occurred.emit(str(error))


# ---------------- MAIN APPLICATION ----------------
class GeminiApp(QWidget):
    LIGHT_THEME = """
        QWidget { background-color: #F5F7FB; color: #202938; font-family: Segoe UI; font-size: 14px; }
        QFrame#card { background-color: #FFFFFF; border: 1px solid #E3E8F0; border-radius: 14px; }
        QLabel#title { font-size: 23px; font-weight: bold; color: #243B73; }
        QLabel#subtitle { color: #667085; font-size: 13px; }
        QLabel#hello { font-size: 24px; font-weight: bold; color: #243B73; }
        QLabel#section { font-size: 15px; font-weight: bold; }
        QTextEdit { background-color: #FFFFFF; color: #202938; border: 1px solid #D7DDE7; border-radius: 9px; padding: 10px; selection-background-color: #C9D7FF; }
        QPushButton { background-color: #E9EDF5; color: #263248; border: none; border-radius: 9px; padding: 10px 16px; font-weight: bold; }
        QPushButton:hover { background-color: #DCE4F4; }
        QPushButton#primary { background-color: #405BD5; color: white; }
        QPushButton#primary:hover { background-color: #3049B5; }
        QPushButton:disabled { background-color: #E5E7EB; color: #9299A5; }
        QProgressBar { border: none; background-color: #E6EAF1; border-radius: 4px; height: 6px; }
        QProgressBar::chunk { background-color: #405BD5; border-radius: 4px; }
    """

    DARK_THEME = """
        QWidget { background-color: #171A23; color: #E8EAF0; font-family: Segoe UI; font-size: 14px; }
        QFrame#card { background-color: #222633; border: 1px solid #343A4A; border-radius: 14px; }
        QLabel#title { font-size: 23px; font-weight: bold; color: #A9B9FF; }
        QLabel#subtitle { color: #A0A8BA; font-size: 13px; }
        QLabel#hello { font-size: 24px; font-weight: bold; color: #A9B9FF; }
        QLabel#section { font-size: 15px; font-weight: bold; }
        QTextEdit { background-color: #191D28; color: #E8EAF0; border: 1px solid #41485B; border-radius: 9px; padding: 10px; selection-background-color: #405BD5; }
        QPushButton { background-color: #343A4A; color: #E8EAF0; border: none; border-radius: 9px; padding: 10px 16px; font-weight: bold; }
        QPushButton:hover { background-color: #444C60; }
        QPushButton#primary { background-color: #526BE8; color: white; }
        QPushButton#primary:hover { background-color: #667CF0; }
        QPushButton:disabled { background-color: #2C303B; color: #777E8D; }
        QProgressBar { border: none; background-color: #343A4A; border-radius: 4px; height: 6px; }
        QProgressBar::chunk { background-color: #7186FF; border-radius: 4px; }
    """

    def __init__(self):
        super().__init__()
        self.is_dark = False
        self.gemini_worker = None
        self.piper_worker = None
        self.current_audio_path = None

        self.audio_output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.9)

        self.setWindowTitle("AI Voice Assistant")
        self.resize(850, 750)
        self.setMinimumSize(650, 600)
        self.build_ui()
        self.apply_theme()

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 22, 28, 22)
        main_layout.setSpacing(18)

        # Header
        header = QHBoxLayout()
        title_layout = QVBoxLayout()
        self.title_label = QLabel("AI Voice Assistant")
        self.title_label.setObjectName("title")
        self.subtitle_label = QLabel("Your AI voice assistant for questions and documentation")
        self.subtitle_label.setObjectName("subtitle")
        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.subtitle_label)
        header.addLayout(title_layout)
        header.addStretch()
        self.theme_button = QPushButton("🌙 Dark mode")
        self.theme_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_button.clicked.connect(self.toggle_theme)
        header.addWidget(self.theme_button)
        main_layout.addLayout(header)

        # Welcome card
        welcome_card = QFrame()
        welcome_card.setObjectName("card")
        welcome_layout = QVBoxLayout(welcome_card)
        welcome_layout.setContentsMargins(20, 18, 20, 18)
        welcome_layout.setSpacing(6)
        hello_label = QLabel("Hi! 👋")
        hello_label.setObjectName("hello")
        welcome_text = QLabel("How can I help you today? Ask me anything.")
        welcome_text.setObjectName("subtitle")
        welcome_text.setWordWrap(True)
        welcome_layout.addWidget(hello_label)
        welcome_layout.addWidget(welcome_text)
        main_layout.addWidget(welcome_card)

        # Question card
        question_card = QFrame()
        question_card.setObjectName("card")
        question_layout = QVBoxLayout(question_card)
        question_layout.setContentsMargins(18, 18, 18, 18)
        question_layout.setSpacing(12)
        question_title = QLabel("Your question")
        question_title.setObjectName("section")
        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("Type your question here...")
        self.question_input.setFixedHeight(115)
        question_buttons = QHBoxLayout()
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_question)
        self.ask_button = QPushButton("Ask Gemini")
        self.ask_button.setObjectName("primary")
        self.ask_button.clicked.connect(self.ask_gemini)
        question_buttons.addWidget(self.clear_button)
        question_buttons.addStretch()
        question_buttons.addWidget(self.ask_button)
        question_layout.addWidget(question_title)
        question_layout.addWidget(self.question_input)
        question_layout.addLayout(question_buttons)
        main_layout.addWidget(question_card)

        # Response card
        response_card = QFrame()
        response_card.setObjectName("card")
        response_layout = QVBoxLayout(response_card)
        response_layout.setContentsMargins(18, 18, 18, 18)
        response_layout.setSpacing(12)
        response_header = QHBoxLayout()
        response_title = QLabel("AI response")
        response_title.setObjectName("section")
        self.copy_button = QPushButton("Copy response")
        self.copy_button.clicked.connect(self.copy_response)
        response_header.addWidget(response_title)
        response_header.addStretch()
        response_header.addWidget(self.copy_button)
        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        self.answer_output.setPlaceholderText("Your answer will appear here...")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        response_layout.addLayout(response_header)
        response_layout.addWidget(self.answer_output, 1)
        response_layout.addWidget(self.progress)

        speech_row = QHBoxLayout()
        self.speech_label = QLabel("🔊 Piper voice output is ready")
        self.speech_label.setObjectName("subtitle")
        self.listen_button = QPushButton("Listen to response")
        self.listen_button.setEnabled(False)
        self.listen_button.clicked.connect(self.speak_response)
        self.stop_button = QPushButton("Stop audio")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_audio)
        speech_row.addWidget(self.speech_label)
        speech_row.addStretch()
        speech_row.addWidget(self.listen_button)
        speech_row.addWidget(self.stop_button)
        response_layout.addLayout(speech_row)
        main_layout.addWidget(response_card, 1)

        footer = QLabel('<a href="https://ai.google.dev/gemini-api/docs">Gemini API documentation</a>')
        footer.setOpenExternalLinks(True)
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(footer)

    # Theme
    def apply_theme(self):
        if self.is_dark:
            self.setStyleSheet(self.DARK_THEME)
            self.theme_button.setText("☀️ Light mode")
        else:
            self.setStyleSheet(self.LIGHT_THEME)
            self.theme_button.setText("🌙 Dark mode")

    def toggle_theme(self):
        self.is_dark = not self.is_dark
        self.apply_theme()

    # Gemini request
    def ask_gemini(self):
        question = self.question_input.toPlainText().strip()
        if not question:
            QMessageBox.information(self, "Empty question", "Please type a question first.")
            return
        if self.gemini_worker is not None and self.gemini_worker.isRunning():
            return

        self.stop_audio()
        self.ask_button.setEnabled(False)
        self.ask_button.setText("Thinking...")
        self.progress.show()
        self.speech_label.setText("Gemini is preparing your answer...")
        self.answer_output.clear()
        self.listen_button.setEnabled(False)

        self.gemini_worker = GeminiWorker(question)
        self.gemini_worker.answer_ready.connect(self.show_answer)
        self.gemini_worker.error_occurred.connect(self.show_error)
        self.gemini_worker.status_changed.connect(self.speech_label.setText)
        self.gemini_worker.finished.connect(self.request_finished)
        self.gemini_worker.start()

    def show_answer(self, answer):
        self.answer_output.setPlainText(answer)
        self.listen_button.setEnabled(bool(answer.strip()))
        # Automatically generate and play the spoken answer.
        self.speak_response()

    def show_error(self, error):
        self.answer_output.setPlainText("Sorry, something went wrong.\n\n" + error)
        self.speech_label.setText("Could not get an answer from Gemini.")

    def request_finished(self):
        self.progress.hide()
        self.ask_button.setEnabled(True)
        self.ask_button.setText("Ask Gemini")
        if self.gemini_worker is not None:
            self.gemini_worker.deleteLater()
            self.gemini_worker = None

    # Piper speech
    def speak_response(self):
        text_to_speak = self.answer_output.toPlainText().strip()
        if not text_to_speak:
            QMessageBox.information(self, "No response", "There is no response to read aloud yet.")
            return
        if self.piper_worker is not None and self.piper_worker.isRunning():
            return

        # Keep the generated file until another speech is generated or app closes.
        fd, output_path = tempfile.mkstemp(prefix="piper_answer_", suffix=".wav")
        os.close(fd)
        self.current_audio_path = output_path
        self.listen_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.speech_label.setText("Generating speech with Piper...")

        self.piper_worker = PiperWorker(text_to_speak, output_path)
        self.piper_worker.audio_ready.connect(self.play_audio)
        self.piper_worker.error_occurred.connect(self.speech_error)
        self.piper_worker.status_changed.connect(self.speech_label.setText)
        self.piper_worker.finished.connect(self.speech_finished)
        self.piper_worker.start()

    def play_audio(self, audio_path):
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(audio_path)))
        self.player.play()
        self.stop_button.setEnabled(True)
        self.speech_label.setText("Playing Piper voice output...")

    def stop_audio(self):
        self.player.stop()
        self.stop_button.setEnabled(False)
        if self.answer_output.toPlainText().strip():
            self.listen_button.setEnabled(True)

    def speech_error(self, error):
        self.speech_label.setText("Piper speech generation failed.")
        QMessageBox.warning(self, "Piper TTS error", error)

    def speech_finished(self):
        if self.piper_worker is not None:
            self.piper_worker.deleteLater()
            self.piper_worker = None
        self.listen_button.setEnabled(bool(self.answer_output.toPlainText().strip()))

    # Other buttons
    def clear_question(self):
        self.question_input.clear()
        self.answer_output.clear()
        self.stop_audio()
        self.speech_label.setText("🔊 Piper voice output is ready")

    def copy_response(self):
        answer = self.answer_output.toPlainText()
        if answer:
            QApplication.clipboard().setText(answer)
            self.copy_button.setText("Copied!")
            self.copy_button.setText("Copy response")

    def closeEvent(self, event):
        self.player.stop()
        # Do not close while a QThread is still running; ask the workers to finish.
        active_workers = [w for w in (self.gemini_worker, self.piper_worker) if w is not None and w.isRunning()]
        if active_workers:
            QMessageBox.information(
                self,
                "Please wait",
                "The application is still working. Please wait for the current request or speech generation to finish, then close it.",
            )
            event.ignore()
            return
        if self.current_audio_path and os.path.exists(self.current_audio_path):
            try:
                os.remove(self.current_audio_path)
            except OSError:
                pass
        event.accept()


# ---------------- RUN APPLICATION ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = GeminiApp()
    window.show()
    sys.exit(app.exec())
