import os
import sys

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QFrame,
    QProgressBar, QMessageBox
)
from PySide6.QtGui import QFont
from google import genai


# ---------------- CONFIGURATION ----------------

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

client = genai.Client(api_key=API_KEY) if API_KEY else None


# ---------------- GEMINI WORKER ----------------

class GeminiWorker(QThread):
    answer_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, question):
        super().__init__()
        self.question = question

    def run(self):
        try:
            if client is None:
                raise RuntimeError(
                    "API key not found. Set the GEMINI_API_KEY "
                    "environment variable first."
                )

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=self.question
            )

            answer = response.text or "No response was returned."
            self.answer_ready.emit(answer)

        except Exception as error:
            self.error_occurred.emit(str(error))


# ---------------- MAIN APPLICATION ----------------

class GeminiApp(QWidget):

    LIGHT_THEME = """
        QWidget {
            background-color: #F5F7FB;
            color: #202938;
            font-family: Segoe UI;
            font-size: 14px;
        }

        QFrame#card {
            background-color: #FFFFFF;
            border: 1px solid #E3E8F0;
            border-radius: 14px;
        }

        QLabel#title {
            font-size: 23px;
            font-weight: bold;
            color: #243B73;
        }

        QLabel#subtitle {
            color: #667085;
            font-size: 13px;
        }

        QLabel#hello {
            font-size: 24px;
            font-weight: bold;
            color: #243B73;
        }

        QLabel#section {
            font-size: 15px;
            font-weight: bold;
        }

        QTextEdit {
            background-color: #FFFFFF;
            border: 1px solid #D7DDE7;
            border-radius: 9px;
            padding: 10px;
            selection-background-color: #C9D7FF;
        }

        QPushButton {
            background-color: #E9EDF5;
            color: #263248;
            border: none;
            border-radius: 9px;
            padding: 10px 16px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #DCE4F4;
        }

        QPushButton#primary {
            background-color: #405BD5;
            color: white;
        }

        QPushButton#primary:hover {
            background-color: #3049B5;
        }

        QPushButton:disabled {
            background-color: #E5E7EB;
            color: #9299A5;
        }

        QProgressBar {
            border: none;
            background-color: #E6EAF1;
            border-radius: 4px;
            height: 6px;
        }

        QProgressBar::chunk {
            background-color: #405BD5;
            border-radius: 4px;
        }
    """

    DARK_THEME = """
        QWidget {
            background-color: #171A23;
            color: #E8EAF0;
            font-family: Segoe UI;
            font-size: 14px;
        }

        QFrame#card {
            background-color: #222633;
            border: 1px solid #343A4A;
            border-radius: 14px;
        }

        QLabel#title {
            font-size: 23px;
            font-weight: bold;
            color: #A9B9FF;
        }

        QLabel#subtitle {
            color: #A0A8BA;
            font-size: 13px;
        }

        QLabel#hello {
            font-size: 24px;
            font-weight: bold;
            color: #A9B9FF;
        }

        QLabel#section {
            font-size: 15px;
            font-weight: bold;
        }

        QTextEdit {
            background-color: #191D28;
            color: #E8EAF0;
            border: 1px solid #41485B;
            border-radius: 9px;
            padding: 10px;
            selection-background-color: #405BD5;
        }

        QPushButton {
            background-color: #343A4A;
            color: #E8EAF0;
            border: none;
            border-radius: 9px;
            padding: 10px 16px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #444C60;
        }

        QPushButton#primary {
            background-color: #526BE8;
            color: white;
        }

        QPushButton#primary:hover {
            background-color: #667CF0;
        }

        QPushButton:disabled {
            background-color: #2C303B;
            color: #777E8D;
        }

        QProgressBar {
            border: none;
            background-color: #343A4A;
            border-radius: 4px;
            height: 6px;
        }

        QProgressBar::chunk {
            background-color: #7186FF;
            border-radius: 4px;
        }
    """

    def __init__(self):
        super().__init__()

        self.is_dark = False
        self.worker = None

        self.setWindowTitle("AI Voice Assistant")
        self.resize(850, 750)
        self.setMinimumSize(650, 600)

        self.build_ui()
        self.apply_theme()

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 22, 28, 22)
        main_layout.setSpacing(18)

        # ---------- HEADER ----------

        header = QHBoxLayout()

        title_layout = QVBoxLayout()

        self.title_label = QLabel("AI Voice Assistant")
        self.title_label.setObjectName("title")

        self.subtitle_label = QLabel(
            "Your AI voice assistant for questions and documentation"
        )
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

        # ---------- WELCOME MESSAGE ----------

        welcome_card = QFrame()
        welcome_card.setObjectName("card")

        welcome_layout = QVBoxLayout(welcome_card)
        welcome_layout.setContentsMargins(20, 18, 20, 18)
        welcome_layout.setSpacing(6)

        self.hello_label = QLabel("Hi! 👋")
        self.hello_label.setObjectName("hello")

        welcome_text = QLabel(
            "How can I help you today? Ask me anything."
        )
        welcome_text.setObjectName("subtitle")
        welcome_text.setWordWrap(True)

        welcome_layout.addWidget(self.hello_label)
        welcome_layout.addWidget(welcome_text)

        main_layout.addWidget(welcome_card)

        # ---------- QUESTION CARD ----------

        question_card = QFrame()
        question_card.setObjectName("card")

        question_layout = QVBoxLayout(question_card)
        question_layout.setContentsMargins(18, 18, 18, 18)
        question_layout.setSpacing(12)

        question_title = QLabel("Your question")
        question_title.setObjectName("section")

        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText(
            "Type your question here..."
        )
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

        # ---------- RESPONSE CARD ----------

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
        self.answer_output.setPlaceholderText(
            "Your answer will appear here..."
        )

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()

        response_layout.addLayout(response_header)
        response_layout.addWidget(self.answer_output, 1)
        response_layout.addWidget(self.progress)

        # Future speech feature: not connected to TTS yet.
        speech_row = QHBoxLayout()

        self.speech_label = QLabel(
            "🔊 Voice output: coming soon"
        )
        self.speech_label.setObjectName("subtitle")

        self.listen_button = QPushButton("Listen to response")
        self.listen_button.setEnabled(False)
        self.listen_button.setToolTip(
            "Connect a TTS model such as Piper to enable audio."
        )

        speech_row.addWidget(self.speech_label)
        speech_row.addStretch()
        speech_row.addWidget(self.listen_button)

        response_layout.addLayout(speech_row)
        main_layout.addWidget(response_card, 1)

        # ---------- FOOTER ----------

        footer = QLabel(
            '<a href="https://ai.google.dev/ai-voice-assistant/docs">'
            "AI Voice Assistant documentation</a>"
        )
        footer.setOpenExternalLinks(True)
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main_layout.addWidget(footer)

    # ---------- THEME ----------

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

    # ---------- GEMINI ----------

    def ask_gemini(self):
        question = self.question_input.toPlainText().strip()

        if not question:
            QMessageBox.information(
                self,
                "Empty question",
                "Please type a question first."
            )
            return

        if self.worker is not None and self.worker.isRunning():
            return

        self.ask_button.setEnabled(False)
        self.ask_button.setText("Thinking...")
        self.progress.show()
        self.answer_output.clear()

        self.worker = GeminiWorker(question)
        self.worker.answer_ready.connect(self.show_answer)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.finished.connect(self.request_finished)
        self.worker.start()

    def show_answer(self, answer):
        self.answer_output.setPlainText(answer)

    def show_error(self, error):
        self.answer_output.setPlainText(
            "Sorry, something went wrong.\n\n" + error
        )

    def request_finished(self):
        self.progress.hide()
        self.ask_button.setEnabled(True)
        self.ask_button.setText("Ask Gemini")

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    # ---------- BUTTON ACTIONS ----------

    def clear_question(self):
        self.question_input.clear()
        self.answer_output.clear()

    def copy_response(self):
        answer = self.answer_output.toPlainText()

        if answer:
            QApplication.clipboard().setText(answer)
            self.copy_button.setText("Copied!")
            # Restore the original label immediately.
            self.copy_button.setText("Copy response")


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    window = GeminiApp()
    window.show()

    sys.exit(app.exec())