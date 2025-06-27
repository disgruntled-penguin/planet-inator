import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QColorDialog, QPushButton, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QToolTip
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QPainter
from gui.qt_sim_simulation import QtSimSimulation

class SimulationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 600)
        self.sim = QtSimSimulation(600, 600)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_sim)
        self.timer.start(30)  # ~33 FPS
        self._qimage = None

    def update_sim(self):
        self.sim.step()
        self.sim.draw()
        arr = self.sim.get_surface_array()
        h, w, ch = arr.shape
        self._qimage = QImage(arr.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        self.update()

    def paintEvent(self, event):
        if self._qimage:
            painter = QPainter(self)
            painter.drawImage(0, 0, self._qimage)

    def update_doofs_params(self, params):
        self.sim.update_doofs(params)

class DoofsPlanetControlPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Doofs Planet Specs", parent)
        layout = QFormLayout()
        self.mass = QDoubleSpinBox()
        self.mass.setRange(0.1, 1000)
        self.mass.setValue(400)
        self.mass.setSingleStep(1)
        layout.addRow("Mass (in Earth masses)", self.mass)

        self.size = QDoubleSpinBox()
        self.size.setRange(0.01, 1.0)
        self.size.setValue(0.07)
        self.size.setSingleStep(0.01)
        layout.addRow("Size (display radius)", self.size)

        self.color_btn = QPushButton()
        self.color_btn.setStyleSheet("background-color: purple;")
        self.color_btn.clicked.connect(self.choose_color)
        self.color = QColor(128, 0, 128)
        layout.addRow("Color", self.color_btn)

        self.a = QDoubleSpinBox(); self.a.setRange(0.1, 50); self.a.setValue(3.3); self.a.setSingleStep(0.1)
        layout.addRow("a (semi-major axis, AU)", self.a)
        self.e = QDoubleSpinBox(); self.e.setRange(0.0, 0.99); self.e.setValue(0.3); self.e.setSingleStep(0.01)
        layout.addRow("e (eccentricity)", self.e)
        self.inc = QDoubleSpinBox(); self.inc.setRange(0, 180); self.inc.setValue(15); self.inc.setSingleStep(1)
        layout.addRow("inc (deg)", self.inc)
        self.Omega = QDoubleSpinBox(); self.Omega.setRange(0, 360); self.Omega.setValue(80); self.Omega.setSingleStep(1)
        layout.addRow("Omega (deg)", self.Omega)
        self.omega = QDoubleSpinBox(); self.omega.setRange(0, 360); self.omega.setValue(60); self.omega.setSingleStep(1)
        layout.addRow("omega (deg)", self.omega)
        self.f = QDoubleSpinBox(); self.f.setRange(0, 360); self.f.setValue(10); self.f.setSingleStep(1)
        layout.addRow("f (deg)", self.f)
        self.setLayout(layout)

        # Signal connections for live update
        self.mass.valueChanged.connect(self.emit_update)
        self.size.valueChanged.connect(self.emit_update)
        self.a.valueChanged.connect(self.emit_update)
        self.e.valueChanged.connect(self.emit_update)
        self.inc.valueChanged.connect(self.emit_update)
        self.Omega.valueChanged.connect(self.emit_update)
        self.omega.valueChanged.connect(self.emit_update)
        self.f.valueChanged.connect(self.emit_update)

    def choose_color(self):
        color = QColorDialog.getColor(self.color, self, "Choose Color")
        if color.isValid():
            self.color = color
            self.color_btn.setStyleSheet(f"background-color: {color.name()};")
            self.emit_update()

    def emit_update(self):
        params = {
            'mass': self.mass.value(),
            'size': self.size.value(),
            'color': (self.color.red(), self.color.green(), self.color.blue()),
            'a': self.a.value(),
            'e': self.e.value(),
            'inc': self.inc.value(),
            'Omega': self.Omega.value(),
            'omega': self.omega.value(),
            'f': self.f.value()
        }
        if hasattr(self, 'on_update') and callable(self.on_update):
            self.on_update(params)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulation + Doofs Planet Controls")
        self.resize(1000, 700)
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Shortcuts label with tooltips
        shortcuts = [
            ("Stars: T", "Toggle stars on/off (T)"),
            ("Zoom In: +", "Zoom in (+)"),
            ("Zoom Out: -", "Zoom out (-)"),
            ("Reset Zoom: R", "Reset zoom (R)"),
            ("Pause: P", "Pause simulation (P)"),
            ("Screenshot: S", "Save screenshot (S)"),
            ("Quit: Q", "Quit simulation (Q)")
        ]
        shortcuts_layout = QHBoxLayout()
        for label, tip in shortcuts:
            lbl = QLabel(label)
            lbl.setStyleSheet("font-weight: bold; margin-right: 15px;")
            lbl.setToolTip(tip)
            shortcuts_layout.addWidget(lbl)
        main_layout.addLayout(shortcuts_layout)

        # Main content: simulation + controls
        content_layout = QHBoxLayout()
        self.sim_widget = SimulationWidget()
        content_layout.addWidget(self.sim_widget, stretch=2)
        self.controls = DoofsPlanetControlPanel()
        content_layout.addWidget(self.controls, stretch=1)
        main_layout.addLayout(content_layout)

        # Connect controls to simulation
        self.controls.on_update = self.sim_widget.update_doofs_params
        self.controls.emit_update()  # Initial sync

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 