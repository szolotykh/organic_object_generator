import sys
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QLabel, QScrollArea)
from pyvistaqt import QtInteractor
from main import generate_mesh

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Organic Object Generator - GPU View")
        self.resize(1280, 720)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        controls_panel = QWidget()
        scroll.setWidget(controls_panel)
        controls_layout = QVBoxLayout(controls_panel)
        form_layout = QFormLayout()
        
        # --- Parameters ---
        self.shape_type = QComboBox()
        self.shape_type.addItems(["sphere", "cylinder", "tube"])
        
        self.dist_type = QComboBox()
        self.dist_type.addItems(["inside", "surface"])
        
        self.conn_type = QComboBox()
        self.conn_type.addItems(["nearest", "random"])

        self.r_sphere = self._create_dspin(1.0, 0.1, 10.0)
        self.h_cylinder = self._create_dspin(3.0, 0.1, 10.0)
        self.r_cylinder = self._create_dspin(1.0, 0.1, 10.0)
        self.r_inner_tube = self._create_dspin(0.2, 0.1, 5.0)
        self.r_outer_tube = self._create_dspin(1.0, 0.1, 5.0)
        
        self.n_pts = self._create_ispin(180, 10, 2000)
        self.k_nn = self._create_ispin(4, 1, 20)
        self.r_tube = self._create_dspin(0.07, 0.01, 1.0, step=0.01)
        self.res = self._create_ispin(80, 20, 300) # Default lower for responsiveness
        self.sigma_blur = self._create_dspin(0.4, 0.1, 5.0)
        self.lap_iters = self._create_ispin(15, 0, 100)
        self.seed = self._create_ispin(42, 0, 99999)

        # Add rows
        form_layout.addRow("Shape Type:", self.shape_type)
        form_layout.addRow("Distribution:", self.dist_type)
        form_layout.addRow("Connection:", self.conn_type)
        form_layout.addRow("R Sphere:", self.r_sphere)
        form_layout.addRow("H Cylinder:", self.h_cylinder)
        form_layout.addRow("R Cylinder:", self.r_cylinder)
        form_layout.addRow("R Inner Tube:", self.r_inner_tube)
        form_layout.addRow("R Outer Tube:", self.r_outer_tube)
        form_layout.addRow("N Points:", self.n_pts)
        form_layout.addRow("K Nearest:", self.k_nn)
        form_layout.addRow("R Tube (Thickness):", self.r_tube)
        form_layout.addRow("Resolution:", self.res)
        form_layout.addRow("Sigma Blur:", self.sigma_blur)
        form_layout.addRow("Smoothing Iters:", self.lap_iters)
        form_layout.addRow("Seed:", self.seed)

        controls_layout.addLayout(form_layout)
        
        self.btn_generate = QPushButton("Generate Mesh")
        self.btn_generate.setStyleSheet("font-weight: bold; padding: 10px; background-color: #4CAF50; color: white;")
        self.btn_generate.clicked.connect(self.generate_object)
        controls_layout.addWidget(self.btn_generate)
        controls_layout.addStretch()

        # Right panel for 3D view
        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        
        # Use Eye Dome Lighting (EDL) to simulate shadows and improve depth perception
        # This is often more effective than standard shadow mapping for complex organic shapes
        self.plotter.enable_eye_dome_lighting()
        
        # Layout assembly
        # Limit width of controls
        scroll.setFixedWidth(350)
        main_layout.addWidget(scroll)
        main_layout.addWidget(self.plotter)

        # Initial generation
        self.generate_object()

    def _create_dspin(self, val, min_val, max_val, step=0.1):
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(val)
        spin.setSingleStep(step)
        return spin

    def _create_ispin(self, val, min_val, max_val):
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(val)
        return spin

    def generate_object(self):
        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("Generating...")
        QApplication.processEvents()

        try:
            mesh = generate_mesh(
                shape_type=self.shape_type.currentText(),
                distribution_type=self.dist_type.currentText(),
                connection_type=self.conn_type.currentText(),
                R_sphere=self.r_sphere.value(),
                H_cylinder=self.h_cylinder.value(),
                R_cylinder=self.r_cylinder.value(),
                R_inner_tube=self.r_inner_tube.value(),
                R_outer_tube=self.r_outer_tube.value(),
                N_pts=self.n_pts.value(),
                k_nn=self.k_nn.value(),
                r_tube=self.r_tube.value(),
                res=self.res.value(),
                sigma_blur=self.sigma_blur.value(),
                lap_iters=self.lap_iters.value(),
                seed=self.seed.value()
            )

            # Convert trimesh to pyvista
            # Create faces array formatted for PyVista: [n_pts, p1, p2, p3, n_pts, ...]
            faces = np.hstack(np.c_[np.full(mesh.faces.shape[0], 3), mesh.faces])
            pv_mesh = pv.PolyData(mesh.vertices, faces)

            self.plotter.clear()
            # Use standard smooth shading. EDL will handle the depth visualization.
            # Removed pbr=True as it can look flat without a specific environment map.
            self.plotter.add_mesh(pv_mesh, color="teal", smooth_shading=True)
            self.plotter.reset_camera()
            
        except Exception as e:
            print(f"Error: {e}")
        
        self.btn_generate.setText("Generate Mesh")
        self.btn_generate.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
