import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path
import streamlit as st


class DetectionManager:
    def __init__(self):
        self.model_path = str(
            Path(__file__).resolve().parent.parent / "best.pt"
        )

        self.model = None

        self.class_names = {
            0: "Stones / Stone Pillars / Stone Structures",
            1: "Crops / Farmland",
            2: "Non-archaeological (deserts, water, mountains, etc.)",
            3: "Heritage Sites (temples, palaces, forts, museums)"
        }

        self.class_colors = {
            0: (139, 69, 19),
            1: (34, 139, 34),
            2: (105, 105, 105),
            3: (184, 134, 11)
        }

        self.load_model()

    def load_model(self):
        """Load the YOLO model"""
        try:
            st.write("Model path:", self.model_path)

            if os.path.exists(self.model_path):
                st.success(f"Model found: {self.model_path}")

                self.model = YOLO(self.model_path)

                return True

            else:
                st.error(f"Model file not found at {self.model_path}")

                st.write("Current directory:", os.getcwd())

                app_folder = Path(__file__).resolve().parent.parent

                st.write("App folder:", str(app_folder))

                st.write(
                    "Files in app folder:",
                    os.listdir(app_folder)
                )

                return False

        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False