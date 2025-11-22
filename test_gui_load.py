
import sys
import os
import tkinter as tk

# Mock matplotlib to avoid display issues
import unittest
from unittest.mock import MagicMock
import sys

sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.backends.backend_tkagg'] = MagicMock()
sys.modules['matplotlib.widgets'] = MagicMock()

# Ensure we are in the python path
sys.path.append(os.getcwd())

from gui.main_gui import FIBSEMGUIApp

class TestGUI(unittest.TestCase):
    def test_init(self):
        root = tk.Tk()
        # We need to handle potential extra windows or events
        app = FIBSEMGUIApp(root)
        self.assertIsNotNone(app)
        # Check if new buttons exist
        self.assertTrue(hasattr(app, 'launch_preview'))
        self.assertTrue(hasattr(app, 'get_preview_selection'))
        root.destroy()

if __name__ == '__main__':
    unittest.main()
