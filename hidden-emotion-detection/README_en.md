# Enhanced Hidden Emotion Detection System (enhance_hidden)

## Project Introduction
This project is a real-time video emotion recognition system based on deep learning, integrating face detection, head pose estimation, macro/micro expression and hidden emotion analysis. Suitable for research, education, and practical applications.

## Main Modules
- `main.py`: Main entry, UI scheduling and system initialization.
- `app.py`: Core business logic, manages module calls and data flow.
- `ui/`: GUI code, multi-panel layout based on PyQt5.
- `models/`: Deep learning model definitions and loading.
- `engines/`: Analysis engines (face detection, expression recognition, pose estimation, etc.).
- `core/`: Core algorithms and data structures.
- `utils/`: Utility functions and common components.
- `config/`: Configuration files and parameter management.

## Installation
```bash
git clone https://github.com/yourname/enhance_hidden.git
cd enhance_hidden
pip install -r requirements.txt
```

## Usage
- Start main program:
  ```bash
  python main.py
  ```
- See comments in `main.py` and `app.py` for parameter details.
- Supports real-time camera detection and video file analysis.

## Dependencies
- See requirements.txt. Common dependencies: numpy, pandas, opencv-python, torch, torchvision, PyQt5, dlib, matplotlib, tqdm, pyyaml, etc.

## UI Description
- Six-panel layout showing:
  1. Real-time video
  2. Face and head pose
  3. Key AU activation and intensity
  4. Macro expression analysis
  5. Micro expression analysis
  6. Hidden emotion analysis
- Multi-resolution and cross-platform support.

## Contribution Guide
See [CONTRIBUTING.md](CONTRIBUTING.md).

## License
MIT License

## Contact
1336495069@qq.com 