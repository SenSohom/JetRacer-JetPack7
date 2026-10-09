"""Patch a fresh clone of https://github.com/NVIDIA-AI-IOT/jetracer for JetPack 7 on Jetson Orin Nano.

Usage (on the Jetson):  python3 jetracer_jp7_patch.py ~/jetracer
Safe to run more than once: already-patched files are left alone.

What it changes:
- jetracer/nvidia_racecar.py: send center steering / neutral throttle at startup
  (traitlets only fires on *change*, so setting 0 on a fresh object sends no pulse)
- notebooks: torch2trt -> plain FP16 PyTorch model (torch2trt doesn't support TensorRT 10)
- notebooks: torchvision `pretrained=` -> `weights=` (removed in newer torchvision)
- notebooks: CSICamera capture_fps=65 (old Nano mode) -> 1280x720 @ 60 (IMX219 mode on Orin)
- notebooks: jupyter_clickable_image_widget -> ipyevents click (works with JupyterLab 4)
- notebooks: `except e:` bug -> `except Exception as ex:`
- safety (no RC override once the mux is bypassed):
    road_following.ipynb drive loop always stops the car when interrupted
    road_following_live.ipynb throttle slider starts at 0 instead of 0.15
- xy_dataset.py: numpy 2 compatible random flip
"""
import json
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).expanduser()
NB_DIR = REPO / "notebooks"
CAMERA_ARGS = "capture_width=1280, capture_height=720, capture_fps=60"
LOAD_FP16_MODEL = """# JetPack 7: torch2trt is not used; run the FP16 PyTorch model directly on the GPU
import torch
import torchvision

model_trt = torchvision.models.resnet18(weights=None)
model_trt.fc = torch.nn.Linear(512, 2)
model_trt.load_state_dict(torch.load('road_following_model.pth'))
model_trt = model_trt.cuda().eval().half()"""

DRIVE_LOOP_OLD = """car.throttle = 0.15

while True:
    image = camera.read()
    image = preprocess(image).half()
    output = model_trt(image).detach().cpu().numpy().flatten()
    x = float(output[0])
    car.steering = x * STEERING_GAIN + STEERING_BIAS"""
DRIVE_LOOP_NEW = """car.throttle = 0.15

# No RC override without the mux: always stop the car when this cell is interrupted or fails
try:
    while True:
        image = camera.read()
        image = preprocess(image).half()
        output = model_trt(image).detach().cpu().numpy().flatten()
        x = float(output[0])
        car.steering = x * STEERING_GAIN + STEERING_BIAS
finally:
    car.throttle = 0.0
    car.steering = 0.0"""

REPLACEMENTS = [
    ("torchvision.models.resnet18(pretrained=False)", "torchvision.models.resnet18(weights=None)"),
    ("(pretrained=True)", "(weights='DEFAULT')"),
    ("CSICamera(width=224, height=224, capture_fps=65)", f"CSICamera(width=224, height=224, {CAMERA_ARGS})"),
    ("CSICamera(width=224, height=224)", f"CSICamera(width=224, height=224, {CAMERA_ARGS})"),
    ("    except e:\n        pass", "    except Exception as ex:\n        print(ex)"),
    ("from jupyter_clickable_image_widget import ClickableImageWidget",
     "from ipyevents import Event  # JetPack 7: replaces jupyter_clickable_image_widget"),
    ("camera_widget = ClickableImageWidget(width=camera.width, height=camera.height)",
     "camera_widget = ipywidgets.Image(format='jpeg', width=camera.width, height=camera.height)"),
    ("def save_snapshot(_, content, msg):\n    if content['event'] == 'click':\n        data = content['eventData']\n"
     "        x = data['offsetX']\n        y = data['offsetY']\n",
     "def save_snapshot(event):\n    if event['type'] == 'click':\n"
     "        x = int(event['dataX'])\n        y = int(event['dataY'])\n"),
    ("camera_widget.on_msg(save_snapshot)",
     "click_event = Event(source=camera_widget, watched_events=['click'])\nclick_event.on_dom_event(save_snapshot)"),
    (DRIVE_LOOP_OLD, DRIVE_LOOP_NEW),
    ("max=1.0, value=0.15, step=0.01, orientation=", "max=1.0, value=0.0, step=0.01, orientation="),
]


def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None


def patch_notebook(path):
    nb = json.loads(path.read_text())
    changed = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        new = src
        if DRIVE_LOOP_NEW not in new:
            for old, rep in REPLACEMENTS:
                new = new.replace(old, rep)
        if "from torch2trt import TRTModule" in new:
            new = LOAD_FP16_MODEL
        elif "model_trt = torch2trt(" in new:
            new = "# JetPack 7: no TensorRT conversion needed; the FP16 PyTorch model is used as-is\nmodel_trt = model"
        elif "torch.save(model_trt.state_dict()" in new:
            new = "# JetPack 7: nothing to save; road_following.ipynb loads road_following_model.pth directly"
        if new != src:
            set_source(cell, new)
            changed += 1
    if changed:
        path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"{path.name}: {changed} cells changed")


def patch_racecar():
    path = REPO / "jetracer" / "nvidia_racecar.py"
    src = path.read_text()
    marker = "send center/neutral explicitly at startup"
    if marker in src:
        print("nvidia_racecar.py: already patched")
        return
    anchor = "        self.throttle_motor = self.kit.continuous_servo[self.throttle_channel]\n"
    assert src.count(anchor) == 1, "unexpected nvidia_racecar.py layout"
    src = src.replace(anchor, anchor +
                      f"        # traitlets only fires on change, so {marker}\n"
                      "        self.steering_motor.throttle = self.steering_offset\n"
                      "        self.throttle_motor.throttle = 0.0\n")
    path.write_text(src)
    print("nvidia_racecar.py: patched")


patch_racecar()
for name in ["road_following.ipynb", "road_following_live.ipynb", "interactive_regression.ipynb", "optimize_model.ipynb"]:
    patch_notebook(NB_DIR / name)

ds = NB_DIR / "xy_dataset.py"
src = ds.read_text()
if "float(np.random.random(1)) > 0.5" in src:
    ds.write_text(src.replace("float(np.random.random(1)) > 0.5", "np.random.random() > 0.5"))
    print("xy_dataset.py: numpy 2 fix")
else:
    print("xy_dataset.py: already patched")
