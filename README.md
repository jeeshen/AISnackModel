# AISnack

A Malaysian snack detection and price estimation app built with MobileNetV2, OpenCV, and Streamlit.
Point a camera at a collection of Malaysian snacks and the app identifies every item and totals up the price automatically.

## Features
- Detects 22 Malaysian snack varieties from a single photo
- Colour-based region proposals segmented with HSV binning and morphological operations
- 4-stage NMS pipeline to suppress duplicates and false positives
- MobileNetV2 classifier fine-tuned on a custom snack dataset
- Itemised price breakdown displayed alongside annotated bounding boxes
- Configurable class labels and prices via JSON files

## Installation
```bash
# clone the repo
git clone https://github.com/jeeshen/AISnack.git

# navigate to project directory
cd AISnack

# install dependencies
pip install -r requirements.txt

# place the trained model in the models/ folder
# models/snack_classifier.keras  (not included due to file size)

# run the app
streamlit run app.py
```
