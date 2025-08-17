from flask import Flask, request, render_template
import os
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras.preprocessing import image
import class_label
app = Flask(__name__)

# Set up folder paths
UPLOAD_FOLDER = "static/uploads"
MODEL_FOLDER = "Models"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # Disable caching

# Define class labels and Wikipedia links
class_labels = [
    "Clams", "Corals", "Crabs", "Dolphin", "Eel", "Fish", "Jelly Fish", "Lobster", "Nudibranchs", "Octopus",
    "Otter", "Penguin", "Puffers", "Sea Rays", "Sea Urchins", "Seahorse", "Seal", "Sharks", "Shrimp", "Squid",
    "Starfish", "Turtle_tortoise", "Whale"
]



wiki_links = {
    label: f"https://en.wikipedia.org/wiki/{label.replace(' ', '_')}" for label in class_labels
}

# Load only the required models
model_paths = {
    "Best Fixed Dropout": os.path.join(MODEL_FOLDER, "cnn_final_best_dropout.h5"),
    "Cyclic Dropout": os.path.join(MODEL_FOLDER, "cyclic_dropout.h5"),
    "Dropout 0.5": os.path.join(MODEL_FOLDER, "cnn_dropout_5.h5")
}

models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        models[name] = tf.keras.models.load_model(path)
        print(f"Loaded model: {name}")
    else:
        print(f"Model not found: {path}")


# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array


# Function to predict and determine the best class
def test_selected_models(test_img_path):
    test_img = preprocess_image(test_img_path)
    all_preds = []

    for model in models.values():
        pred = model.predict(test_img)
        all_preds.append(np.argmax(pred))

    vote_count = Counter(all_preds)
    most_common = vote_count.most_common()

    if not most_common:
        return "Unknown", "No information available", "#"

    best_pred_index = most_common[0][0]
    best_class = class_labels[best_pred_index]
    class_description = class_label.class_info.get(best_class, "No information available")
    wiki_link = wiki_links.get(best_class, "#")

    return best_class, class_description, wiki_link


@app.route("/", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"

        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        best_class, class_description, wiki_link = test_selected_models(file_path)

        return render_template("index.html", image=file_path, best_class=best_class,
                               class_description=class_description, wiki_link=wiki_link)

    return render_template("index.html", image=None, best_class=None, class_description=None, wiki_link=None)


if __name__ == "__main__":
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # Disable caching
    app.run(debug=True)
