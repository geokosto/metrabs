import os
import tensorflow as tf
import zipfile
import shutil


class ModelLoader:
    def __init__(self, models_dir):
        self.models_dir = os.path.abspath(models_dir)
        os.makedirs(self.models_dir, exist_ok=True)
        # print(f"Models directory: {self.models_dir}")

    def load_model(self, model_type):
        base_path = os.path.join(self.models_dir, model_type)

        # Check for directories with date suffixes
        matching_dirs = [
            d
            for d in os.listdir(self.models_dir)
            if d.startswith(model_type)
            and os.path.isdir(os.path.join(self.models_dir, d))
        ]

        if matching_dirs:
            # Use the most recent version if multiple matches exist
            model_path = os.path.join(self.models_dir, sorted(matching_dirs)[-1])
        else:
            model_path = base_path

        if not os.path.exists(model_path) or not os.path.isdir(model_path):
            print(f"Model not found locally. Downloading model: {model_type}")
            model_path = self._download_model(model_type)

        # Look for saved_model.pb in the directory structure
        saved_model_path = self._find_saved_model(model_path)
        if saved_model_path:
            print(f"Loading model from: {saved_model_path}")
            return tf.saved_model.load(saved_model_path)

        raise FileNotFoundError(f"SavedModel not found in {model_path}")

    def _find_saved_model(self, path):
        for root, dirs, files in os.walk(path):
            if "saved_model.pb" in files:
                return root
        return None

    def _download_model(self, model_type):
        server_prefix = "https://omnomnom.vision.rwth-aachen.de/data/metrabs"
        zip_filename = f"{model_type}.zip"
        zip_path = os.path.join(self.models_dir, zip_filename)

        if not os.path.exists(zip_path):
            print(f"Downloading {zip_filename}...")
            try:
                tf.keras.utils.get_file(
                    zip_path,
                    f"{server_prefix}/{zip_filename}",
                    cache_dir=None,
                    extract=False,
                )
            except Exception as e:
                print(f"Error downloading file: {str(e)}")
                raise

        print(f"Extracting {zip_filename}...")
        return self._extract_zip(zip_path, self.models_dir)

    def _extract_zip(self, zip_path, extract_to):
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)

            # Get the name of the extracted directory
            extracted_dir = zip_ref.namelist()[0].split("/")[0]
            extracted_path = os.path.join(extract_to, extracted_dir)

            print(f"Model extracted to: {extracted_path}")
            return extracted_path
        except Exception as e:
            print(f"Error extracting zip file: {str(e)}")
            raise
