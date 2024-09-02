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
        model_path = os.path.join(self.models_dir, model_type)

        if os.path.exists(model_path):
            print(f"Loading model from local directory: {model_path}")
            return tf.saved_model.load(model_path)
        else:
            print(f"Model not found locally. Downloading model: {model_type}")
            model_path = self._download_model(model_type)
            return tf.saved_model.load(model_path)

    def _download_model(self, model_type):
        server_prefix = "https://omnomnom.vision.rwth-aachen.de/data/metrabs"
        zip_filename = f"{model_type}_20211019.zip"
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

        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Downloaded file not found: {zip_path}")

        model_path = os.path.join(self.models_dir, model_type)
        if not os.path.exists(model_path):
            print(f"Extracting {zip_filename}...")
            self._extract_zip(zip_path, self.models_dir)

        return model_path

    def _extract_zip(self, zip_path, extract_to):
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        except zipfile.BadZipFile:
            print(f"Error: The file {zip_path} is not a valid zip file.")
            raise
        except Exception as e:
            print(f"Error extracting zip file: {str(e)}")
            raise

        # The extracted folder might have a different name, so we need to rename it
        extracted_folder = os.path.join(
            extract_to, os.path.splitext(os.path.basename(zip_path))[0]
        )
        model_folder = os.path.join(extract_to, os.path.basename(extract_to))

        if os.path.exists(extracted_folder) and extracted_folder != model_folder:
            if os.path.exists(model_folder):
                shutil.rmtree(model_folder)
            os.rename(extracted_folder, model_folder)

        print(f"Model extracted to: {model_folder}")
