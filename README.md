# teachless

![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Open source software for model training and prediction.

`teachless` provides a simple web interface to train and retrain machine learning models using local files, remote URLs, or datasets from Hugging Face. You can also use the trained models for predictions.

## Features

* **Flexible Training**: Train models using local feature/label files or from remote sources like Hugging Face.
* **Re-training**: Upload an existing model to continue training with new data.
* **Data Handling**: Specify columns to drop, define label columns, and select specific rows or ranges for training.
* **Prediction Interface**: A simple tab to upload your model and data to get predictions.
* **Web UI**: Built with Gradio for an easy-to-use interface.

## Installation

```bash
pip install "definers[cuda] @ git+https://github.com/YaronKoresh/definers.git" --extra-index-url https://pypi.nvidia.com
pip install git+https://github.com/YaronKoresh/teachless.git
```

This will also install the required dependencies, such as `gradio` and `definers`.

## Usage

Once installed, you can launch the Gradio web interface by running the following command in your terminal:

```bash
teachless
```

This will start a local web server, and you can access the user interface by navigating to the provided URL.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing & Issues

Contributions are welcome! For bug reports and feature requests, please open an issue on the [Bug Tracker](https://github.com/YaronKoresh/teachless/issues).
