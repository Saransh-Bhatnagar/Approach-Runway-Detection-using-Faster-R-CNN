# Approach Runway Detection using Faster R-CNN

This repository contains the official code and analysis for the Master of Engineering project, "Approach Runway Detection from Images," completed in collaboration with Collins Aerospace at the University of Limerick. The project focuses on developing and rigorously evaluating a deep learning model to detect airport runways from aerial imagery captured during the final approach phase of flight.

![Model Demonstration](https://i.imgur.com/your-image-url.png)
*(**Action:** You should replace the URL above with a link to Figure 4 from your report. You can upload the image to your GitHub repo and link to it directly).*

## 📖 Project Overview

This project implements a **Faster R-CNN (ResNet-50 V2)** model to tackle the challenge of runway detection. Trained exclusively on **14,000+ synthetic images** from the LARD dataset, the model's performance was then benchmarked against a challenging test set composed of synthetic, real-nominal, and real-edge-case images.

The core of this work is not just the model's performance, but the **deep-dive error analysis** that follows. We move beyond standard metrics to quantitatively and qualitatively diagnose the model's fundamental failure modes, providing a clear, data-driven roadmap for future improvements in vision-based landing systems.

## ✨ Key Features & Findings

* **High-Performing Model:** Achieved a **0.597 mAP** on a challenging real-world test set, proving competitive against state-of-the-art benchmarks like YOLO-RWY.
* **Comprehensive Sensitivity Analysis:** A rigorous analysis was performed to identify the optimal operating point for real-world data, demonstrating that a non-standard **IoU threshold of 0.3** maximized the F1 score.
* **Deep-Dive Error Analysis:** The project provides a detailed investigation into the model's failure modes, proving that errors are primarily driven by:
    * **Object Scale:** Performance degrades significantly as runway area decreases and slant distance increases.
    * **Low Visual Contrast:** Failures are strongly correlated with washed-out scenes caused by sun glare or atmospheric haze.
    * **Feature Ambiguity:** The model can "hallucinate" runways on visually similar objects like cockpit panels, logos, and taxiways.
* **Labeling Ambiguity Investigation:** The analysis also uncovered limitations in the dataset, including cases where the model's predictions were more accurate than the incomplete ground truth labels.

## 🛠️ Tech Stack & Dependencies

* Python 3.8+
* PyTorch & Torchvision
* TorchMetrics
* Pandas & NumPy
* Matplotlib & Seaborn
* OpenCV
* Albumentations

## ⚙️ Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

The primary analyses are conducted within Jupyter notebooks.

1.  **Data Preparation:** Ensure the LARD dataset is downloaded and structured as expected. Pre-trained model weights and pre-processed data files should be placed in the appropriate directories.
2.  **Evaluation & Analysis:** Open and run the `main_analysis.ipynb` notebook. This notebook loads the model predictions and ground truth data, runs the evaluation functions, and generates all the key plots and tables presented in the final report.

## 📊 Results

The model's performance was benchmarked across three distinct test sets:

| Dataset | mAP | mAP_50 | mAP_75 |
| :--- | :---: | :---: | :---: |
| LARD_test_synth | 0.7778 | 0.9692 | 0.9043 |
| REAL_Nominal | 0.6363 | 0.8446 | 0.7422 |
| REAL_Edge_Cases | 0.4181 | 0.6136 | 0.4705 |

The detailed error analysis provided critical insights into the model's operational limitations, which are discussed at length in the final report.

## 🔮 Future Work

Based on our findings, we recommend the following avenues for future research:

1.  **Transition to Real-Time Architectures:** Implement and evaluate lightweight, single-stage detectors like YOLO-RWY, which are better suited for small object detection and on-board deployment.
2.  **Improve the Data:** Bridge the performance gap by expanding the training set with more real-world edge cases and targeted "hard negative" examples (e.g., taxiways, cockpits).
3.  **Advance the Task:** Move from object detection to **instance segmentation** (using models like VALNet) to provide the precise geometric data required for autonomous landing systems.

## 🙏 Acknowledgements

This project was conducted under the supervision of Mark Halton (University of Limerick) and Niall Ryan (Collins Aerospace). Their guidance and support were invaluable to the success of this work.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
