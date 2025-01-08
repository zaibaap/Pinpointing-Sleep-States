# Portfolio of Projects

Welcome to my project portfolio! This repository showcases a collection of my previous work, highlighting my experience in data science, and machine learning.

## Table of Contents
- [Projects](#projects)
- [Technologies Used](#technologies-used)
- [How to Use](#how-to-use)
- [Contact](#contact)

## Projects
### 1. ASL_Hand_Gesture_Recognition_Using_CNN

This project focuses on recognizing American Sign Language (ASL) hand gestures for letters A-I using a Convolutional Neural Network (CNN).

## Project Overview

1. **Data Preparation**:  
   - Created a custom dataset of ASL hand gestures resized to 224x224 pixels.  
   - Split the dataset into training, validation, and test sets, ensuring the test set included unseen participants to maintain model integrity.

2. **Model Development**:  
   - Designed a CNN using PyTorch with:
     - Two convolutional layers
     - ReLU activation
     - Max-pooling layers
     - Fully connected layers  
   - Applied regularization techniques like dropout to prevent overfitting.

3. **Training and Tuning**:  
   - Trained the CNN with hyperparameter tuning (batch size, learning rate, dropout rate).  
   - Achieved a validation accuracy of **77%** and mitigated overfitting through iterative tuning.

4. **Transfer Learning**:  
   - Integrated **AlexNet** as a pretrained feature extractor.  
   - Designed a custom classifier trained on the extracted features to predict ASL gestures.  

## Technologies Used
- **PyTorch**: Model building and training.  
- **Torchvision**: Data augmentation and transfer learning.  
- **Google Colab**: Training environment and experimentation.

- [View Project](https://github.com/zaibaap/Portfolio/blob/5b9ddeb20ae567941858f16887fa1c978811e877/asl_hand_gesture_recognition_using_cnn.py)

### 2. **Cart-Pole Control Problem Using Reinforcement Learning**
   This project focuses on solving the Cart-Pole control problem using reinforcement learning techniques. The Cart-Pole environment is a classic benchmark problem for control algorithms. The goal is to balance a pole on a moving cart by applying forces to the cart.

   #### Project Overview:
   - **Environment**: Used the OpenAI Gym's `CartPole-v1` environment.
   - **Algorithms**:
     - **SARSA (On-Policy TD Control)**: Utilized SARSA to learn a policy by balancing exploration and exploitation.
     - **Q-Learning (Off-Policy TD Control)**: Implemented an off-policy approach to learn an optimal action-value function.
     - **Expected SARSA**: Enhanced stability and convergence by using the expected value of future action-value estimates.
   - **State Representation**: Discretized the continuous state space into bins for efficient Q-table updates.
   - **Performance**: Demonstrated improvements in total rewards across algorithms, comparing their efficacy in solving the task.

   #### Technologies Used:
   - **Python**: Core programming language.
   - **OpenAI Gym**: Environment for simulation and testing.
   - **NumPy**: Numerical computation and array manipulation.
   - **Matplotlib**: Visualization of agent performance.

- [View Project](https://github.com/zaibaap/Portfolio/blob/5b9ddeb20ae567941858f16887fa1c978811e877/Cartpole_Control_Problem_Reinforcement_Learning.ipynb)

### 3. Sentiment Analysis Models - Movie Reviews
- **Description**: 
- **Technologies**: Python, Scikit-learn, Matplotlib.
- [View Project](https://github.com/zaibaap/Portfolio/blob/5b9ddeb20ae567941858f16887fa1c978811e877/Sentiment_Analysis_Movie_Reviews_.py)

### 4. Portfolio OPtimization for Different Investment Strategies
- **Description**: 
- **Technologies**: Python, Scikit-learn, Matplotlib.
- [View Project](https://github.com/zaibaap/Portfolio/blob/5b9ddeb20ae567941858f16887fa1c978811e877/portf_optimization_w_diff_Inevestment_strats.ipynb)

## Contact
Feel free to reach out to me:
- GitHub: [zaibaap](https://github.com/zaibaap)
- Email: zaibaa.pathan@mail.utoronto.ca
