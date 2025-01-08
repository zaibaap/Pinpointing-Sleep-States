# Portfolio of Projects  

Welcome to my project portfolio! This repository showcases a collection of my previous work, highlighting my experience in data science and machine learning.  

---

## Table of Contents  
- [Projects](#projects)  
  - [ASL Hand Gesture Recognition Using CNN](#1-asl-hand-gesture-recognition-using-cnn)  
    - [Project Overview](#project-overview-1)  
    - [Technologies Used](#technologies-used-1)  
  - [Cart-Pole Control Problem Using Reinforcement Learning](#2-cart-pole-control-problem-using-reinforcement-learning)  
    - [Project Overview](#project-overview-2)  
    - [Technologies Used](#technologies-used-2)  
  - [Sentiment Analysis Models - Movie Reviews](#3-sentiment-analysis-models---movie-reviews)  
    - [Project Overview](#project-overview-3)  
    - [Technologies Used](#technologies-used-3)  
  - [Portfolio Optimization for Different Investment Strategies](#4-portfolio-optimization-for-different-investment-strategies)  
    - [Project Overview](#project-overview-4)  
    - [Technologies Used](#technologies-used-4)  
- [Contact](#contact)  

---

## Projects  

### 1. ASL Hand Gesture Recognition Using CNN  

#### Project Overview  
This project focuses on recognizing American Sign Language (ASL) hand gestures for letters A-I using a Convolutional Neural Network (CNN).  

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

#### Technologies Used  
- **PyTorch**: Model building and training.  
- **Torchvision**: Data augmentation and transfer learning.  
- **Google Colab**: Training environment and experimentation.  

[**View Project**](https://github.com/zaibaap/Portfolio/blob/5b9ddeb20ae567941858f16887fa1c978811e877/asl_hand_gesture_recognition_using_cnn.py)  

---

### 2. Cart-Pole Control Problem Using Reinforcement Learning  

#### Project Overview  
This project focuses on solving the Cart-Pole control problem using reinforcement learning techniques. The Cart-Pole environment is a classic benchmark problem for control algorithms. The goal is to balance a pole on a moving cart by applying forces to the cart.  

- **Environment**: Used the OpenAI Gym's `CartPole-v1` environment.  
- **Algorithms**:  
  - **SARSA (On-Policy TD Control)**: Utilized SARSA to learn a policy by balancing exploration and exploitation.  
  - **Q-Learning (Off-Policy TD Control)**: Implemented an off-policy approach to learn an optimal action-value function.  
  - **Expected SARSA**: Enhanced stability and convergence by using the expected value of future action-value estimates.  
- **State Representation**: Discretized the continuous state space into bins for efficient Q-table updates.  
- **Performance**: Demonstrated improvements in total rewards across algorithms, comparing their efficacy in solving the task.  

#### Technologies Used  
- **Python**: Core programming language.  
- **OpenAI Gym**: Environment for simulation and testing.  
- **NumPy**: Numerical computation and array manipulation.  
- **Matplotlib**: Visualization of agent performance.  

[**View Project**](https://github.com/zaibaap/Portfolio/blob/5b9ddeb20ae567941858f16887fa1c978811e877/Cartpole_Control_Problem_Reinforcement_Learning.ipynb)  

---

### 3. Sentiment Analysis Models - Movie Reviews  

#### Project Overview  
This project implements a Sentiment Analysis system to classify movie reviews as either positive or negative.  

1. **Text Analysis**: Extracted patterns in positive and negative sentiments.  
2. **Model Development**: Built multiple models, including:  
   - Word-level Long Short-Term Memory (LSTM) networks.  
   - Pre-trained embeddings like GloVe and BERT.  
3. **Evaluation**: Compared custom RNN-based models and pre-trained BERT embeddings for classification tasks.  

#### Technologies Used  
- **Data Preprocessing**: Pandas, NumPy, NLTK for text cleaning, tokenization, and feature extraction.  
- **Visualization**: Matplotlib, Seaborn.  
- **Machine Learning**:  
  - PyTorch for model building and training.  
  - Pre-trained embeddings like GloVe and Hugging Face's BERT.  
- **Optimization**: Adam optimizer, learning rate scheduling, dropout regularization.  

[**View Project**](https://github.com/zaibaap/Portfolio/blob/5b9ddeb20ae567941858f16887fa1c978811e877/Sentiment_Analysis_Movie_Reviews_.py)  

---

### 4. Portfolio Optimization for Different Investment Strategies  

#### Project Overview  
This project explores portfolio optimization using various investment strategies to manage risk and maximize returns.  

1. **Strategies Implemented**:  
   - Buy-and-Hold.  
   - Equally Weighted Portfolio.  
   - Minimum Variance Portfolio.  
   - Maximum Sharpe Ratio Portfolio.  
   - Equal Risk Contributions Portfolio.  

2. **Data Analysis**:  
   - Processed daily closing prices of multiple assets to calculate expected returns and covariance matrices.  
   - Simulated portfolio rebalancing across time periods with evolving market conditions.  

3. **Optimization**:  
   - Implemented advanced techniques like quadratic programming for minimum variance and Sharpe ratio optimization.  
   - Incorporated transaction cost modeling and ensured feasibility with cash constraints.  

4. **Scalability**:  
   - Designed to handle multiple assets and trading periods, supporting diverse investment scenarios.  

#### Technologies Used  
- **Python**: NumPy, Pandas, Matplotlib.  
- **Optimization Libraries**:  
  - CPLEX for solving quadratic programming problems.  
  - SciPy and CyIpopt for general optimization tasks.  

[**View Project**](https://github.com/zaibaap/Portfolio/blob/5b9ddeb20ae567941858f16887fa1c978811e877/portf_optimization_w_diff_Inevestment_strats.ipynb)  

---

## Contact
Feel free to reach out to me if you have any questions or concerns!
- GitHub: [zaibaap](https://github.com/zaibaap)
- Email: zaibaa.pathan@mail.utoronto.ca
