# RockPaperScissorsAI
A Rock Paper Scissors program that lets you play against a Neural Network AI.  
The Neural Network uses trigrams of previous games to predict what the user's next move will be.  
To play a game, enter the function 'game()',  
  
The game(function) takes the following arguments:  
  
-deep - True or False, determines whether the AI will be based on the deep learning model, or choose its moves randomly  
-model - The model that is used, if you want to use the deep learning model, set this parameter to 'model=model'  
-rounds - number of rounds you want to play  
  
The model retrains on the updated dataset after every round
