# DataFest 2018

DataFest a data hackathon for undergraduate students sponsored by the American Statistical Association and hosted by RIT. 

For the given year we had to find something cool in a rich dataset of rugby players. 

We decided to predict the ideal workout and wellness plan for a given player the day before a game/tournament. 

We broke the problem down to the following bullet points

1. Minimize Fatigue the day of a game
2. Maximize Load (Measurement of players activity) day of the game
3. Random Forest Regressor predict fatigue values for all possibilities of workout and wellness plans
4. From the predicted fatigue if it is from 4-7 (above average) predict the total load for that game day
5. Gives us the ideal wellness and workout plan

Then we gathered the data and using two Random Forest Regression models with features of SleepHours, Nutrition, Soreness(Day of the Game), SessionType, Duration and targeting the Load of a player we produced some cool results (Note not all features are filled in because we did not have enough time to train the model, but you could). 

<img href="results.png">