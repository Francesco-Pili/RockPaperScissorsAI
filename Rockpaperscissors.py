import numpy as np
import time



movesdict = {"r":np.array([1,0,0]),
            "p":np.array([0,1,0]),
            "s":np.array([0,0,1])}
inttomove = {0:"Rock",
            1:"Paper",
            2:"Scissors"}

last3games= [np.zeros((6),dtype=int)] *3

def game(deep=False, model = None, rounds=1):
    for i in range(rounds):
        while True:
            try:
                playerchoice = movesdict[input("Rock, Paper, or Scissors?")]
                break
            except KeyError:
                print("Not a valid choice!")
                time.sleep(0.2)

        #computer choice    
        if deep:
            predplayermove= model.predict_classes([[np.array(last3games)]])[0]
            if predplayermove == 0:
                computerchoice = movesdict["p"]
            if predplayermove == 1:
                computerchoice = movesdict["s"]
            if predplayermove == 2:
                computerchoice = movesdict["r"]
        else:
            
            computerchoice = list(movesdict.items())[np.random.randint(0,3)][1]
        

            
        humanwins=0
        compwins=0
        draws=0
        print(f"You chose {inttomove[np.argmax(playerchoice)]}")
        time.sleep(0.2)
        print("I chose...")
        time.sleep(1)
        print(f"{inttomove[np.argmax(computerchoice)]}")
        time.sleep(1)
        if np.all(playerchoice == computerchoice):
            print("Draw!")
            draws+=1
        #player chooses rocj
        if np.all(playerchoice == movesdict["r"]):
            if np.all(computerchoice == movesdict["p"]):
                print("You lost!")
                compwins+=1
            if np.all(computerchoice == movesdict["s"]):
                print("You won!")
                humanwins+=1
        if np.all(playerchoice == movesdict["p"]):
            if np.all(computerchoice == movesdict["s"]):
                print("You lost!")
                compwins+=1
            if np.all(computerchoice == movesdict["r"]):
                print("You won!")
                humanwins+=1
        if np.all(playerchoice == movesdict["s"]):
            if np.all(computerchoice == movesdict["r"]):
                print("You lost!")
                compwins+=1
            if np.all(computerchoice == movesdict["p"]):
                print("You won!")
                humanwins+=1
        playedgames.append(np.array([playerchoice,computerchoice]).reshape(6))
        del last3games[0]
        last3games.append(np.array([playerchoice,computerchoice]).reshape(6))
        
        if deep:
            np.save("RPSdata.npy", np.array(playedgames))
            X,y = compilefromgames(playedgames)
            model.fit(X,y,batch_size=10,epochs=200,verbose=0)
            model.save_weights('RPS.h5')
        
    
def compilefromgames(gamelist):
    X= []
    y=[]
    t= [np.zeros((6),dtype=int)] *3
    d= t +playedgames
    for i, g in enumerate(d[:-3]):
        X.append(d[i:i+3])
        y.append(d[i+3][:3])
    return np.array(X), np.array(y)


def loadmodel():
    model= Sequential()
    model.add(LSTM(66,input_shape=(3, 6)))
    model.add(Dense(132,activation='relu'))
    model.add(Dense(60,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    opt = Adam(lr=0.004)
    model.load_weights('RPS.h5')
    model.compile(optimizer=opt,loss='categorical_crossentropy', metrics= ['accuracy'])
    return model

AI = {"y":True,
     "n":False}[input("Play with AI? y/n").lower()]

if AI:
    playedgames = list(np.load("RPSdata.npy"))
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from keras.optimizers import Adam
    model =loadmodel()
    
