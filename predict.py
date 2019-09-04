#Import scikit-learn dataset library
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import datasets
import progressbar

def declare_models_dis(fatigue_dict):
    #Load dataset
    rpe_dataset = pd.read_csv('data/rpe/rpe_data_day1_data.csv')
    well_dataset = pd.read_csv('data/wellness/wellness_data_day1_data.csv')
    load_dataset = pd.read_csv('data/load/load_data.csv')

    # traversing through dataframe
    # and converting values to numberes
    # based on given values

    rpe_dataset.SessionType[dataset.SessionType == 'Combat'] = 1
    rpe_dataset.SessionType[dataset.SessionType == 'Conditioning'] = 2
    rpe_dataset.SessionType[dataset.SessionType == 'Game'] = 3
    rpe_dataset.SessionType[dataset.SessionType == 'Mobility/Recovery'] = 4
    rpe_dataset.SessionType[dataset.SessionType == 'Skills'] = 5
    rpe_dataset.SessionType[dataset.SessionType == 'Speed'] = 6
    rpe_dataset.SessionType[dataset.SessionType == 'Strength'] = 7

    wel_dataset.Nutrition[dataset.Nutrition == 'Excellent'] = 1
    wel_dataset.Nutrition[dataset.Nutrition == 'Poor'] = 2
    wel_dataset.Nutrition[dataset.Nutrition == 'Okay'] = 3
    wel_dataset.Nutrition[pd.isnull(dataset.Nutrition)] = 4

    # Creating a DataFrame of given dataset.
    data = pd.DataFrame({
        'SessionType': rpe_dataset['SessionType'],
        'Duration': rpe_dataset['Duration'],
        'SleepHours': well_dataset['SleepHours'],
        'Nutrition': well_dataset['Nutrition'],
        'Soreness': well_dataset['Soreness'],
        'Fatigue': well_dataset['Fatigue'],
        'Load': load_dataset['Load']
    })

    X = data[['SessionType',
              'Duration', 'SleepHours', 'Nutrition', 'Soreness']]  # Features
    y = data['Distance']  # Labels

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)  # 70% training and 30% test
    #Import Random Forest Model

    #Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    # Predict the Y col
    y_pred = clf.predict(X_test)

    # Calling the prediction method
    predictDis(clf, fatigue_dict)

def predictDis(clf, fatigue_dict):
    top_distance = 0
    # practice = 0
    session = 0
    duration = 0
    sleep_hours = 0
    distance = 0
    nutrition = 0
    # sleep_quality = 0
    soreness = 0
    fat = 0

    print("\n\nTesting Distance")
    # Iterating over the dictionary
    for key, value in progressbar.progressbar(fatigue_dict.items()):
        current = key[0]
        if (int(current) > 4):
            se = value[0]
            du = value[1]
            sl = value[2]
            n = value[3]
            so = value[4]
            current_distance = clf.predict([[se, du, sl, n, so]])
            if (current_distance > top_distance):
                top_distance = current_distance
                # make the variables here to printout
                # practice = pr
                session = se
                duration = du
                sleep_hours = sl
                # sleep_quality = q
                nutrition = n
                soreness = so
                fat = current

    print("Lowest Fatigue day of Game: " + fat)
    print("Greatest Fat Nut Load day of Game: " + str(top_distance))
    print("Session current day: " + str(session))
    print("Duration current day: " + str(duration))
    print("Sleep Hours current day: " + str(sleep_hours))
    print("Soreness day of Game: " + str(soreness))
    print("Nutrition current day: " + str(nutrition))

def predictFat(clf):
    # Declaring ranges for each feature
    practice_range = range(0, 1)
    session_range = range(1, 7)
    duration_range = range(2, 245)
    sleep_range = range(0, 16)
    nutrition_range = range(1, 4)
    soreness_range = range(1, 7)

    lowest_fatigue = 100
    practice = 0
    session = 0
    duration = 0
    sleep_hours = 0
    distance = 0

    count_for_dict = 0

    fatigue_dict = {}

    count = 0

    # Running loops over all possibilites for values
    # in the given range
    print("\n\nTesting Fatigue")
    for se in progressbar.progressbar(session_range):
        for n in progressbar.progressbar(nutrition_range):
            for so in progressbar.progressbar(soreness_range):
                for du in progressbar.progressbar(duration_range):
                    for sl in sleep_range:
                        cur_fatigue = clf.predict([[se, du, sl, n, so]])
                        # Checking if this fatiugue is lower
                        fatigue_dict[str(cur_fatigue[0].astype(int)) + str(count_for_dict)] = [
                            se, du, sl, n, so]
                        count_for_dict = count_for_dict + 1
                        print(str(count) + "/419893")
                        count = count + 1
    # Calling the next comaprison
    declare_models_dis(fatigue_dict)


def declare_models_fat():
    #Load dataset
    rpe_dataset = pd.read_csv('data/rpe/rpe_data_day1_data.csv')
    well_dataset = pd.read_csv('data/wellness/wellness_data_day1_data.csv')
    load_dataset = pd.read_csv('data/load/load_data.csv')

    # traversing through dataframe
    # and converting values to numberes
    # based on given values

    rpe_dataset.SessionType[rpe_dataset.SessionType == 'Combat'] = 1
    rpe_dataset.SessionType[rpe_dataset.SessionType == 'Conditioning'] = 2
    rpe_dataset.SessionType[rpe_dataset.SessionType == 'Game'] = 3
    rpe_dataset.SessionType[rpe_dataset.SessionType == 'Mobility/Recovery'] = 4
    rpe_dataset.SessionType[rpe_dataset.SessionType == 'Skills'] = 5
    rpe_dataset.SessionType[rpe_dataset.SessionType == 'Speed'] = 6
    rpe_dataset.SessionType[rpe_dataset.SessionType == 'Strength'] = 7

    well_dataset.Nutrition[well_dataset.Nutrition == 'Excellent'] = 1
    well_dataset.Nutrition[well_dataset.Nutrition == 'Poor'] = 2
    well_dataset.Nutrition[well_dataset.Nutrition == 'Okay'] = 3
    well_dataset.Nutrition[pd.isnull(well_dataset.Nutrition)] = 4

    # Creating a DataFrame of given dataset.
    data = pd.DataFrame({
        'SessionType': rpe_dataset['SessionType'],
        'Duration': rpe_dataset['Duration'],
        'SleepHours': well_dataset['SleepHours'],
        'Nutrition': well_dataset['Nutrition'],
        'Soreness': load_dataset['Soreness'],
        'Fatigue': load_dataset['Fatigue'],
        'Load': load_dataset['Load']
    })

    X = data[['SessionType',
              'Duration', 'SleepHours', 'Nutrition', 'Soreness']]  # Features
    y = data['Fatigue']  # Labels

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)  # 70% training and 30% test
    #Import Random Forest Model

    #Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    # Predict the Y col
    y_pred = clf.predict(X_test)

    # Calling the prediction method
    predictFat(clf)


declare_models_fat()
