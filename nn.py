import sys
import csv

import numpy as np
import yahtzee as yah

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

categories = ["1","2","3","4","5","6","3K","4K","FH","SS","LS","C","Y"]
goals = [3, 6, 9, 12, 15, 18, 20, 10, 15, 20, 15, 25, 10]

''' Helper Functions '''
def is_two_pair(roll):
    pairs = 0
    for i in range(6):
        if roll.count(i) >=2:
            pairs += 1
    return pairs >= 2

def is_three_non_adjacent(roll):
    consec = 0
    n = 4
    second_chance = True
    i = 1
    while i <= 6 and consec < n:
        if roll.count(i) > 0:
            consec += 1
        else:
            if second_chance:
                consec += 1
                second_chance = False
            else:
                consec = 0
                second_chance = True
        i += 1
    return consec == n

def complete_bin(intended_length, list):
    original_length = len(list)
    if original_length < intended_length:
        for j in range(intended_length - original_length):
            list.insert(0, 0)

def diff_numbers(roll):
    diff = 0
    for i in range(6):
        if roll.count(i+1) > 0:
            diff += 1
    return diff

def convert_categories(s):
    features = []
    scoresheet = s.split(" ")

    for i in range(12):
        if categories[i] in scoresheet:
            features.append(1)
        else:
            features.append(0)

    if "Y" in scoresheet:
        features.extend([1,0])
    elif "Y+" in scoresheet:
        features.extend([0,1])
    else:
        features.extend([0,0])

    return features # 14

''' Encoding Functions '''

def convert_roll_to_categories(roll):
    upper = []
    lower = []

    for i in range(6):
        upper.append(roll.count(i+1))
    norm_u = [float(i)/max(upper) for i in upper]

    if roll.is_straight(5):
        lower.append(1)
    elif roll.is_straight(4):
        lower.append(0.75)
    elif roll.is_straight(3):
        lower.append(0.5)
    elif roll.is_straight(2):
        lower.append(0.25)
    else:
        lower.append(0)

    if roll.is_full_house():
        lower.append(1)
    elif is_two_pair(roll):
        lower.append(0.75)
    else:
        lower.append(0)

    norm_u.extend(lower)
    return norm_u

def convert_reroll_to_labels(scoresheet, s):
    # if empty list returned
    if s == '[]':
        return [0,0,0,0,0,0,0,0,0,0,1] # keep nothing

    # if category returned, put in right category
    if s[0] != '[':
        r = [0] * 11
        if s == '1' or s == '2' or s == '3' or s == '4' or s == '5' or s == '6':
            r[int(s)-1] = 1
        elif s == '3K' or s == '4K' or s == 'Y' or s == 'Y+':
            r[6] = 1
        elif s == 'SS' or s == 'LS':
            r[7] = 1
        elif s == 'FH':
            r[8] = 1
        elif s == 'C':
            r[9] = 1
        return r

    existing_categories = scoresheet.split(" ")
    roll = yah.YahtzeeRoll.parse(s[1:-1])
    numbers = diff_numbers(roll) # distinct numbers in roll
    r = [0] * 11

    # shorter reroll
    if numbers == 1:
        if s[1] not in existing_categories:  # n is open
            r[int(s[1])-1] = 1
        elif len(s) > 1 and '3K' not in existing_categories \
            or '4K' not in existing_categories \
            or 'Y' not in existing_categories \
            or 'Y+' in existing_categories:                               # n is closed
            r[6] = 1
        return r
    elif (roll.is_straight(3) or is_three_non_adjacent(roll)) and ('SS' not in existing_categories or 'LS' not in existing_categories):
        r[7] = 1
        return r
    elif (is_two_pair(roll) or (numbers == 2 and len(s) >= 3)) and 'FH' not in existing_categories:
        r[8] = 1
        return r
    elif (('3K' not in existing_categories and roll.is_n_kind(2)) \
        or ('4K' not in existing_categories and roll.is_n_kind(3)) \
        or ('Y' not in existing_categories and roll.is_n_kind(4)) \
        or ('Y+' in existing_categories and roll.is_n_kind(4))):
        r[6] = 1
        return r
    elif 'C' not in existing_categories:
        r[9] = 1            # TODO: UNSURE
        return r             #r[10] = 1  should probably nenver happen
    else:
        max_val=0
        max_am=0
        for i in range(6):
            if roll.count(i+1) >= max_am and str(i+1) not in existing_categories:
                max_am = roll.count(i+1)
                max_val = i+1
        if max_val != 0:
            r[int(max_val-1)] = 1
            return r

    if numbers >=2 and ('SS' not in existing_categories or 'LS' not in existing_categories):
        r[7] = 1
        return r

def encode_input(scoresheet, roll, rerolls, x_all):
    # encode UPx
    up = ''.join(c for c in scoresheet[-2:] if c.isdigit())
    up_list = [0] * 6
    index = int(int(up)/10)-1

    if index != -1:
        up_list[index] = 1

    # encode reroll
    reroll_list = [0] * 3
    reroll_list[2 - int(rerolls)] = 1

    # encode categories
    enc = convert_categories(scoresheet)
    enc.extend(up_list)

    # encode rolling
    a=convert_roll_to_categories(roll)

    enc.extend(a)
    enc.extend(reroll_list)
    x_all.append(enc)
    return enc

''' Reads your training examples from standard input, returns a trained neural network '''
def train():
    x_all = []
    y_all = []

    # read the data
    reader = csv.reader(sys.stdin)
    c = 0
    for row in reader:
        # row[0] scoresheet, row[1] roll, row[2] # rerolls, row[3] reroll
        encode_input(row[0], yah.YahtzeeRoll.parse(row[1]), row[2], x_all)
        a=convert_reroll_to_labels(row[0], row[3])
        y_all.append(a)

    features = len(x_all[0])
    norm_low = 0.0
    norm_high = 1.0
    mins = [0.0] * features
    maxes = [0.0] * features

    for i in range(0, features):
        mins[i] = min([x[i] for x in x_all])
        maxes[i] = max([x[i] for x in x_all])

    x_norm = x_all
    train_size = len(x_norm)

    x_train = np.matrix(x_norm)
    y_train = np.matrix(y_all)

    # set the topology of the neural network
    model = Sequential()
    model.add(Dense(300, activation="relu", input_dim = features))
    model.add(Dropout(0.1))
    model.add(Dense(y_train.shape[1], activation = "softmax"))

    # set up optimizer
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd)

    # train!
    model.fit(x_train, y_train, epochs=100, batch_size=50)

    return model

class NNStrategy:
    ''' takes the object returned from train as its parameter '''

    def __init__(self, model):
        self.net = model

    def choose_dice(self, sheet, roll, rerolls):

        x = []
        scoresheet = sheet.as_state_string()
        input = encode_input(scoresheet, roll, rerolls, x)
        existing_categories = scoresheet.split(" ")

        y = self.net.predict(np.matrix(input))
        label = np.argmax(y[0])

        if label <= 5 and str(label+1) not in existing_categories:
            return roll.select_all([label + 1])
        elif label == 6 and (('3K' not in existing_categories and roll.is_n_kind(2)) \
            or ('4K' not in existing_categories and roll.is_n_kind(3)) \
            or ('Y' not in existing_categories and roll.is_n_kind(4)) \
            or ('Y+' in existing_categories and roll.is_n_kind(4))):
                return roll.select_for_n_kind(sheet, rerolls)
        elif label == 7 and ('SS' not in existing_categories or 'LS' not in existing_categories):
            return roll.select_for_straight(sheet)
        elif label == 8 and 'FH' not in existing_categories:
            return roll.select_for_full_house()
        elif label == 10:
            return yah.YahtzeeRoll.parse("")
        elif rerolls == 1:
            return roll.select_for_chance(rerolls)

        for i in range(6,0,-1):
            if roll.count(i) > 1 and str(i) not in existing_categories:
                return roll.select_all([i])

        return yah.YahtzeeRoll.parse("")

    '''
        inputs: a scoresheet, a roll
        returns: index of an unused category on that scoresheet
    '''
    def choose_category(self, sheet, roll):

        x = []
        scoresheet = sheet.as_state_string()
        existing_categories = scoresheet.split(" ")

        input = encode_input(scoresheet, roll, 0, x)
        y = self.net.predict(np.matrix(input))
        label = np.argmax(y[0])

        if label <= 6:
            if roll.is_n_kind(5) and 'Y' not in existing_categories and 'Y+' not in existing_categories:

                return categories.index('Y')
            elif roll.is_n_kind(5) and 'Y+' in existing_categories:
                min_val = 50
                min_cat = None
                for i in range(11, -1, -1):
                    if categories[i] not in existing_categories:
                        if goals[i] < min_val:
                            min_val = goals[i]
                            min_cat = categories[i]
                return categories.index(min_cat)
            elif roll.is_n_kind(4) and '4K' not in existing_categories:

                return categories.index('4K')
            elif roll.is_n_kind(3) and '3K' not in existing_categories:

                return categories.index('3K')

            #TODO; de xreiazetai na kano return kati an exo Chance left because
            #i'm ruining a choose_category
            if label <= 5 and str(label+1) not in existing_categories:

                return categories.index(str(label+1))
        elif label == 7:
            if roll.is_straight(5) and 'LS' not in existing_categories:

                return categories.index('LS')
            elif roll.is_straight(4) and 'SS' not in existing_categories:

                return categories.index('SS')
        elif label == 8 and roll.is_full_house() and 'FH' not in existing_categories:

            return categories.index('FH')
        elif label == 9 and 'C' not in existing_categories:

            return categories.index('C')

        # At this point, we have failed to score full points in any category
        # Return category with min goal value (minimum expected payoff)
        #TODO: Use Goal Scores HERE

        min_val = 50
        min_cat = None

        for i in range(12, -1, -1):
            if categories[i] not in existing_categories:
                if i == 12 and 'Y+' in existing_categories:
                    continue
                if goals[i] < min_val:
                    min_val = goals[i]
                    min_cat = categories[i]
        return categories.index(min_cat)
