import sys
import csv

import numpy as np
import yahtzee as yah

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

categories = ["1","2","3","4","5","6","3K","4K","FH","SS","LS","C","Y"]

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
        features.extend([1,1])
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

    if roll.is_straight(4):
        lower.append(2)
    elif roll.is_straight(3):
        lower.append(1)
    else:
        lower.append(0)

    if roll.is_full_house():
        lower.append(2)
    elif is_two_pair(roll):
        lower.append(1)
    else:
        lower.append(0)
    norm_l = [float(i)/max(max(lower),1) for i in lower]

    norm_u.extend(norm_l)
    return norm_u

def are_same(seen_roll, returned_roll):
    # return similarity score
    if returned_roll.as_list() == []:
        if seen_roll.as_list() == []:
            return 1
        return 0

    #print(returned_roll.as_list(), seen_roll.as_list())
    sim = 0.0
    for i in range(6):
        sim += min(seen_roll.count(i+1), returned_roll.count(i+1))
    #print(sim / len(returned_roll.as_list()))
    return sim / len(returned_roll.as_list())

def select_n_kind(roll, existing_categories):
    if '3K' in existing_categories \
        and '4K' in existing_categories \
        and ('Y' in existing_categories \
        or 'Y+' not in existing_categories):
        return yah.YahtzeeRoll.parse("")

    max_val=0
    max_am=0
    for i in range(6):
        if roll.count(i+1) > max_am:
            max_am = roll.count(i+1)
            max_val = i+1
    return roll.select_all([max_val])

def select_straight(roll, sheet):
    if 'SS' not in sheet:
        runs = roll.longest_runs()
        if len(runs[0]) >= 3:
            return roll.select_one(runs[0])
        else:
            # choose between possibly multiple runs of 2; keep the higher
            # one if chance is open or it has strictly more open categories
            counts = [sum([(0 if str(n) in sheet else 1) for n in x]) for x in runs]
            run = runs[0]
            if len(runs) > 1 and ('C' not in sheet or counts[1] > counts[0]):
                run = runs[1]
            return roll.select_one(run)
    else:
        # keep the straight that we have the most of
        low = roll.select_one(range(1, 6))
        high = roll.select_one(range(2, 7))

        if len(low.as_list()) > len(high.as_list()):
            return low
        else:
            return high

def convert_reroll_to_labels(scoresheet, reroll, original_roll, rerolls):
    # if empty list returned
    if reroll == '[]':
        return [0,0,0,0,0,0,0,0,0,0,1] # keep nothing

    # if category returned, put in right category
    if reroll[0] != '[':
        r = [0] * 11
        if reroll == '1' or reroll == '2' or reroll == '3' or reroll == '4' or reroll == '5' or reroll == '6':
            r[int(reroll)-1] = 1
        elif reroll == '3K' or reroll == '4K' or reroll == 'Y' or reroll == 'Y+':
            r[6] = 1
        elif reroll == 'SS' or reroll == 'LS':
            r[7] = 1
        elif reroll == 'FH':
            r[8] = 1
        elif reroll == 'C':
            r[9] = 1
        else:
            r[10] = 1
        return r

    existing_categories = scoresheet.split(" ")
    roll = yah.YahtzeeRoll.parse(reroll[1:-1])
    numbers = diff_numbers(roll) # distinct numbers in roll
    r = [0] * 11

    max_sim = 0
    max_sim_cat = None
    for i in range(6):
        sim = are_same(roll, original_roll.select_all([i+1]))
        if sim == 1:
            r[i] = 1
            return r
        if sim > max_sim:
            max_sim = sim
            max_sim_cat = i

    sim = are_same(roll, select_n_kind(original_roll, existing_categories))
    if sim == 1:
        r[6] = 1
        return r
    if sim > max_sim:
        max_sim = sim
        max_sim_cat = 6

    sim = are_same(roll, select_straight(original_roll, existing_categories))
    if sim == 1:
        r[7] = 1
        return r
    if sim > max_sim:
        max_sim = sim
        max_sim_cat = 7

    sim = are_same(roll, original_roll.select_for_full_house())
    if sim == 1:
        r[8] = 1
        return r
    if sim > max_sim:
        max_sim = sim
        max_sim_cat = 8

    sim = are_same(roll, original_roll.select_for_chance(rerolls))
    if sim == 1:
        r[9] = 1
        return r
    if sim > max_sim:
        max_sim = sim
        max_sim_cat = 9
    #print(reroll)
    #print(max_sim)
    #print(max_sim_cat)
    r[max_sim_cat]=1
    return r

    '''
    # if keep them all, put in right category
    if len(reroll) == 7:
        if numbers == 1:   # ns of a kind
            if reroll[1] not in existing_categories:  # n is open
                r[int(reroll[1])-1] = 1
            else:                       # n is closed
                r[6] = 1
            return r
        if roll.is_straight(4):         # straight
            r[7] = 1
            return r
        if numbers == 2:                   # full house
            r[8] = 1
            return r
    elif numbers == 1:
        if s[1] not in existing_categories:  # n is open
            r[int(reroll[1])-1] = 1
        else:                               # n is closed
            r[6] = 1
        return r
    elif roll.is_straight(3) or is_three_non_adjacent(roll):
        r[7] = 1
        return r
    elif is_two_pair(roll) or numbers == 2:
        r[8] = 1
        return r
    else:
        r[10] = 1
    return r
    '''


def encode_input(scoresheet, roll, rerolls, x_all):
    # encode UPx
    up = ''.join(c for c in scoresheet[-2:] if c.isdigit())
    up_list = [0] * 6
    index = int(int(up)/10)-1

    if index != -1:
        up_list[index] = 1

    # encode reroll
    reroll_list = [int(x) for x in bin(int(rerolls))[2:]]
    complete_bin(2, reroll_list)

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
        a=convert_reroll_to_labels(row[0], row[3], yah.YahtzeeRoll.parse(row[1]), row[2])
        y_all.append(a)

        c+=1
        if c==50000:
            #print(row)
            #print(x_all[c-1])
            #print(y_all[c-1])
            break

    features = len(x_all[0])
    norm_low = 0.0
    norm_high = 1.0
    mins = [0.0] * features
    maxes = [0.0] * features

    for i in range(0, features):
        mins[i] = min([x[i] for x in x_all])
        maxes[i] = max([x[i] for x in x_all])

    x_norm = [[(x[i] - mins[i]) / (maxes[i] - mins[i]) * (norm_high - norm_low) + norm_low for i in range(0, features)] for x in x_all]
    # split into training data and test data
    test_size = int(len(x_norm) / 5)
    train_size = len(x_norm) - test_size

    x_train = np.matrix(x_norm[:train_size])
    y_train = np.matrix(y_all[:train_size])

    x_test = np.matrix(x_norm[train_size:])
    y_test = y_all[train_size:]

    # set the topology of the neural network
    model = Sequential()
    model.add(Dense(300, activation="relu", input_dim = features))
    model.add(Dropout(0.1))
    model.add(Dense(y_train.shape[1], activation = "softmax"))

    # set up optimizer
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd)

    # train!
    model.fit(x_train, y_train, epochs=100, batch_size=50)

    # get predictions (one-hot encoded row for each test input)
    # convert to class 0, 1, 2 by finding index of maximum value;
    y_predict = [max(enumerate(y), key=lambda x:x[1])[0] for y in model.predict(x_test)]

    # do the same for test data
    y_correct = [max(enumerate(y), key=lambda x:x[1])[0] for y in y_test]

    # get list of (prediction, expectation) pairs, convert to list of
    # 1's where equal, 0's where unequal, and sum result to get number
    # of correct predictions
    print(sum((1 if y[0] == y[1] else 0) for y in zip(y_predict, y_correct)) / len(y_predict))

    return model

class NNStrategy:
    ''' takes the object returned from train as its parameter '''

    #def __init__(self, model):
        #self.net = model
    def __init__(self, model):
        self.net = model

    '''
        inputs: a scoresheet, a roll, and # of rerolls
        returns: a subroll of the roll
    '''
    def choose_dice(self, sheet, roll, rerolls):

        x = []
        scoresheet = sheet.as_state_string()
        input = encode_input(scoresheet, roll, rerolls, x)
        existing_categories = scoresheet.split(" ")

        y = self.net.predict(np.matrix(input))
        label = np.argmax(y[0])
        #print("Net output: ", y)
        #TODO: fix THESE

        if label <= 5:
            return roll.select_all([label + 1])
        elif label == 6 and (('3K' not in existing_categories and roll.is_n_kind(3)) \
            or ('4K' not in existing_categories and roll.is_n_kind(4)) \
            or ('Y' not in existing_categories and roll.is_n_kind(5)) \
            or ('Y+' in existing_categories and roll.is_n_kind(5))):
                return roll.select_for_n_kind(sheet, rerolls)
        elif label == 7 and ('SS' not in existing_categories or 'LS' not in existing_categories):
            return roll.select_for_straight(sheet)
        elif label == 8 and 'FH' not in existing_categories:
            return roll.select_for_full_house()
        elif label == 10:
            return yah.YahtzeeRoll.parse("")
        elif rerolls == 1:
            return roll.select_for_chance(rerolls)
##new
        '''
        if '3K' not in existing_categories and roll.is_n_kind(2) \
        or '4K' not in existing_categories and roll.is_n_kind(3) \
        or 'Y' not in existing_categories and roll.is_n_kind(4):
            for i in range(6,0,-1):
                if roll.count(i) > 3:
                    return roll.select_all([i])
            for i in range(6,0,-1):
                if roll.count(i) > 2:
                    return roll.select_all([i])
            for i in range(6,0,-1):
                if roll.count(i) > 1:
                    return roll.select_all([i])
        '''
##endnenw
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
        #print("Net output: ", y)

        #TODO: fix THESE

        if label <= 6:
            if roll.is_n_kind(5) and 'Y' not in existing_categories and 'Y+' not in existing_categories:

                return categories.index('Y')
            elif roll.is_n_kind(5) and 'Y+' in existing_categories:
                for i in range(11, -1, -1): # in bonuns Yahtzee, return some other category, but not Y
                    if categories[i] not in existing_categories:

                        return categories.index(categories[i])
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
        # => score a 0 in hard to achieve categories

        for i in range(6, 0, -1):
            if roll.count(i) >= 2 and str(i) not in existing_categories:
                return categories.index(str(i))

        for i in range(11, -1, -1):
            #print(scoresheet)
            if categories[i] not in existing_categories:
                #print('k')
                #print('I AM RETURNING '+ categories[i] + ' num '+str(categories.index(categories[i])))
                return categories.index(categories[i])
        if 'Y' not in existing_categories and 'Y+' not in existing_categories:
            #print('j')
            return categories.index('Y')
