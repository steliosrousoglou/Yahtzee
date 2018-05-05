from multiset import Multiset
import random
import nn

class YahtzeeRoll:
    ''' The outcome of rolling 0 through 5 6-sided dice.
    '''

    def __init__(self):
        ''' Creates an empty roll.
        '''
        self.dice = Multiset(6)

    def reroll(self):
        ''' Adds dice showing uniformly randomly selected numbers to this roll until
            this roll has 5 dice.
        '''
        self.dice.add_random(5 - self.dice.size())


    def subroll(self, other):
        ''' Determines if this roll is a subset of the given roll.

            other -- a Yahtzee roll
        '''
        return self.dice.subset(other.dice)


    def count(self, num):
        ''' Determines how mnay dice are showing the given number.

            num -- an integer
        '''
        return self.dice.count(num - 1)


    def total(self):
        ''' Returns the total showing on the dice.
        '''
        # total in the dice + 1 for each to account for 0...5 vs 1...6
        return self.dice.total() + self.dice.size()


    def is_n_kind(self, n):
        ''' Determines if this roll is n-of-a-kind.

            n -- a positive integer
        '''
        i = 1
        while i <= 6 and self.count(i) < n:
            i += 1
        return i <= 6


    def is_full_house(self):
        ''' Determines if this roll is a full house.
        '''
        double = False
        triple = False
        i = 1
        while i <= 6 and not (double and triple) and self.count(i) in [0, 2, 3]:
            if self.count(i) == 2:
                double = True
            elif self.count(i) == 3:
                triple = True
            i += 1
        return double and triple


    def is_straight(self, n):
        ''' Determines if this roll is a straight of at least the given length.

            n -- a positive integer
        '''
        consec = 0
        i = 1
        while i <= 6 and consec < n:
            if self.count(i) > 0:
                consec += 1
            else:
                consec = 0
            i += 1
        return consec == n


    @staticmethod
    def parse(str):
        ''' Returns a roll containing dice showing the numbers corresponding to the
            digits in the given string.

            str -- a string containing up to 5 digits, each in the range 1 through 6
        '''
        if len(str) > 5:
            raise ValueError("length must be at most 5: " + str)

        result = YahtzeeRoll()
        for digit in str:
            if not digit.isdigit():
                raise ValueError("invalid digit in " + str)
            num = int(digit)
            if num < 1 or num > 6:
                raise ValueError("invalid digit in " + str)
            result.dice.add(num - 1)
        return result


    def as_list(self):
        ''' Returns a list of the numbers showing in this roll.'''
        return [x + 1 for x in self.dice.as_list()]


    def select_all(self, nums, maximum=5):
        ''' Returns the subroll of this roll that contains all occurrences
            of the given numbers.

            nums -- a list of integers betwen 1 and 6
        '''
        keep = ""
        for n in nums:
            if n < 1 or n > 6:
                raise ValueError("value out of range in " + str(nums))
            for i in range(min(self.count(n), maximum)):
                keep = keep + str(n)
        return YahtzeeRoll.parse(keep)


    def select_one(self, nums):
        ''' Returns the subroll of this roll that contains one occurrence
            of each of the given numbers that are also in this roll.

            nums -- a list of integers betwen 1 and 6
        '''
        keep = ""
        for n in nums:
            if n < 1 or n > 6:
                raise ValueError("value out of range in " + str(nums))
            if self.count(n) > 0:
                keep = keep + str(n)
        return YahtzeeRoll.parse(keep)


    def select_for_chance(self, rerolls):
        ''' Returns the subroll of this roll that maximizes the expected
            score in chance.

            rerolls -- 1 or 2
        '''
        if rerolls == 2:
            return self.select_all([5, 6])
        else:
            return self.select_all([4, 5, 6])


    def select_for_full_house(self):
        ''' Returns a subroll of this roll that gives the chance of
            obtaining a full house.
        '''
        # keep up to three of numbers we have at least 2 of
        keep = []
        for i in range(1, 7):
            if self.count(i) >= 2:
                keep.append(i)
        return self.select_all(keep, 3)


    def select_for_straight(self, sheet):
        ''' Returns a subroll that gives a good chance of obtaining
            the longest straight left unmarked on the given scoresheet.

            sheet -- a Yahtzee scoresheet
        '''
        # if SS is open, keep longest run
        if not sheet.is_marked(YahtzeeScoresheet.SMALL_STRAIGHT):
            runs = self.longest_runs()
            if len(runs[0]) >= 3:
                return self.select_one(runs[0])
            else:
                # choose between possibly multiple runs of 2; keep the higher
                # one if chance is open or it has strictly more open categories
                counts = [sum([(0 if sheet.is_marked(n - 1) else 1) for n in x]) for x in runs]
                run = runs[0]
                if len(runs) > 1 and (not sheet.is_marked(YahtzeeScoresheet.CHANCE) or counts[1] > counts[0]):
                    run = runs[1]
                return self.select_one(run)
        else:
            # keep the straight that we have the most of
            low = self.select_one(range(1, 6))
            high = self.select_one(range(2, 7))

            if len(low.as_list()) > len(high.as_list()):
                return low
            else:
                return high

#12346

    def longest_runs(self):
        ''' Returns a list of all the longest consecutive runs in this
            roll.  For example, if this roll is [1 2 4 4 5] then the
            list returned is [[1, 2], [4, 5]].
        '''
        runs = []
        longest = 0
        curr_len = 0
        for i in range(1, 7):
            if self.count(i) > 0:
                curr_len += 1
                if curr_len == longest:
                    runs.append(list(range(i - curr_len + 1, i + 1)))
                elif curr_len > longest:
                    runs = [list(range(i - curr_len + 1, i + 1))]
                    longest = curr_len
            else:
                curr_len = 0
        return runs


    def select_for_n_kind(self, sheet, rerolls):
        ''' Returns the subroll that maximizes expected score in
            3K, 4K, or Y
        '''
        max_keep = 5
        if not sheet.is_marked(YahtzeeScoresheet.FOUR_KIND) and sheet.is_marked(YahtzeeScoresheet.YAHTZEE) and sheet.scores[YahtzeeScoresheet.YAHTZEE] == 0:
            max_keep = 4
        elif not sheet.is_marked(YahtzeeScoresheet.THREE_KIND) and sheet.is_marked(YahtzeeScoresheet.FOUR_KIND) and sheet.is_marked(YahtzeeScoresheet.YAHTZEE) and sheet.scores[YahtzeeScoresheet.YAHTZEE] == 0:
            max_keep = 3
        high_freq = 0
        most_freq = None
        for i in range(1, 7):
            if self.count(i) >= high_freq:
                high_freq = self.count(i)
                most_freq = i

        keep_nums = [most_freq]

        # keep 4's, 5's, and 6's if aready have what we need
        # (4's only if down to last reroll)
        if ((max_keep == 3 and sheet.score(YahtzeeScoresheet.THREE_KIND, self) > 0)
            or (max_keep == 4 and sheet.score(YahtzeeScoresheet.FOUR_KIND, self) > 0)):
            for i in range(3 + rerolls, 7):
                if i != most_freq:
                    keep_nums.append(i)

        return self.select_all(keep_nums, max_keep)


    def __str__(self):
        ''' Returns a string representation of this roll.
        '''
        return str(self.as_list())

class YahtzeeScoresheet:
    ''' A standard Yahtzee scoresheet.
    '''
    # category indices
    THREE_KIND = 6
    FOUR_KIND = 7
    FULL_HOUSE = 8
    SMALL_STRAIGHT = 9
    LARGE_STRAIGHT = 10
    CHANCE = 11
    YAHTZEE = 12

    categories = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "3K",
        "4K",
        "FH",
        "SS",
        "LS",
        "C",
        "Y"
    ]

    def __init__(self):
        self.categories = YahtzeeScoresheet.categories

        # the functions that determine scoring
        self.rules = [
            self.upper(1),
            self.upper(2),
            self.upper(3),
            self.upper(4),
            self.upper(5),
            self.upper(6),
            self.n_kind(3),
            self.n_kind(4),
            self.full_house(25),
            self.straight(4, 30),
            self.straight(5, 40),
            self.n_kind(1),
            self.yahtzee(50)
        ]

        # the functions that update subtotals and bonuses
        self.totals = [
            self.upper_total(),
            self.lower_total(),
            self.yahtzee_bonus()
        ]

        self._upper_total = 0
        self._upper_bonus = 0
        self._lower_total = 0
        self._yahtzee_bonus = 0
        self._turns = 0

        self.scores = [None] * len(self.categories)


    def upper(self, num):
        def points(roll):
            return num * roll.count(num)
        return points


    def n_kind(self, count):
        def points(roll):
            if roll.is_n_kind(count):
                return roll.total()
            else:
                return 0
        return points


    def straight(self, count, score):
        def points(roll):
            if roll.is_straight(count) or self.is_joker(roll):
                return score
            else:
                return 0
        return points


    def full_house(self, score):
        def points(roll):
            if roll.is_full_house() or self.is_joker(roll):
                return score
            else:
                return 0
        return points


    def is_joker(self, roll):
        return (roll.is_n_kind(5)
                and self.scores[YahtzeeScoresheet.YAHTZEE] is not None
                and self.scores[YahtzeeScoresheet.YAHTZEE] > 0
                and self.scores[roll.as_list()[0] - 1] is not None)


    def yahtzee(self, score):
        def points(roll):
            if roll.is_n_kind(5):
                return score
            else:
                return 0
        return points


    def upper_total(self):
        def update(cat, roll, score):
            if cat < 6:
                self._upper_total += score
                if self._upper_total >= 63:
                    self._upper_bonus = 35
        return update


    def lower_total(self):
        def update(cat, roll, score):
            if cat >= 6:
                self._lower_total += score
        return update


    def yahtzee_bonus(self):
        def update(cat, roll, score):
            if cat != YahtzeeScoresheet.YAHTZEE and roll.is_n_kind(5) and self.scores[YahtzeeScoresheet.YAHTZEE] is not None and self.scores[YahtzeeScoresheet.YAHTZEE] > 0:
                self._yahtzee_bonus += 100
        return update

    def is_marked(self, cat):
        ''' Determines if the given category is marked on this scorsheet.

            cat -- the index of a category on this scoresheet
        '''
        if cat < 0 or cat >= len(self.scores):
            raise ValueError("invalid category index: %d" % cat)
        return self.scores[cat] is not None


    def score(self, cat, roll):
        ''' Returns the score that would be earned on this scoresheet
            by scoring the given roll in the given category.

            cat -- the index of an unused category
            roll -- a complete Yahtzee roll
        '''
        if cat < 0 or cat > YahtzeeScoresheet.YAHTZEE:
            raise ValueError("invalid category index: %d" % cat)
        if self.scores[cat] is not None:
            raise ValueError("category already used: %d" % cat)

        return self.rules[cat](roll)


    def mark(self, cat, roll):
        ''' Updates this scoresheet by scoring the given roll
            in the given category.

            cat -- the index of an unused category
            roll -- a complete Yahtzee roll
        '''
        if cat < 0 or cat > YahtzeeScoresheet.YAHTZEE:
            raise ValueError("invalid category index: %d" % cat)
        if self.scores[cat] is not None:
            raise ValueError("category already used: %d" % cat)

        turn_score = self.rules[cat](roll)
        self.scores[cat] = turn_score
        for tot in self.totals:
            tot(cat, roll, turn_score)
        self._turns += 1


    def grand_total(self):
        ''' Returns the total score, including bonus, marked on this scoresheet.
        '''
        return self._upper_total + self._upper_bonus + self._lower_total + self._yahtzee_bonus


    def game_over(self):
        ''' Determines if this scoresheet has all categories marked.
        '''
        return self._turns == len(self.scores)

    def as_list(self):
        result = list(zip(self.categories, self.scores))
        result.append(('UPPER TOTAL', self._upper_total))
        result.append(('UPPER BONUS', self._upper_bonus))
        result.append(('YAHTZEE BONUS', self._yahtzee_bonus))
        result.append(('GRAND TOTAL', self.grand_total()))
        return result


    def as_state_string(self):
        ''' Returns a string representation of this scoresheet suitable
            as input to StrategyQuery.
        '''
        free = [self.categories[i] for i in range(0, 12) if self.is_marked(i)]
        if self.scores[YahtzeeScoresheet.YAHTZEE] is not None:
            free.append("Y+" if self.scores[YahtzeeScoresheet.YAHTZEE] > 0 else "Y")
        free.append("UP%d" % min(self._upper_total, 63))
        return " ".join(free)



def null_log(sheet, roll, rerolls):
    pass


def stdout_log(sheet, roll, rerolls):
    print(sheet.as_state_string() + "," + "".join(str(x) for x in roll.as_list()) + "," + str(rerolls))

def play_solitaire(choose_dice, choose_category, log=null_log):
    ''' Returns the score earned in one game played with the strategy
        defined by the two given functions.  Each position is logged with
        the given function.

        choose_dice -- a function that takes a scoresheet, roll, and number of
                       rerolls, returns a subroll of the roll
        choose_category -- a function that takes a non-filled scoresheet and
                           a roll and returns the index of an unused category
                           on that scoresheet
        log -- a function that takes a scoresheet, roll, and number of rerolls
    '''
    # start with empty scoresheet

    sheet = YahtzeeScoresheet()
    while not sheet.game_over():
        # do the initial roll
        roll = YahtzeeRoll()
        roll.reroll()

        # reroll twice
        for i in [2, 1]:
            log(sheet, roll, i)

            # choose dice to keep
            keep = choose_dice(sheet, roll, i)
            #print("Rerolls: "+ str(i))
            #print(sheet.as_state_string())
            #print(roll.as_list())
            #print(keep.as_list())
            if not keep.subroll(roll):
                raise ValueError("dice to keep %s not a subset of roll %s" % (keep.as_list(), roll.as_list()))
            keep.reroll()
            roll = keep

        log(sheet, roll, 0)

        # choose category to use and mark it
        cat = choose_category(sheet, roll)
        #print("Rerolls: "+ str(0))
        #print(sheet.as_state_string())
        #print(roll.as_list())
        #print(cat+1)
        sheet.mark(cat, roll)

    return sheet.grand_total()


def print_scoresheet(sheet):
    print("\n".join(str(id) + " " + name + " " + str(score) for (id, (name, score)) in zip(range(1, 18), sheet.as_list())))

class RandomStrategy:
    ''' A strategy that picks a category at the beginning of each
        turn and tries to score in that category.

        This is not intended to be a good strategy.  It averages about 191.
    '''
    def __init__(self):
        self.cat = None
        self.goals = [3, 6, 9, 12, 15, 18, 20, 10, 15, 20, 15, 25, 10]

    def choose_dice(self, sheet, roll, rerolls):
        if rerolls == 2:
            self.pick_random_category(sheet)

        if self.cat >= 0 and self.cat < YahtzeeScoresheet.THREE_KIND:
            return roll.select_all([self.cat + 1])
        elif self.cat in [YahtzeeScoresheet.THREE_KIND, YahtzeeScoresheet.FOUR_KIND, YahtzeeScoresheet.YAHTZEE]:
            return roll.select_for_n_kind(sheet, rerolls)
        elif self.cat == YahtzeeScoresheet.FULL_HOUSE:
            return roll.select_for_full_house()
        elif self.cat == YahtzeeScoresheet.CHANCE:
            return roll.select_for_chance(rerolls)
        else:
            return roll.select_for_straight(sheet)


    def choose_category(self, sheet, roll):
        ''' Returns the free category that minimizes regret.
        '''
        regrets = [(cat, self.goals[cat] - sheet.score(cat, roll))  for cat in range(0, 13) if not sheet.is_marked(cat)]
        return min(regrets, key=lambda x:x[1])[0]


    def pick_random_category(self, sheet):
        self.cat = None
        count = 0
        for c in range(0, 13):
            if not sheet.is_marked(c):
                count += 1
                if random.random() < 1.0 / count:
                    self.cat = c


def choose_dice_interactive(sheet, roll, rerolls):
    if rerolls == 2:
        print_scoresheet(sheet)
    print(roll)
    keep = None
    while keep is None:
        response = input("Select dice to keep (or 'all'):")
        if response == 'all':
            keep = roll
        else:
            try:
                keep = YahtzeeRoll.parse(response)
                if not keep.subroll(roll):
                    keep = None
                    print("select only dice in the current roll")
            except ValueError:
                keep = None
                print("select only dice in the current roll")
    return keep


def choose_category_interactive(sheet, roll):
    print(roll)
    cat = None
    while cat is None:
        try:
            cat = int(input("Choose category by index:")) - 1
            if sheet.is_marked(cat):
                print("select an unused category")
                cat = None
        except ValueError:
            cat = None
            print("select the index of an unused catagory")
    return cat


def evaluate_strategy(n, choose_dice, choose_category, log=null_log):
    ''' Evaluates a strategy by using it for the given number of games
        and returning its average score.

        n -- a positive integer
        choose_dice -- a function that takes a scoresheet, a roll, and a
                       number of rerolls and returns a subroll of the roll
        choose_category -- a function that takes a scoresheet and a roll
                           and returns the index of an unused category on
                           that scoresheet
    '''
    total = 0
    for i in range(n):
        total += play_solitaire(choose_dice, choose_category, log)
    return total / n

def main():
    strat = nn.NNStrategy()
    run = 0
    n=1000
    for i in range(n):
        mean = evaluate_strategy(1, strat.choose_dice, strat.choose_category, stdout_log)
        run += mean
        #print(mean)
    print("Mean: " + str(run/n))

'''
# for training data generation
def main():
    strat = RandomStrategy()
    for i in range(1000):
        mean = evaluate_strategy(1, strat.choose_dice, strat.choose_category, stdout_log)
        #print(mean)
'''


if __name__ == "__main__":
    main()
