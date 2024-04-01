import collections
import pandas as pd
import numpy as np
from model import RNN_model
import torch

global full_dictionary
global full_dictionary_common_letter_sorted
global freq_by_length
global n_gram
global current_dictionary
global history_condition 


def load_model(model_path):
    model = RNN_model(target_dim=26, hidden_units=16)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def build_dictionary(dictionary_file_location):
        text_file = open(dictionary_file_location, "r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary

def init_df(dictionary):
        """
        use words list to generate dictionary frequency
        each key is word length
        each value is a dataframe with column is location of each length
        """
        group_by_length = collections.defaultdict(list)
        for word in dictionary:
            group_by_length[len(word)].append(word)

        res = {}
        for key in group_by_length.keys():
            word_list = group_by_length[key]
            tmp = pd.DataFrame([list(word) for word in word_list])
            tmp.columns = [chr(i + 97) for i in range(tmp.shape[1])]
            res[key] = tmp
        return res
def init_n_gram(n, full_dictionary):
        n_gram = {-1:[]}
        for word in full_dictionary:
            single_word_gram = gen_n_gram(word, n)
            if len(word) not in n_gram:
                n_gram[len(word)] = single_word_gram
            else:
                n_gram[len(word)].extend(single_word_gram)
            n_gram[-1].extend(single_word_gram)
        res = {}
        for key in n_gram.keys():
            res[key] = collections.Counter(n_gram[key])
        return res
def gen_n_gram(word, n):
        n_gram = []
        for i in range(n, len(word)+1):
            if word[i-n:i] not in n_gram:
                n_gram.append(word[i-n:i])
        return n_gram


def get_current_word(target_word,guessed_letters):
        """
        combine target word and guessed letters to generate obscured word
        """
        word_seen = [letter if letter in guessed_letters else "_" for letter in target_word]
        return word_seen

def gen_condition( word):
        tmp = {i: word[i] for i in range(len(word)) if word[i] != "_"}
        condition = {}
        for key, val in tmp.items():
            if val not in condition:
                condition[val] = [key]
            else:
                condition[val].append(key)
        return condition

def update_df(df, condition):
        """
        :param df: dataframe
        each column is one location of a word
        each row is a word
        :param condition: dictionary
        key is letter
        value is which index does this letter appear
        means we only select the words which has letter <value> at index <key>
        note that we don't select words that has letter <value> at other index
        e.g. if condition = {1:'a'}, then "app" is selected while "aha" not
        :return:
        df: updated dataframe
        """
        if len(condition) == 0:
            return df

        for letter, idx in condition.items():
            # find rows satisfy
            # 1. corresponding column == val
            # 2. all the other column != val
            query = ""
            for i in range(df.shape[1]):
                col = df.columns.values[i]
                if i in idx:
                    query += "{} == '{}' and ".format(col, letter)
                else:
                    query += "{} != '{}' and ".format(col, letter)
            query = query[:-5]
            new_df = df.query(query)
            df = new_df.copy()
            del new_df
        return df

def freq_from_df(df):
        key, cnt = np.unique(df.values, return_counts=True)
        freq = [(k, val) for k, val in zip(key, cnt)]
        return sorted(freq, key=lambda x: x[1], reverse=True)

def find_by_gram(all_gram, pre=None, suff=None):
        selected_gram = []
        for key, val in all_gram.items():
            if (pre is not None) and (key[0] == pre):
                selected_gram.append((key[1], val))
            if (suff is not None) and (key[1] == suff):
                selected_gram.append((key[0], val))

        res = {}
        for letter, freq in selected_gram:
            if letter not in res:
                res[letter] = freq
            else:
                res[letter] += freq
        final_res = [(key, val) for key, val in res.items()]
        return sorted(final_res, key=lambda x: x[1], reverse=True)

def encode_obscure_words(word):
        word_idx = [ord(i) - 97 if i != "_" else 26 for i in word]
        obscured_word = np.zeros((len(word), 27), dtype=np.float32)
        for i, j in enumerate(word_idx):
            obscured_word[i, j] = 1
        return obscured_word

def lessgo():
    global full_dictionary
    global full_dictionary_common_letter_sorted
    global freq_by_length
    global n_gram
    global current_dictionary
    global history_condition
    full_dictionary=build_dictionary('training.txt')
    full_dictionary_common_letter_sorted = collections.Counter("".join(full_dictionary)).most_common()
    freq_by_length = init_df(full_dictionary)
    n_gram = init_n_gram(2, full_dictionary)
    current_dictionary = []
    history_condition = []

model=load_model('model_humara.pth')

def guess( word,guessed_letters):  # word input example: "_ p p _ e "
        # divided word group by word length
        global full_dictionary
        global full_dictionary_common_letter_sorted
        global freq_by_length
        global n_gram
        global current_dictionary
        global history_condition
        lessgo()
        all_words = freq_by_length[len(word)]
        all_gram = n_gram[-1]
        # all_gram = self.n_gram[len(word)]

        # first guess by letter frequency in each word group
        new_condition = gen_condition(word)

        if len(history_condition) != 0 and new_condition != history_condition[-1]:
            history_condition.append(new_condition)

        all_words = update_df(all_words, new_condition)
        freq = freq_from_df(all_words)
        for i in range(len(freq)):
            if freq[i][0] not in guessed_letters:
                return freq[i][0]

        # if we run out of letters, use 2-gram to predict
        for i in range(len(word)):
            if word[i] == "_":  # this is where we should apply 2-gram
                if (i == 0) or (word[i-1] == "_"):
                    guess = find_by_gram(all_gram, pre=None, suff=word[i+1])
                elif (i == len(word) - 1) or (word[i+1] == "_"):
                    guess = find_by_gram(all_gram, pre=word[i-1], suff=None)
                else:
                    guess = find_by_gram(all_gram, pre=word[i-1], suff=word[i+1])
                break

        for i in range(len(guess)):
            if guess[i][0] not in guessed_letters:
                return guess[i][0]
        # if we run out of 2-gram, use LSTM model to predict
        # the benefit of LSTM model is to add more uncertainty to the prediction
        guessed_multi_hot = np.zeros(26, dtype=np.float32)
        for letter in guessed_letters:
            idx = ord(letter) - 97
            guessed_multi_hot[idx] = 1.0

        obscure_words = encode_obscure_words(word)
        obscure_words = np.asarray(obscure_words)
        guessed_multi_hot = np.asarray(guessed_multi_hot)
        obscure_words = torch.from_numpy(obscure_words)
        guessed_multi_hot = torch.from_numpy(guessed_multi_hot)
        out = model(obscure_words, guessed_multi_hot)
        guess = torch.argmax(out, dim=2).item()
        guess = chr(guess + 97)
        return guess


