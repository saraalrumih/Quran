import re
import collections
import time
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def remove_adjacent_duplicates(string):
    prev_w = ''
    new_string = []
    for word in string.split(" "):
        if prev_w == word:
            continue
        else:
            new_string.append(word)
            prev_w = word
    return ' '.join(new_string)
report = ""
def mistake_detection(pred, ref):
    diacritics = ['ّ', 'ْ', 'ٌ', 'ُ', 'ٍ', 'ِ', 'ً', 'َ', 'َّ', 'ِّ', 'ُّ']
    char_similarity_count = 0
    char_dissimilarity_count = 0
    dissimilarity_list = list()
    diacritics_dissimilarity = list()
    global report
    report = ""
    # remove extra spaces
    pred = re.sub(' {2,}', ' ', pred.strip())
    # ignore adjacent duplicate words
    pred = remove_adjacent_duplicates(pred)

    p_no_d = re.sub('[ًٌٍَُِّْ]', '', pred.strip())
    p_no_d = remove_adjacent_duplicates(p_no_d)
    r_no_d = re.sub('[ًٌٍَُِّْ]', '', ref)

    # check if the user did jump a verse
    if similar(p_no_d, r_no_d)<0.6:
        # print(pred, " is not the intended verse")
        report += pred+ " is not the intended verse"
        return report

    # check if the user correctly recite the verse
    if similar(pred, ref) == 1:
        # print("Correct recitation")
        report += "Correct recitation"
        return report
    # check no missing or adding a word
    if(len(p_no_d.split(" "))<len(ref.split(" "))):
        # print("Deletion mistake")
        report += "Deletion mistake  \n"
        candidate_missed_words = r_no_d.split(" ")

        for p_word, r_word in zip(pred.split(" "), ref.split(" ")):
            if similar(p_word, r_word) > 0.5:
                rw_no_d = re.sub('[ًٌٍَُِّْ]', '', r_word)
                candidate_missed_words.remove(rw_no_d)
                check_characters(p_word, r_word)
            else:
                for word in candidate_missed_words:
                    if similar(p_word, word) > 0.5:
                        # print(p_word, " the intended word, not in the correct location")
                        report += p_word+ " the intended word, not in the correct location  \n"
                        candidate_missed_words.remove(word)
        if len(candidate_missed_words)==len(r_no_d.split(" ")):
            # print("You are correct!, no missed words")
            report += "You are correct!, no missed words  \n"
        else:
            # print("You missed the words: ", candidate_missed_words)
            report += "You missed the words: "+ str(candidate_missed_words) +"  \n"
    elif (len(p_no_d.split(" "))>len(ref.split(" "))):
        # print("Addition mistake")
        report += "Addition mistake  \n"
        # which word is extra?
        # ignore when user repeat the word twice or more to remember the next, e.g., قل هو هو الله
        previous_word = ''
        word_counts = collections.Counter(p_no_d.split(" "))
        for r_word in r_no_d.split(" "):
            for key in word_counts.keys():
                if similar(key, r_word) > 0.8:
                    word_counts[key] -= 1
        # print("The extra words are: ", word_counts)
        # print("The extra words are: ", [w for w in word_counts.keys() if word_counts[w]>0])
        report += "The extra words are: " + str([w for w in word_counts.keys() if word_counts[w]>0]) + "  \n"
    else:
        # pred length = ref length
        # print("No missing or extra words")
        # check for replaced word
        replaced_words = []
        for p_word, r_word in zip(pred.split(" "), ref.split(" ")):
            if similar(p_word, r_word) > 0.5:
                # print(p_word, " is the intended word")
                check_characters(p_word, r_word)
            else:
                # print(p_word, "is not the intended word")
#                report += p_word+ "is not the intended word  \n"
                check_characters(p_word, r_word)
                replaced_words.append({"target":r_word, "recognized":p_word})
        # check character + diacritics missing, replace, add
        if replaced_words:
            # print("The substituted words are: ", replaced_words)
            report += "The substituted words are: "+ str(replaced_words) + "  \n"
            # for word in replaced_words:
            #     check_characters(word['recognized'], word['target'])
    rreport = str(report)
    del report
    return rreport


def check_characters(p_word, r_word):
    previous = []
    nodublicatestring = []
    global report

    if similar(p_word, r_word) == 1:
        # print("Correct word recitation")
        return
    elif similar(p_word, r_word) == 0:
        # print("Incorrect word recitation of ", r_word)
        report += "Incorrect word recitation of " + r_word +"  \n"
        return
    elif len(p_word)<len(r_word):
        # Missing characters/diacritics
        missed_char = [l for l in r_word if l not in p_word]
        # print("The missed characters/diacritics are: ", missed_char)
        report += "The missed characters/diacritics in "+ p_word +"are: "+ str(missed_char) + "  \n"
        return
        # TO DO: check for replaced characters
    elif len(p_word)>len(r_word):
        # Extra characters/diacritics
        char_counts = collections.Counter(p_word)
        for r_ch in r_word:
            for key in char_counts.keys():
                if similar(key, r_ch) == 1:
                    char_counts[key] -= 1
        # print("The extra characters/diacritics in ", p_word ,"are: ", [ch for ch in char_counts.keys() if char_counts[ch] > 0])
        report += "The extra characters/diacritics in "+ p_word +"are: "+ str([ch for ch in char_counts.keys() if char_counts[ch] > 0]) + "  \n"
        return
    else:
        # Replace characters/diacritics
        replaced_char = []
        for p_l, r_l in zip(p_word, r_word):
            if similar(p_l, r_l) != 1:
                # print(p_l, " is not the intended character")
#                report += p_l+ " is not the intended character  \n"
                replaced_char.append({"target": r_l, "recognized": p_l})
        if replaced_char:
            # print("The substituted characters/diacritics in ", p_word," are: ", replaced_char)
            report += "The substituted characters/diacritics in "+ p_word+" are: "+ str(replaced_char)+"  \n"
            
