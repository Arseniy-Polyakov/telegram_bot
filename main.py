import re
import timeit
import logging
import json
logging.basicConfig(level=logging.INFO, filename="text_evaluation_script/logging.log", filemode="w")
logging.debug("A DEBUG Message")
logging.info("An INFO")
logging.warning("A WARNING")
logging.error("An ERROR")
logging.critical("A message of CRITICAL severity")

import nltk
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

with open("text_evaluation_script/input_text.txt", "rt", encoding="utf-8") as file:
    input_text = file.read()

#PREPRPOCESSING 
text_preprocessed = re.sub(r"[^A-Za-z0-9\- ]", "", input_text)
text_preprocessed = text_preprocessed.lower()
logging.info(text_preprocessed)
text_tokenized = nltk.word_tokenize(text_preprocessed)
text_lemmatized = [lemmatizer.lemmatize(token) for token in text_tokenized]
logging.info(" TEXT LEMMATIZED " + str(text_lemmatized))

#NER 
pos_tags = nltk.pos_tag(text_lemmatized)
entities = [token for token in pos_tags if token[1] == "NNP" or token[1] == "NNPS"]
logging.info(" ENTITIES " + str(entities))

#GENERAL STATISTICS
sentences = nltk.sent_tokenize(input_text)
sentences_count = len(sentences)
words_count = len(text_lemmatized)
logging.info(" I. GENERAL STATISTICS. 1. THE NUMBER OF SENTENCES " + str(sentences_count))
logging.info(" I. GENERAL STATISTICS. 2. THE NUMBER OF WORDS " + str(words_count))

with open("text_evaluation_script/stopwords.txt", "rt", encoding="utf-8") as file:
    file = file.read()
    stopwords = file.split()

words_without_stopwords = [word for word in text_lemmatized if word not in stopwords]
words_without_stopwords_count = len(words_without_stopwords)
logging.info(" I. GENERAL STATISTICS. 3. THE NUMBER OF WORDS WITHOUT STOPWORDS " + str(words_without_stopwords_count))

average_words_per_sent = round(words_count / sentences_count, 2)
average_words_without_stopwords_per_sent = round(words_without_stopwords_count / sentences_count, 2)
logging.info(" I. GENERAL STATISTICS. 4. THE AVERAGE NUMBER OF WORDS IN THE SENTENCE " + str(average_words_per_sent))
logging.info(" I. GENERAL STATISTICS. 5. THE AVERAGE NUMBER OF WORDS (WITHOUR STOPWORDS) IN THE SENTENCE " + str(average_words_without_stopwords_per_sent))

data_general = {"Sentences": sentences_count, 
                             "Words": words_count, 
                             "Words without stopwords": words_without_stopwords_count, 
                             "Words per sentence": average_words_per_sent, 
                             "Words without stopwords per sentence": average_words_without_stopwords_per_sent}
logging.info(" I. GENERAL STATISTICS " + str(data_general))

#PHONOLOGICAL LEVEL
def syllables_counting(word):
    vowels = ["a", "o", "e", "y", "u", "i"]
    digrafs_and_trigrafs = ["ai", "ay", "ea", "ee", "ei", "ey", "oa", "oe", "oo", "ou", 
            "ow", "ua", "ue", "ui", "uy", "eau", "iou", "aye", "iou"]
    vowels_count = len([symbol for symbol in word if symbol in vowels])
    digrafs_and_trigrafs_count = len([item for item in digrafs_and_trigrafs if item in word])
    limitations = ["he", "she", "me", "we", "cafe", "apostrophe"]
    final_count = vowels_count - digrafs_and_trigrafs_count
    if word.endswith("e") and word not in limitations: 
        final_count -= 1
    return final_count

syllables_count = sum([syllables_counting(word) for word in words_without_stopwords])
average_syllables_per_word = round(syllables_count / words_without_stopwords_count, 2)
average_syllables_per_sent = round(syllables_count / sentences_count, 2)
logging.info(" II. PHONOLOGICAL LEVEL. 1. THE NUMBER OF SYLLABLES IN THE TEXT "+ str(syllables_count))
logging.info(" II. PHONOLOGICAL LEVEL. 2. THE AVERAGE NUMBER OF SYLLABLES IN THE WORD " + str(average_syllables_per_word))
logging.info(" II. PHONOLOGICAL LEVEL. 3. THE AVERAGE NUMBER OF SYLLABLES IN THE SENTENCE " + str(average_syllables_per_sent))


syllables = {"1-syllable": 0, 
             "2-syllables": 0, 
             "3-syllables": 0, 
             "4-syllables": 0, 
             "5 and more syllables": 0}

for word in words_without_stopwords:
    if syllables_counting(word) == 1: 
        syllables["1-syllable"] += 1
    elif syllables_counting(word) == 2:
        syllables["2-syllables"] += 1
    elif syllables_counting(word) == 3:
        syllables["3-syllables"] += 1
    elif syllables_counting(word) == 4:
        syllables["4-syllables"] += 1
    else:
        syllables["5 and more syllables"] += 1

logging.info(" II. PHONOLOGICAL LEVEL. 4. THE NUMBER OF 1-SYLLABLE, 2-SYLLABLES, 3-SYLLABLES, 4-SYLLABLES" + str(syllables))

one_syllable_average = round(syllables["1-syllable"] / sentences_count, 2)
two_syllables_average = round(syllables["2-syllables"] / sentences_count, 2)
three_syllables_average = round(syllables["3-syllables"] / sentences_count, 2)
four_syllables_average = round(syllables["4-syllables"] / sentences_count, 2)
five_and_more_syllables_average = round(syllables["5 and more syllables"] / sentences_count, 2)

data_phonological = {"Syllables": syllables_count, 
                     "Average syllables per word": average_syllables_per_word, 
                     "Average syllables per sentences": average_syllables_per_sent, 
                     "1-syllable words": syllables["1-syllable"], 
                     "2-syllables words": syllables["2-syllables"], 
                     "3-syllables words": syllables["3-syllables"], 
                     "4-syllables words": syllables["4-syllables"], 
                     "5 and more syllables": syllables["5 and more syllables"], 
                     "1-syllable per sentence": one_syllable_average, 
                     "2-syllables per sentence": two_syllables_average, 
                     "3-syllables per sentence": three_syllables_average, 
                     "4-syllables per sentence": four_syllables_average, 
                     "5 and more syllables per sentence": five_and_more_syllables_average}

polysyllabic = syllables["3-syllables"] + syllables["4-syllables"] + syllables["5 and more syllables"]
logging.info(" II. PHONOLOGICAL LEVEL " + str(data_phonological))

#GRAMMATICAL LEVEL
pos_tagging = nltk.pos_tag(words_without_stopwords, tagset="universal")
logging.info(" III. GRAMMATICAL LEVEL. POS-TAGGING " + str(pos_tagging)) 

data_grammatical = {"Nouns": 0, 
                    "Adjectives": 0, 
                    "Verbs": 0, 
                    "Adverbs": 0, 
                    "Pronouns": 0, 
                    "Numerals": 0,
                    "Functional parts of speech": 0, 
                    "Nominativity": 0, 
                    "Descriptivity": 0}

for tag in pos_tagging:
    if tag[1] == "NOUN":
        data_grammatical["Nouns"] += 1
    elif tag[1] == "ADJ":
        data_grammatical["Adjectives"] += 1
    elif tag[1] == "VERB": 
        data_grammatical["Verbs"] += 1 
    elif tag[1] == "ADV":
        data_grammatical["Adverbs"] += 1 
    elif tag[1] == "PRON":
        data_grammatical["Pronouns"] += 1
    elif tag[1] == "NUM":
        data_grammatical["Numerals"] += 1
    else:
        data_grammatical["Functional parts of speech"] += 1

data_grammatical["Nominativity"] += round(data_grammatical["Nouns"] / words_without_stopwords_count, 2)
data_grammatical["Descriptivity"] += round(data_grammatical["Adjectives"] / words_without_stopwords_count, 2)

logging.info(" III. GRAMMATICAL LEVEL. DATA " + str(data_grammatical))

#LEXICAL LEVEL 
with open("text_evaluation_script/words.json", "rt", encoding="utf-8") as file:
    lexis_words = json.load(file)

with open("text_evaluation_script/collocations.json", "rt", encoding="utf-8") as file:
    lexis_collocations = json.load(file)

data_lexical = {"A1 words in the text": 0, 
                "A2 words in the text": 0, 
                "B1 words in the text": 0, 
                "B2 words in the text": 0, 
                "C1 words in the text": 0,
                "Named Entities in the text": 0,  
                "A1 words per sentence": 0, 
                "A2 words per sentence": 0,
                "B1 words per sentence": 0,
                "B2 words per sentence": 0,
                "C1 words per sentence": 0, 
                "Named Entities per sentence": 0,
                "A1 collocations in the text": 0, 
                "A2 collocations in the text": 0, 
                "B1 collocations in the text": 0, 
                "B2 collocations in the text": 0, 
                "C1 collocations in the text": 0, 
                "A1 collocations per sentence": 0, 
                "A2 collocations per sentence": 0,
                "B1 collocations per sentence": 0,
                "B2 collocations per sentence": 0,
                "C1 collocations per sentence": 0, 
                "TTR (Type Token Ratio)": 0, 
                "RTTR (Root Type Token Ratio)": 0, 
                "CTTR (Corrected Type Token Ratio)": 0,}

data_lexical["A1 words in the text"] += len([word for word in lexis_words.items() if word[0] in words_without_stopwords and word[1] == "A1"])
data_lexical["A2 words in the text"] += len([word for word in lexis_words.items() if word[0] in words_without_stopwords and word[1] == "A2"])
data_lexical["B1 words in the text"] += len([word for word in lexis_words.items() if word[0] in words_without_stopwords and word[1] == "B1"])
data_lexical["B2 words in the text"] += len([word for word in lexis_words.items() if word[0] in words_without_stopwords and word[1] == "B2"])
data_lexical["C1 words in the text"] += len([word for word in lexis_words.items() if word[0] in words_without_stopwords and word[1] == "C1"])
data_lexical["Named Entities in the text"] += len(entities)

data_lexical["A1 words per sentence"] += round(data_lexical["A1 words in the text"] / sentences_count, 2)
data_lexical["A2 words per sentence"] += round(data_lexical["A2 words in the text"] / sentences_count, 2)
data_lexical["B1 words per sentence"] += round(data_lexical["B1 words in the text"] / sentences_count, 2)
data_lexical["B2 words per sentence"] += round(data_lexical["B2 words in the text"] / sentences_count, 2)
data_lexical["C1 words per sentence"] += round(data_lexical["C1 words in the text"] / sentences_count, 2)
data_lexical["Named Entities per sentence"] += round(len(entities) / sentences_count, 2)

data_lexical["A1 collocations in the text"] += len([word for word in lexis_collocations.items() if word[0] in text_preprocessed and word[1] == "A1"])
data_lexical["A2 collocations in the text"] += len([word for word in lexis_collocations.items() if word[0] in text_preprocessed and word[1] == "A2"])
data_lexical["B1 collocations in the text"] += len([word for word in lexis_collocations.items() if word[0] in text_preprocessed and word[1] == "B1"])
data_lexical["B2 collocations in the text"] += len([word for word in lexis_collocations.items() if word[0] in text_preprocessed and word[1] == "B2"])
data_lexical["C1 collocations in the text"] += len([word for word in lexis_collocations.items() if word[0] in text_preprocessed and word[1] == "C1"])

data_lexical["A1 collocations per sentence"] += round(data_lexical["A1 collocations in the text"] / sentences_count, 2)
data_lexical["A2 collocations per sentence"] += round(data_lexical["A2 collocations in the text"] / sentences_count, 2)
data_lexical["B1 collocations per sentence"] += round(data_lexical["B1 collocations in the text"] / sentences_count, 2)
data_lexical["B2 collocations per sentence"] += round(data_lexical["B2 collocations in the text"] / sentences_count, 2)
data_lexical["C1 collocations per sentence"] += round(data_lexical["C1 collocations in the text"] / sentences_count, 2)

data_lexical["TTR (Type Token Ratio)"] += round(len(set(text_preprocessed)) / words_count, 2)
data_lexical["RTTR (Root Type Token Ratio)"] += round(len(set(text_preprocessed)) / (words_count ** 0.5), 2)
data_lexical["CTTR (Corrected Type Token Ratio)"] += round(len(set(text_preprocessed)) / ((2 * words_count) ** 0.5), 2)

logging.info(" IV. LEXICAL LEVEL " + str(data_lexical))

#STATISTICAL METRICS
data_statistical = {"Flesh Reading Ease (FRE)": round(206.835 - (1.015 * (words_count / sentences_count)) - (84.6 * (syllables_count / words_count)), 2), 
                    "Flesh Kincaid Grade Level (FKGL)": round((0.39 * (words_count / sentences_count)) + (11.8 * (syllables_count / words_count)) - 15.59, 2),
                    "LIX": round((words_count / sentences_count) + (polysyllabic * 100 / words_count), 2), 
                    "SMOG": round(3 + (polysyllabic ** 0.5), 2)}

logging.info(" V. STATISTICAL METRICS " + str(data_statistical))

with open("text_evaluation_script/topics.json", "rt", encoding="utf-8") as file:
    topics = json.load(file)

topics_count = {"animals": 0, 
                "appearance": 0, 
                "communication": 0, 
                "culture": 0, 
                "food_and_drink": 0, 
                "functions": 0, 
                "health": 0, 
                "homes_and_buildings": 0, 
                "leisure": 0, 
                "notions": 0, 
                "people": 0, 
                "politics_and_society": 0, 
                "science_and_technology": 0, 
                "sports": 0, 
                "the_natural_world": 0, 
                "time_and_space": 0, 
                "travel": 0,
                "work_and_business": 0}

for token in topics.items():
    if token[0] in text_lemmatized:
        topics_count[topics[token[0]]] += 1

#TOPIC MODELING
topics_count = dict(sorted(topics_count.items(), key = lambda x: x[1], reverse=True)[:5]) 
the_most_popular_topics = list(topics_count.keys())

logging.info(" TOPIC MODELING " + str(topics_count))
logging.info(" TOP 5 THE MOST COMMON TOPICS " + str(the_most_popular_topics))

logging.info(" TIME SCRIPT: " + str(timeit.timeit()) + " SEC ")