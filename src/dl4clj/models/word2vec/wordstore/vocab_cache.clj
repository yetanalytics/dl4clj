(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/models/word2vec/wordstore/VocabCache.html"}
  dl4clj.models.word2vec.wordstore.vocab-cache
  (:import [org.deeplearning4j.models.word2vec.wordstore VocabCache]))

(defn add-token
  "Adds a token to the cache"
  [^VocabCache this element]
  (.addToken this element))

(defn add-word-to-index 
  ""
  [^VocabCache this index word]
  (.addWordToIndex this (int index) word))

(defn contains-word
  "Returns true if the cache contains the given word"
  [^VocabCache this word]
  (.containsWord this word))

(defn doc-appeared-in
  "Count of documents a word appeared in"
  [^VocabCache this word]
  (.docAppearedIn this word))

(defn element-at-index
  "Returns SequenceElement at the given index or null"
  [^VocabCache this index]
  (.elementAtIndex this (int index)))

(defn has-token
  "Returns whether the cache contains this token or not"
  [^VocabCache this token]
  (.hasToken this token))

(defn import-vocabulary
  "imports vocabulary"
  [^VocabCache this vocab-cache]
  (.importVocabulary this vocab-cache))

(defn increment-doc-count
  "Increment the document count"
  [^VocabCache this word how-much]
  (.incrementDocCount this word, (int how-much)))

(defn increment-total-doc-count
  "Increment the doc count"
  [^VocabCache this]
  (.incrementTotalDocCount this))

(defn increment-total-doc-count
  "Increment the doc count"
  [^VocabCache this by]
  (.incrementTotalDocCount this (int by)))

(defn increment-word-count
  "Increment the count for the given word"
  [^VocabCache this word]
  (.incrementWordCount this word))

(defn increment-word-count
  "Increment the count for the given word by the amount increment"
  [^VocabCache this word increment]
  (.incrementWordCount this word (int increment)))

(defn index-of
  "Returns the index of a given word"
  [^VocabCache this word]
  (.indexOf this word))

(defn load-vocab
  "Load vocab"
  [^VocabCache this]
  (.loadVocab this ))

(defn num-words
  "Returns the number of words in the cache"
  [^VocabCache this]
  (.numWords this ))

(defn remove-element
  "Removes element with specified label from vocabulary Please note: Huffman index should be updated after element removal"
  [^VocabCache this label]
  (.removeElement this label))

(defn save-vocab
  "Saves the vocab: this allow for reuse of word frequencies"
  [^VocabCache this]
  (.saveVocab this ))

(defn set-count-for-doc
  "Set the count for the number of documents the word appears in"
  [^VocabCache this word count]
  (.setCountForDoc this word (int count)))

(defn token-for
  "Returns the token (again not necessarily in the vocab) for this word"
  [^VocabCache this word]
  (.tokenFor this word))

(defn tokens
  "All of the tokens in the cache, (not necessarily apart of the vocab)"
  [^VocabCache this]
  (.tokens this ))

(defn total-number-of-docs
  "Returns the total of number of documents encountered in the corpus"
  [^VocabCache this]
  (.totalNumberOfDocs this ))

(defn total-word-occurrences
  "The total number of word occurrences"
  [^VocabCache this]
  (.totalWordOccurrences this ))

(defn update-words-occurencies
  "Updates counters"
  [^VocabCache this]
  (.updateWordsOccurencies this ))

(defn vocab-exists
  "Vocab exists already"
  [^VocabCache this]
  (.vocabExists this ))

(defn vocab-words
  "Returns all of the vocab word nodes"
  [^VocabCache this]
  (.vocabWords this ))

(defn ord-at-index
  "Returns the word contained at the given index or null"
  [^VocabCache this index]
  (.ordAtIndex this (int index)))

(defn word-for 
  ""
  [^VocabCache this word]
  (.wordFor this word))

(defn word-frequency
  "Returns the number of times the word has occurred"
  [^VocabCache this word]
  (.wordFrequency this word))

(defn words
  "Returns all of the words in the vocab"
  [^VocabCache this]
  (.words this ))
