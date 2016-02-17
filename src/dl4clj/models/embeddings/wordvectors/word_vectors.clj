(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/wordvectors/WordVectors.html"}
  dl4clj.models.embeddings.wordvectors.word-vectors
  (:import [org.deeplearning4j.models.embeddings.wordvectors WordVectors]))

(defmulti accuracy
  "Accuracy based on questions which are a space separated list of strings where the first
word is the query word, the next 2 words are negative, and the last word is the
predicted word to be nearest"
  (fn [this string] (type this)))

(defmulti get-word-vector 
  "Get the word vector for a given matrix" 
  (fn [this string] (type this)))

(defmulti get-word-vector-matrix
  "Get the word vector for a given matrix"
  (fn [this string] (type this)))

(defmulti get-word-vector-matrix-normalized 
  "Returns the word vector divided by the norm2 of the array"
  (fn [this string] (type this)))

(defmulti has-word
  "Returns true if the model has this word in the vocab"
  (fn [this string] (type this)))

(defmulti index-of
  (fn [this string] (type this)))

(defmulti lookup-table
  "Lookup table for the vectors"
  (fn [this] (type this)))

(defmulti similarity
  "Returns the similarity of 2 words"
  (fn [this string1 string2] (type this)))

(defmulti similar-words-in-vocab-to 
  "Find all words with a similar characters in the vocab"
  (fn [this string ^Double accuracy] (type this)))

(defmulti vocab
  "Vocab for the vectors"
  (fn [this] (type this)))
        
(defmulti words-nearest
  "Words nearest based on positive and negative"
  (fn [this & more] (type this)))

(defmulti words-nearest-sum
  (fn [this & more] (type this)))
